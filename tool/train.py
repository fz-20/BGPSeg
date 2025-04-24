import os, sys
import trimesh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import random
import numpy as np
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
from util.ABCPrimitive import ABCPrimitive_Dataset
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn, collate_fn_region
from util.logger import get_logger
from util.loss_util import compute_type_loss, compute_embedding_loss_boundary, mean_shift_gpu, compute_iou
from util.lr import MultiStepWithWarmup, PolyLR

from ctypes import *
BGPC = cdll.LoadLibrary('lib/cpp/build/libBGPC.so')


def get_parser():
    parser = argparse.ArgumentParser(description='Boundary-Guided Primitive Instance Segmentation of Point Clouds')
    parser.add_argument('--config', type=str, default='config/ABCPrimitive/ABCPrimitive.yaml', help='config file')
    parser.add_argument('opts', help='see config/ABCPrimitive/ABCPrimitive.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.arch == 'BGPSeg':
        from model.BGPSeg import BoundaryPredictor as BoundaryModel
        from model.BGPSeg import BGPSeg as Model
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    boundarymodel = BoundaryModel(in_channels=args.fea_dim)
    model = Model(in_channels=args.fea_dim+1, num_classes=args.classes)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda() 

    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':     
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        # logger.info(boundarymodel)
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            boundarymodel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(boundarymodel).cuda()
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        boundarymodel = torch.nn.parallel.DistributedDataParallel(
            boundarymodel,
            device_ids=[gpu],
            find_unused_parameters=True
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu],
            find_unused_parameters=True
        )

    else:
        boundarymodel = torch.nn.DataParallel(boundarymodel.cuda())
        model = torch.nn.DataParallel(model.cuda())


    if args.boundaryweight:
        if os.path.isfile(args.boundaryweight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.boundaryweight))
            checkpoint = torch.load(args.boundaryweight)
            boundarymodel.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.boundaryweight))
        else:
            raise ValueError("=> no weight found at '{}'".format(args.boundaryweight))
    else:
        raise ValueError("Please enter the boundary model path")

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']

            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 'ABCPrimitive':
        train_data = ABCPrimitive_Dataset(split='train', data_root=args.data_root, loop=args.train_loop)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)

    if args.evaluate:
        if args.data_name == 'ABCPrimitive':
            val_data = ABCPrimitive_Dataset(split='val', data_root=args.data_root, loop=args.val_loop)
        else:
            raise ValueError("The dataset {} is not supported.".format(args.data_name))
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_region)

    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
            warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == 'MultiStep':
        assert args.scheduler_update == 'epoch'
        milestones = [int(x) for x in args.milestones.split(",")] if hasattr(args, "milestones") else [int(args.epochs*0.4), int(args.epochs*0.8)]
        gamma = args.gamma if hasattr(args, 'gamma') else 0.1
        if main_process():
            logger.info("scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(args.scheduler_update, milestones, gamma))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) 
    elif args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    elif args.scheduler == 'Cosine':
        if main_process():
            logger.info("scheduler: Cosine. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*iter_per_epoch)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:    
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        loss_train, BA_emb_loss_train, type_loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, boundarymodel, criterion, optimizer, epoch, scaler, scheduler)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('BA_emb_loss_train', BA_emb_loss_train, epoch_log)
            writer.add_scalar('type_loss_train', type_loss_train, epoch_log)
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        is_eval = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            is_eval = True

        if is_eval:
            seg_miou, label_miou, loss_val, BA_emb_loss_val, type_loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, boundarymodel, criterion)

            if main_process():
                writer.add_scalar('BA_emb_loss_val', BA_emb_loss_val, epoch_log)
                writer.add_scalar('type_loss_val', type_loss_val, epoch_log)
                writer.add_scalar('seg_miou', seg_miou, epoch_log)
                writer.add_scalar('label_miou', label_miou, epoch_log)
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = seg_miou > best_iou
                best_iou = max(best_iou, seg_miou)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou}, filename)
            if is_best:
                logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')
                shutil.copyfile(filename, args.save_path + '/model/model_{}.pth'.format(epoch_log))

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Seg mIou: %.3f' % (best_iou))


def train(train_loader, model, boundarymodel, criterion, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    BA_emb_loss_meter = AverageMeter()
    type_loss_meter = AverageMeter()

    boundarymodel.eval()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (fn, coord, normals, boundary, label, semantic, param, offset, edges, dse_edges) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)
        coord, normals, boundary, label, semantic, param, offset, edges, dse_edges = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), boundary.cuda(non_blocking=True), \
                    label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True), edges.cuda(non_blocking=True), dse_edges.cuda(non_blocking=True)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            # fn = "visual/boundary_fea_cache/train/Us_{}.pt".format(fn[:8])
            # if os.path.exists(fn):
            #     # print("{} Us find cache".format(fn))
            #     boundary_pred = torch.load(fn).cuda()  # [N, 2]
            #     # ent = torch.load(fn_ent)
            #     # print(v.shape, ent.shape)
            # else:
            #     boundary_pred = boundarymodel([coord, normals, offset])
            #     torch.save(boundary_pred, fn)

            # Boundary Predictor
            boundary_pred = boundarymodel([coord, normals, offset])

            # Boundary-Guided Feature Extractor
            primitive_embedding, type_per_point = model([coord, normals, offset], edges, boundary_gt = boundary.int(), boundary_pred = boundary_pred)

            assert type_per_point.shape[1] == args.classes

            # compute loss
            BA_emb_loss, pull_loss, push_loss = compute_embedding_loss_boundary(coord, boundary, primitive_embedding, label, offset)
            type_loss = compute_type_loss(type_per_point, semantic, criterion)

            loss = BA_emb_loss + args.type_loss_weight * type_loss
            
        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = semantic.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(type_per_point.max(1)[1], semantic, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        
        # All Reduce loss
        if args.multiprocessing_distributed:
            dist.all_reduce(BA_emb_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(type_loss.div_(torch.cuda.device_count()))

        BA_emb_loss_, type_loss_ = BA_emb_loss.data.cpu().numpy(), type_loss.data.cpu().numpy()
        BA_emb_loss_meter.update(BA_emb_loss_.item())
        type_loss_meter.update(type_loss_.item())

        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'BA_emb_loss {BA_emb_loss_meter.val:.4f} '
                        'Type_Loss {type_loss_meter.val:.4f} '
                        'Acc {accuracy:.4f} '
                        'Lr: {lr}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          BA_emb_loss_meter=BA_emb_loss_meter,
                                                          type_loss_meter=type_loss_meter,
                                                          accuracy=accuracy,
                                                          lr=lr))
        if main_process():
            writer.add_scalar('BA_emb_loss_train_batch', BA_emb_loss_meter.val, current_iter)
            writer.add_scalar('type_loss_train_batch', type_loss_meter.val, current_iter)
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))

    return loss_meter.avg, BA_emb_loss_meter.avg, type_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, boundarymodel, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    BA_emb_loss_meter = AverageMeter()
    type_loss_meter = AverageMeter()
    seg_iou_meter = AverageMeter()
    type_iou_meter = AverageMeter()

    torch.cuda.empty_cache()

    boundarymodel.eval()
    model.eval()
    end = time.time()
    for i, (fn, coord, normals, boundary, label, semantic, param, offset, edges, dse_edges, face, F_offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, normals, boundary, label, semantic, param, offset, edges, dse_edges = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), boundary.cuda(non_blocking=True), \
                    label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True), edges.cuda(non_blocking=True), dse_edges.cuda(non_blocking=True)

        if semantic.shape[-1] == 1:
            semantic = semantic[:, 0]  # for cls
        
        with torch.no_grad():
            # fn = "visual/boundary_fea_cache/val/Us_{}.pt".format(fn[:8])
            # if os.path.exists(fn):
            #     # print("{} Us find cache".format(fn))
            #     boundary_pred = torch.load(fn).cuda()  # [N, 2]
            #     # ent = torch.load(fn_ent)
            #     # print(v.shape, ent.shape)
            # else:
            #     boundary_pred = boundarymodel([coord, normals, offset])
            #     torch.save(boundary_pred, fn)
            boundary_pred = boundarymodel([coord, normals, offset])

            primitive_embedding, type_per_point = model([coord, normals, offset], edges, boundary_gt=boundary.int(), boundary_pred = boundary_pred, is_train=False)

            BA_emb_loss, pull_loss, push_loss = compute_embedding_loss_boundary(coord, boundary, primitive_embedding, label, offset)
            type_loss = compute_type_loss(type_per_point, semantic, criterion)

            loss = BA_emb_loss + args.type_loss_weight * type_loss

        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = semantic.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(type_per_point.max(1)[1], semantic, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        
        softmax = torch.nn.Softmax(dim=1)
        boundary_pred_score = softmax(boundary_pred)
        test_siou = []
        test_piou = []

        for j in range(len(offset)):
            if j == 0:
                F = face[0:F_offset[j]]
                V = coord[0:offset[j]]
                b_gt = boundary[0:offset[j]]
                prediction = boundary_pred_score[0:offset[j]]
                type_pred = type_per_point[0:offset[j]]
                embedding = primitive_embedding[0:offset[j]]
            else:
                F = face[F_offset[j-1]:F_offset[j]]
                V = coord[offset[j-1]:offset[j]]
                b_gt = boundary[offset[j-1]:offset[j]]
                prediction = boundary_pred_score[offset[j-1]:offset[j]]
                type_pred = type_per_point[offset[j-1]:offset[j]]
                embedding = primitive_embedding[offset[j-1]:offset[j]]
            F = F.numpy().astype('int32')
            face_labels_with_embed = np.zeros((F.shape[0]), dtype='int32')
            masks = np.zeros((V.shape[0]), dtype='int32')
            gt_face_labels = np.zeros((F.shape[0]), dtype='int32')
            gt_masks = np.zeros((V.shape[0]), dtype='int32')
            output_label_with_embed = np.zeros((V.shape[0]), dtype='int32')
            gt_output_label = np.zeros((V.shape[0]), dtype='int32')
            b_gt = b_gt.data.cpu().numpy().astype('int32')
            b = (prediction[:,1]>prediction[:,0]).data.cpu().numpy().astype('int32')   # boundary predictions
            edg = trimesh.geometry.faces_to_edges(F)
            pb = np.zeros((edg.shape[0]), dtype='int32')
            for k in range(V.shape[0]):
                if b_gt[k] == 1:
                    pb[edg[:,0]==k] = 1
                    pb[edg[:,1]==k] = 1           
            
            # Ground Truth
            BGPC.primitive_clustering(c_void_p(pb.ctypes.data), c_void_p(F.ctypes.data),
            V.shape[0], F.shape[0], c_void_p(gt_face_labels.ctypes.data), c_void_p(gt_masks.ctypes.data),
            c_float(0.99), c_void_p(gt_output_label.ctypes.data))

            pb = np.zeros((edg.shape[0]), dtype='int32')
            for k in range(V.shape[0]):
                if b[k] == 1:
                    pb[edg[:,0]==k] = 1
                    pb[edg[:,1]==k] = 1

            spec_cluster_pred, point_embedding = mean_shift_gpu(embedding, offset, bandwidth=args.bandwidth, batch_size=args.cluster_bs)
            point_embedding = point_embedding.data.cpu().numpy().astype('float32')

            BGPC.Boundary_guided_primitive_clustering(c_void_p(point_embedding.ctypes.data), point_embedding.shape[1], c_void_p(pb.ctypes.data), c_void_p(F.ctypes.data),
            V.shape[0], F.shape[0], c_void_p(face_labels_with_embed.ctypes.data), c_void_p(masks.ctypes.data),
            c_float(0.99), c_void_p(output_label_with_embed.ctypes.data))

            pred_semantic = np.argmax(type_pred.data.cpu().numpy(), axis=1)

            s_iou_with_embed, p_iou_with_embed = compute_iou(gt_output_label, output_label_with_embed, pred_semantic, semantic.cpu().numpy())
            
            test_siou.append(s_iou_with_embed)
            test_piou.append(p_iou_with_embed)            
        
        s_iou = np.mean(test_siou)
        p_iou = np.mean(test_piou)
        s_iou = torch.tensor(s_iou).cuda()
        p_iou = torch.tensor(p_iou).cuda()

        if args.multiprocessing_distributed:
            dist.all_reduce(BA_emb_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(type_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(s_iou.div_(torch.cuda.device_count()))  
            dist.all_reduce(p_iou.div_(torch.cuda.device_count()))
        BA_emb_loss_, type_loss_ = BA_emb_loss.data.cpu().numpy(), type_loss.data.cpu().numpy()

        BA_emb_loss_meter.update(BA_emb_loss_.item())
        type_loss_meter.update(type_loss_.item())
        seg_iou_meter.update(s_iou)
        type_iou_meter.update(p_iou)
        
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'BA_emb_loss {BA_emb_loss_meter.val:.4f} ({BA_emb_loss_meter.avg:.4f}) '
                        'Type_Loss {type_loss_meter.val:.4f} ({type_loss_meter.avg:.4f}) '
                        'Acc {accuracy:.4f} '
                        'Seg_IoU {seg_iou_meter.val:.4f} ({seg_iou_meter.avg:.4f}) '
                        'Type_IoU {type_iou_meter.val:.4f} ({type_iou_meter.avg:.4f}) '.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          BA_emb_loss_meter=BA_emb_loss_meter,
                                                          type_loss_meter=type_loss_meter,
                                                          accuracy=accuracy,
                                                          seg_iou_meter=seg_iou_meter,
                                                          type_iou_meter=type_iou_meter))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes - 4):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('Val result: Seg_mIoU/Type_mIoU {:.4f}/{:.4f}.'.format(seg_iou_meter.avg, type_iou_meter.avg))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return seg_iou_meter.avg, type_iou_meter.avg, loss_meter.avg, BA_emb_loss_meter.avg, type_loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
