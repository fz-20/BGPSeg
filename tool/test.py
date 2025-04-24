import os, sys
import requests
import trimesh
# from util.visualize_util import open3d_vis_prim_face
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import random
import numpy as np
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from util import config
from util.ABCPrimitive import ABCPrimitive_Dataset
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn_region
from util.logger import get_logger
from util.loss_util import compute_type_loss, compute_embedding_loss_boundary, mean_shift_gpu, compute_iou

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
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
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
    global args, best_iou, best_iou_cluster
    args, best_iou, best_iou_cluster = argss, 0, 0
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

    # Load boundary predictor model
    if args.boundary_model_path:
        if os.path.isfile(args.boundary_model_path):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.boundary_model_path))
            checkpoint = torch.load(args.boundary_model_path)
            boundarymodel.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.boundary_model_path))
        else:
            raise ValueError("=> no weight found at '{}'".format(args.boundary_model_path))
    else:
        raise ValueError("Please enter the boundary model path")

    # Load model
    if args.model_path:
        if os.path.isfile(args.model_path):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.model_path))
        else:
            raise ValueError("=> no weight found at '{}'".format(args.model_path))
    else:
        raise ValueError("Please enter the model path")

    if args.evaluate:
        if args.data_name == 'ABCPrimitive':
            val_data = ABCPrimitive_Dataset(split='val', data_root=args.data_root, loop=args.test_loop)
        else:
            raise ValueError("The dataset {} is not supported.".format(args.data_name))
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_region)

    ###################
    # start testing #
    ###################

    seg_miou, label_miou, loss_val, BA_emb_loss_val, type_loss_val, mIoU_val, mAcc_val, allAcc_val = test(val_loader, model, boundarymodel, criterion)
    # print("seg_miou: {}, label_miou: {}".format(seg_miou, label_miou))
    return 0
        

def test(val_loader, model, boundarymodel, criterion):
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

            # Boundary-Guided Primitive Clustering
            BGPC.Boundary_guided_primitive_clustering(c_void_p(point_embedding.ctypes.data), point_embedding.shape[1], c_void_p(pb.ctypes.data), c_void_p(F.ctypes.data),
            V.shape[0], F.shape[0], c_void_p(face_labels_with_embed.ctypes.data), c_void_p(masks.ctypes.data),
            c_float(0.99), c_void_p(output_label_with_embed.ctypes.data))

            pred_semantic = np.argmax(type_pred.data.cpu().numpy(), axis=1)

            s_iou_with_embed, p_iou_with_embed = compute_iou(gt_output_label, output_label_with_embed, pred_semantic, semantic.cpu().numpy())

            # save results
            np.savez_compressed(args.save_path + '/predictions/%s'%((fn[0] + '.npz')), V=V.cpu().numpy(),F=F,L=face_labels_with_embed,L_gt=gt_face_labels, S=pred_semantic, S_gt=semantic.cpu().numpy(), L_p=output_label_with_embed, L_p_gt=gt_output_label)
            
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
