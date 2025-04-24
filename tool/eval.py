import os
import sys
import numpy as np
import torch
from lapsolver import solve_dense
from multiprocessing import Pool
import open3d as o3d
from torch.autograd.variable import Variable
import pickle


def to_one_hot(target_t, maxx=500):
	N = target_t.shape[0]
	maxx = np.max(target_t) + 1
	if maxx <= 0:
		maxx = 1
	target_one_hot = np.zeros((N, maxx))

	for i in range(target_t.shape[0]):
		if target_t[i] >= 0:
			target_one_hot[i][target_t[i]] = 1
	#target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)

	target_one_hot = torch.from_numpy(target_one_hot)
	return target_one_hot


def relaxed_iou_fast(pred, gt, max_clusters=500):
	batch_size, N, K = pred.shape
	normalize = torch.nn.functional.normalize
	one = torch.ones(1)

	norms_p = torch.unsqueeze(torch.sum(pred, 1), 2)
	norms_g = torch.unsqueeze(torch.sum(gt, 1), 1)
	cost = []

	for b in range(batch_size):
		p = pred[b]
		g = gt[b]
		c_batch = []
		dots = p.transpose(1, 0) @ g
		r_iou = dots
		r_iou = r_iou / (norms_p[b] + norms_g[b] - dots + 1e-7)
		cost.append(r_iou)
	cost = torch.stack(cost, 0)
	return cost

def primitive_type_segment_torch(pred, weights):
	"""
	Returns the primitive type for every segment in the predicted shape.
	:param pred: N x L
	:param weights: N x k
	"""
	d = torch.unsqueeze(pred, 2).float() * torch.unsqueeze(weights, 1).float()
	d = torch.sum(d, 0)
	return torch.max(d, 0)[1]


def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)

def chamfer_distance(pred, gt, sqrt=False):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3)
    if sqrt:
        diff = guard_sqrt(diff)

    cd = torch.mean(torch.min(diff, 1)[0], 1) + torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd) / 2.0
    return cd

def mean_IOU_primitive_segment_usecd(matching, predicted_labels, labels, pred_prim, gt_prim, points):
    """
    Primitive type IOU, this is calculated over the segment level.
    First the predicted segments are matched with ground truth segments,
    then IOU is calculated over these segments.
    :param matching
    :param pred_labels: N x 1, pred label id for segments
    :param gt_labels: N x 1, gt label id for segments
    :param pred_prim: K x 1, pred primitive type for each of the predicted segments
    :param gt_prim: N x 1, gt primitive type for each point
    """
    batch_size = labels.shape[0]
    IOU = []
    RECALL = []
    IOU_prim = []
    IOU_prim_weighted = []

    for b in range(batch_size):
        iou_b = []
        recall_b = []
        iou_b_prim = []
        iou_b_prim_weighted = []
        iou_b_prims = []
        len_labels = np.unique(labels[b]).shape[0]
        rows, cols = matching[b]
        # len_labels = len(cols)
        count = 0

        recall_pos = 0

        for r, c in zip(rows, cols):
            pred_indices = predicted_labels[b] == r
            gt_indices = labels[b] == c

            # use only matched segments for evaluation
            if (np.sum(gt_indices) == 0) or (np.sum(pred_indices) == 0):
                continue

            # also remove the gt labels that are very small in number
            # ====================================================================== 
            if np.sum(gt_indices) < 100:
                continue
            
            tp = np.sum(np.logical_and(pred_indices, gt_indices))

            iou = tp / (np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)

            w = np.sum(np.logical_and(pred_indices, gt_indices)) / (np.sum(gt_indices) + 1e-8)  
            
            # cal_recall = tp / (tp + fn)
            # https://blog.csdn.net/weixin_39347054/article/details/105408293
            # recall = tp / (tp + np.sum(np.logical_and(pred_indices == False, gt_indices == True)) + 1e-8)       
            # 

            chamfer_dis = chamfer_distance(points[pred_indices, :].unsqueeze(0), points[gt_indices, :].unsqueeze(0)) / 2
            # print(chamfer_dis)
            if chamfer_dis < 0.1:
                recall_pos += 1

            iou_b.append(iou)
            # recall_b.append(recall)
            # evaluation of primitive type prediction performance
            gt_prim_type_k = gt_prim[b][gt_indices][0]
			
			# Some point cloud segments have primitive types that are not unique, and take the mode.
            if gt_prim[b][gt_indices][0] != np.argmax(np.bincount(gt_prim[b][gt_indices])):     
                gt_prim_type_k = np.argmax(np.bincount(gt_prim[b][gt_indices]))
            try:
                predicted_prim_type_k = pred_prim[b][r]
            except:
                print("wrong")

            iou_b_prim_weighted.append(iou * (gt_prim_type_k == predicted_prim_type_k))
            iou_b_prim.append(gt_prim_type_k == predicted_prim_type_k)
            iou_b_prims.append([gt_prim_type_k, predicted_prim_type_k])

        # find the mean of IOU over this shape
        IOU.append(np.mean(iou_b))
        RECALL.append(float(recall_pos) / len_labels)
        IOU_prim.append(np.mean(iou_b_prim))
        IOU_prim_weighted.append(np.mean(iou_b_prim_weighted))
    return np.mean(IOU), np.mean(IOU_prim), iou_b_prims, np.mean(RECALL), np.mean(IOU_prim_weighted)

def SIOU_matched_segments_usecd(target, pred_labels, primitives_pred, primitives, weights, points):
    """
    Computes iou for segmentation performance and primitive type
    prediction performance.
    First it computes the matching using hungarian matching
    between predicted and ground truth labels.
    Then it computes the iou score, starting from matching pairs
    coming out from hungarian matching solver. Note that
    it is assumed that the iou is only computed over matched pairs.
    That is to say, if any column in the matched pair has zero
    number of points, that pair is not considered.
    
    It also computes the iou for primitive type prediction. In this case
    iou is computed only over the matched segments.
    """
    # 2 is open spline and 9 is close spline
    primitives[primitives == 0] = 9
    primitives[primitives == 6] = 9
    primitives[primitives == 7] = 9
    primitives[primitives == 8] = 2

    primitives_pred[primitives_pred == 0] = 9
    primitives_pred[primitives_pred == 6] = 9
    primitives_pred[primitives_pred == 7] = 9
    primitives_pred[primitives_pred == 8] = 2

    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(pred_labels)

    cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
    cost_ = 1.0 - cost.data.cpu().numpy()
    matching = []

    for b in range(1):
        rids, cids = solve_dense(cost_[b])
        matching.append([rids, cids])

    primitives_pred_hot = to_one_hot(primitives_pred, 10).float()

    # this gives you what primitive type the predicted segment has.
    prim_pred = primitive_type_segment_torch(primitives_pred_hot, weights).data.cpu().numpy()
    target = np.expand_dims(target, 0)
    pred_labels = np.expand_dims(pred_labels, 0)
    prim_pred = np.expand_dims(prim_pred, 0)
    primitives = np.expand_dims(primitives, 0)

    segment_iou, primitive_iou, iou_b_prims, segment_reacall, weight_p_iou = mean_IOU_primitive_segment_usecd(matching, pred_labels, target, prim_pred,
                                                                         primitives, points)
    return segment_iou, primitive_iou, matching, iou_b_prims, segment_reacall, weight_p_iou

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Residual_Number_metric(pred, gt):
    
    gt_num = np.unique(gt).shape[0]
    pred_num = np.unique(pred).shape[0]

    return np.abs(gt_num - pred_num)


data_path = sys.argv[1]
files = [data_path + '/predictions/' + f for f in os.listdir(os.path.join(data_path, 'predictions')) if f[-3:] == 'npz']
files.sort()
if not os.path.exists(data_path + '/relation-iou'):
	os.makedirs(data_path + '/relation-iou')
if not os.path.exists(data_path + '/relation'):
	os.makedirs(data_path + '/relation')
if not os.path.exists(data_path + '/statistics'):
	os.makedirs(data_path + '/statistics')

s_ious = []
p_ious = []

# def SaveRelation_iou(i):
# 	global files
for i in range(len(files)):
	f = files[i]
	fn = f.split('/')[-1][:8]
	# if os.path.exists(data_path + '/relation-iou/%d.npz'%(i)):
	# 	continue
	data = np.load(f)
	V, L_f, L_f_gt, S, S_gt, F = data['V'], data['L'], data['L_gt'], data['S'], data['S_gt'], data['F']
	L = data['L_p']
	L_gt = data['L_p_gt']

	weights = to_one_hot(L, np.unique(L).shape[0])
	s_iou, p_iou, _, _, s_recall, weight_p_iou = SIOU_matched_segments_usecd(
		L_gt,
		L,
		S,
		S_gt,
		weights,
		torch.from_numpy(V)
	)

	if np.isnan(s_iou) or np.isnan(p_iou):
		s_iou = 0.0
		p_iou = 0.0
		weight_p_iou = 0.0
		print('ignore nan!')
		# import ipdb;
		# ipdb.set_trace()
		# return


	S[S == 9] = 0
	S[S == 6] = 0
	S[S == 7] = 0
	S[S == 8] = 2

	S_gt[S_gt == 9] = 0
	S_gt[S_gt == 6] = 0
	S_gt[S_gt == 7] = 0
	S_gt[S_gt == 8] = 2
	intersection, union, target = intersectionAndUnion(S, S_gt, 6)
      
	num_dis = Residual_Number_metric(L, L_gt)

	# # Visualization
	# if not os.path.exists(data_path + '/visualize'):
	# 	os.makedirs(data_path + '/visualize')
	# V = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0
	# colors = np.random.rand(len(V), 3)
	# pcd = o3d.geometry.PointCloud()
	# pcd.points = o3d.utility.Vector3dVector(V)
	# pcd.colors = o3d.utility.Vector3dVector(colors[L_f])
	# o3d.io.write_point_cloud(data_path + '/visualize/%s-BGPSeg.ply'%(fn), pcd)

	result = np.array([s_iou, p_iou, s_recall, weight_p_iou, num_dis])
	semantic_result = np.array([intersection, union, target])

	np.savez_compressed(data_path + '/relation-iou/%s.npz'%(fn), result=result, semantic_result=semantic_result)
	print('Seg mIOU of {}: {:.4f}'.format(fn, s_iou))
# with Pool(2) as p:
# 	p.map(SaveRelation_iou, [i for i in range(len(files))])

iou_files = os.listdir(data_path + '/relation-iou')
s_ious = []
p_ious = []
s_recalls = []
weight_p_ious = []
num_dis_list = []
intersection_meter = AverageMeter()
union_meter = AverageMeter()
target_meter = AverageMeter()
for f in iou_files:
	data = np.load(data_path + '/relation-iou/%s'%(f))
	r = data['result']
	s_ious.append(r[0])
	p_ious.append(r[1])
	s_recalls.append(r[2])
	weight_p_ious.append(r[3])
	num_dis_list.append(r[4])
	s = data['semantic_result']
	intersection_meter.update(s[0])
	union_meter.update(s[1])
	target_meter.update(s[2])

fp = open(data_path + '/statistics/iou.txt','w')
fp.write("seg mIOU=%f label mIOU=%f w-label mIOU=%f Recall=%f ResN=%f\n"%(np.mean(s_ious), np.mean(p_ious), np.mean(weight_p_ious), np.mean(s_recalls), np.mean(num_dis_list)))
fp.close()
iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
mIoU = np.mean(iou_class)
mAcc = np.mean(accuracy_class)
allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

fp = open(data_path + '/statistics/semantic.txt','w')
fp.write("mIoU=%f mAcc=%f OA=%f\n"%(mIoU, mAcc, allAcc))
fp.close()


def SaveRelation_AP(i):
	global files
	f = files[i]
	data = np.load(f)
	L = data['L_p']
	L_gt = data['L_p_gt']

	# build match
	M = np.max(L) + 1
	N = np.max(L_gt) + 1
	if M < 0:
		M = 0
	if N < 0:
		N = 0

	relation_pred = [{} for i in range(M)]
	relation_gt = [{} for i in range(N)]

	label_count = np.zeros((M))
	label_count_gt = np.zeros((N))

	for i in range(L.shape[0]):
		if L[i] >= 0 and L_gt[i] >= 0:
			if not L_gt[i] in relation_pred[L[i]]:
				relation_pred[L[i]][L_gt[i]] = 1
			else:
				relation_pred[L[i]][L_gt[i]] += 1

			if not L[i] in relation_gt[L_gt[i]]:
				relation_gt[L_gt[i]][L[i]] = 1
			else:
				relation_gt[L_gt[i]][L[i]] += 1

		if L[i] >= 0:
			label_count[L[i]] += 1
		if L_gt[i] >= 0:
			label_count_gt[L_gt[i]] += 1

	with open(data_path + '/relation/%s.pkl'%(f.split('/')[-1][:8]), 'wb') as handle:
		pickle.dump((relation_pred, relation_gt, label_count, label_count_gt, L, L_gt), handle, protocol = pickle.HIGHEST_PROTOCOL)


# -----------compute APs metric--------------

with Pool(32) as p:
	p.map(SaveRelation_AP, [i for i in range(len(files))])

overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
min_region_size = 20

files = [data_path + '/relation/' + f for f in os.listdir(data_path + '/relation/')][:]
aps = []
for oi, overlap_th in enumerate(overlaps):
	hard_false_negatives = 0
	y_true = np.empty(0)
	y_score = np.empty(0)
	c = 0
	for fi, f in enumerate(files):
		c += 1
		with open(f, 'rb') as handle:
			relation_pred, relation_gt, label_count, label_count_gt, L, L_gt = pickle.load(handle)
		M = len(relation_pred)
		N = len(relation_gt)

		cur_true = np.ones((N))
		cur_score = np.ones((N)) * -1
		cur_match = np.zeros((N), dtype=np.bool_)

		for gt_idx in range(N):
			found_match = False
			if label_count_gt[gt_idx] < min_region_size:
				continue
			for pred_idx, intersection in relation_gt[gt_idx].items():
				overlap = intersection / (label_count[pred_idx] + label_count_gt[gt_idx] - intersection + 0.0)
				if overlap > overlap_th:
					confidence = overlap
					if cur_match[gt_idx]:
						max_score = np.max([cur_score[gt_idx], confidence])
						min_score = np.min([cur_score[gt_idx], confidence])
						cur_score[gt_idx] = max_score

						cur_true = np.append(cur_true, 0)
						cur_score = np.append(cur_score, min_score)
						cur_match = np.append(cur_match, True)
					else:
						found_match = True
						cur_match[gt_idx] = True
						cur_score[gt_idx] = confidence

			if not found_match:
				hard_false_negatives += 1

		cur_true = cur_true[cur_match == True]
		cur_score = cur_score[cur_match == True]

		for pred_idx in range(M):
			if label_count[pred_idx] < min_region_size or label_count[pred_idx] < np.sum(label_count) * 0.1:
				continue
			found_gt = False
			for gt_idx, intersection in relation_pred[pred_idx].items():
				overlap = intersection / (label_count[pred_idx] + label_count_gt[gt_idx] - intersection + 0.0)
				if overlap > overlap_th:
					found_gt = True
					break
			if not found_gt:
				num_ignore = np.sum((L_gt[L == pred_idx] < 0).astype('int32'))
				for gt_idx, intersection in relation_pred[pred_idx].items():
					if label_count_gt[gt_idx] < min_region_size:
						num_ignore += intersection

				if num_ignore / (label_count[pred_idx] + 0.0) < overlap_th:
					cur_true = np.append(cur_true, 0)
					confidence = 0
					cur_score = np.append(cur_score, confidence)


		y_true = np.append(y_true, cur_true)
		y_score = np.append(y_score, cur_score)

	score_arg_sort      = np.argsort(y_score)
	y_score_sorted      = y_score[score_arg_sort]
	y_true_sorted       = y_true[score_arg_sort]
	y_true_sorted_cumsum = np.cumsum(y_true_sorted)

	# unique thresholds
	(thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
	num_prec_recall = len(unique_indices) + 1

	# prepare precision recall
	num_examples      = len(y_score_sorted)
	num_true_examples = y_true_sorted_cumsum[-1]
	precision         = np.zeros(num_prec_recall)
	recall            = np.zeros(num_prec_recall)

	# deal with the first point
	y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
	# deal with remaining
	for idx_res,idx_scores in enumerate(unique_indices):
		cumsum = y_true_sorted_cumsum[idx_scores-1]
		tp = num_true_examples - cumsum
		fp = num_examples      - idx_scores - tp
		fn = cumsum + hard_false_negatives
		p  = float(tp)/(tp+fp)
		r  = float(tp)/(tp+fn)
		precision[idx_res] = p
		recall   [idx_res] = r

	# first point in curve is artificial
	precision[-1] = 1.
	recall   [-1] = 0.

	# compute average of precision-recall curve
	recall_for_conv = np.copy(recall)
	recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
	recall_for_conv = np.append(recall_for_conv, 0.)

	stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
	# integrate is now simply a dot product
	ap_current = np.dot(precision, stepWidths)

	aps.append(ap_current)
	print(overlap_th, ap_current)

aps = np.array(aps)
fp = open(data_path + '/statistics/AP.txt', 'w')
fp.write('ap50 = %f\n'%(aps[0]))
fp.write('ap75 = %f\n'%(aps[5]))
fp.write('ap25 = %f\n'%(aps[9]))
fp.write('map = %f\n'%(np.average(aps[:8])))
fp.close()
