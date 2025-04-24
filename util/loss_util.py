import torch
import torch.nn.functional as F
import numpy as np
from torch import sqrt, exp
import time
import math
from lapsolver import solve_dense
from lib.pointops.functions import pointops


def compute_type_loss(pred, gt, criterion):

    valid_class = (gt != -1)
    pred = pred[valid_class]
    gt = gt[valid_class]
    loss = criterion(pred, gt)
    return loss


def compute_boundary_loss(pred, gt):
    '''
    Boundary Loss, weighted binary cross-entropy loss.
    '''
    
    pos_weight = torch.sum(gt == 0).float() / torch.sum(gt == 1).float()
    gt = F.one_hot(gt.long(), 2).float()
    boundary_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = boundary_loss(pred, gt)

    return loss


def compute_embedding_loss_boundary(point, boundary, pred_feat, gt_label, offset, t_pull=0.5, t_pull_boundary=0.8, t_push=1.5):
    '''
    Boundary-Adaptive Embedding Loss
    pred_feat: (N, K)
    gt_label: (N)
    '''
    num_pts, feat_dim = pred_feat.shape
    device = pred_feat.device
    pull_loss = torch.Tensor([0.0]).to(device)
    push_loss = torch.Tensor([0.0]).to(device)
    for i in range(len(offset)):
        if i == 0:
            pred = pred_feat[0:offset[i]]
            gt = gt_label[0:offset[i]]
            b = boundary[0:offset[i]]
            p = point[0:offset[i]]
            offset_i = offset[i]
        else:
            pred = pred_feat[offset[i-1]:offset[i]]
            gt = gt_label[offset[i-1]:offset[i]]
            b = boundary[offset[i-1]:offset[i]]
            p = point[offset[i-1]:offset[i]]
            offset_i = offset[i] - offset[i-1]

        # for i in range(batch_size):
        # gt = gt - 1
        num_class = gt.max() + 2

        embeddings = []
        b_embeddings = []   # points near the boundary
        nb_embeddings = []  # points not near the boundary

        # 若点knn中有边界点，则该点作为边界附近的点，作mask
        nsample = 4
        neighbor_idx, _ = pointops.knnquery(nsample, p, p, offset_i, offset_i)
        neighbor_b = b[neighbor_idx.long()]
        neighbor_b = torch.sum(neighbor_b, dim=1)
        b_mask = (neighbor_b > 0)

        for j in range(num_class):
            mask = (gt == j - 1)
            feature = pred[mask]           
            if len(feature) == 0:
                continue
            embeddings.append(feature)
            nb_feature = pred[mask & ~b_mask]
            nb_embeddings.append(nb_feature)
            b_feature = pred[mask & b_mask]
            b_embeddings.append(b_feature)

        centers = []

        for feature in embeddings:
            center = torch.mean(feature, dim=0).view(1, -1)
            centers.append(center)

        # intra-embedding loss
        pull_loss_tp = torch.Tensor([0.0]).to(device)
        for b_feature, nb_feature, center in zip(b_embeddings, nb_embeddings, centers):
            if len(nb_feature) == 0:
                pull_loss_tp += .0
            else:
                dis = torch.norm(nb_feature - center, 2, dim=1) - t_pull
                dis = F.relu(dis)
                pull_loss_tp += torch.mean(dis)
            if len(b_feature) == 0:
                pull_loss_tp += .0
            else:
                dis = torch.norm(b_feature - center, 2, dim=1) - t_pull_boundary
                dis = F.relu(dis) * 1.2
                pull_loss_tp += torch.mean(dis)

        pull_loss = pull_loss + pull_loss_tp / len(embeddings)

        # inter-embedding loss
        try:
            centers = torch.cat(centers, dim=0)  # (num_class, K)
        except:
            import ipdb
            ipdb.set_trace()

        if centers.shape[0] == 1:
            continue

        dst = torch.norm(centers[:, None, :] - centers[None, :, :], 2, dim=2)

        eye = torch.eye(centers.shape[0]).to(device)
        pair_distance = torch.masked_select(dst, eye == 0)

        pair_distance = t_push - pair_distance
        pair_distance = F.relu(pair_distance)
        push_loss += torch.mean(pair_distance)

    pull_loss = pull_loss / len(offset)
    push_loss = push_loss / len(offset)
    loss = pull_loss + push_loss
    return loss, pull_loss, push_loss


class MeanShift_GPU():
    ''' Do meanshift clustering with GPU support'''
    def __init__(self,bandwidth = 2.5, batch_size = 1000, max_iter = 10, eps = 1e-5, check_converge = False):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.bandwidth = bandwidth
        self.eps = eps # use for check converge
        self.cluster_eps = 1e-1 # use for check cluster
        self.check_converge = check_converge # Check converge will take 1.5 time longer
          
    def distance_batch(self,a,B):
        ''' Return distance between each element in a to each element in B'''
        return sqrt(((a[None,:] - B[:,None])**2)).sum(2)
    
    def distance(self,a,b):
        return np.sqrt(((a-b)**2).sum())
    
    def fit(self,data):
        with torch.no_grad():
            n = len(data)
            if not data.is_cuda:
                data_gpu = data.cuda()
                X = data_gpu.clone()
            else:
                X = data.clone()
            #X = torch.from_numpy(np.copy(data)).cuda()
            
            for _ in range(self.max_iter):
                max_dis = 0;
                for i in range(0,n,self.batch_size):
                    s = slice(i,min(n,i+ self.batch_size))
                    if self.check_converge:
                        dis = self.distance_batch(X,X[s])
                        max_batch = torch.max(dis)
                        if max_dis < max_batch:
                            max_dis = max_batch;
                        weight = dis
                        weight = self.gaussian(dis, self.bandwidth)
                    else:
                        weight = self.gaussian(self.distance_batch(X,X[s]), self.bandwidth)
                    num = (weight[:,:,None]*X).sum(dim=1)
                    X[s] = num / weight.sum(1)[:,None]                    
                    
                #import pdb; pdb.set_trace()
                #Check converge
                if self.check_converge:
                    if max_dis < self.eps:
                        print("Converged")
                        break
            
            # end_time = time.time()
            # print("algorithm time (s)", end_time- begin_time)
            # Get center and labels
            if True:
                # Convert to numpy cpu show better performance
                points = X.cpu().data.numpy()
                labels, centers = self.cluster_points(points)
            else:
                # use GPU
                labels, centers = self.cluster_points(points)
                
            labels = np.array(labels)
            centers = np.array(centers)
            return labels,centers,X
        
    def gaussian(self,dist,bandwidth):
        return exp(-0.5*((dist/bandwidth))**2)/(bandwidth*math.sqrt(2*math.pi))
        
    def cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for j,center in enumerate(cluster_centers):
                    dist = self.distance(point, center)
                    if(dist < self.cluster_eps):
                        cluster_ids.append(j)
                        break
                if(len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids, cluster_centers


def mean_shift_gpu(x, offset, bandwidth, batch_size=700):
    # x: [N, f]
    N, c = x.shape
    IDX = np.zeros(N, dtype=int)
    ms = MeanShift_GPU(bandwidth = bandwidth, batch_size = batch_size, max_iter = 10, eps = 1e-5, check_converge = False)
    for i in range(len(offset)):
        if i == 0:
            pred = x[0:offset[i]]
        else:
            pred = x[offset[i-1]:offset[i]]

        tic = time.time()
        labels, centers, X_fea = ms.fit(pred)
        # print ('[{}/{}] time for Mean shift clustering'.format(i+1, len(offset)), time.time() - tic)
        if i == 0:
            IDX[0:offset[i]] = labels
        else:
            IDX[offset[i-1]:offset[i]] = labels

    return IDX, X_fea


def mean_IOU_primitive_segment(matching, predicted_labels, labels, pred_prim, gt_prim):
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
	IOU_prim = []

	for b in range(batch_size):
		iou_b = []
		iou_b_prim = []
		iou_b_prims = []
		len_labels = np.unique(predicted_labels[b]).shape[0]
		rows, cols = matching[b]
		count = 0
		for r, c in zip(rows, cols):
			pred_indices = predicted_labels[b] == r
			gt_indices = labels[b] == c

			# use only matched segments for evaluation
			if (np.sum(gt_indices) == 0) or (np.sum(pred_indices) == 0):
				continue

			# also remove the gt labels that are very small in number
			if np.sum(gt_indices) < 100:
				continue

			iou = np.sum(np.logical_and(pred_indices, gt_indices)) / (
						np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
			iou_b.append(iou)

			# evaluation of primitive type prediction performancegt
			gt_prim_type_k = gt_prim[b][gt_indices][0]
			# if gt_prim[b][gt_indices][0] != stats.mode(gt_prim[b][gt_indices]).mode[0]:     # When background points exist
			# 	gt_prim_type_k = stats.mode(gt_prim[b][gt_indices]).mode[0]
               
            # Some point cloud segments have primitive types that are not unique, and take the mode.
			if gt_prim[b][gt_indices][0] != np.argmax(np.bincount(gt_prim[b][gt_indices])):     
				gt_prim_type_k = np.argmax(np.bincount(gt_prim[b][gt_indices]))
			try:
				predicted_prim_type_k = pred_prim[b][r]
			except:
				import ipdb;
				ipdb.set_trace()

			iou_b_prim.append(gt_prim_type_k == predicted_prim_type_k)
			iou_b_prims.append([gt_prim_type_k, predicted_prim_type_k])

		# find the mean of IOU over this shape
		IOU.append(np.mean(iou_b))
		IOU_prim.append(np.mean(iou_b_prim))
	return np.mean(IOU), np.mean(IOU_prim), iou_b_prims

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

def SIOU_matched_segments(target, pred_labels, primitives_pred, primitives, weights):
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
	primitives[primitives == 9] = 0
	primitives[primitives == 6] = 0
	primitives[primitives == 7] = 0
	primitives[primitives == 8] = 2

	primitives_pred[primitives_pred == 9] = 0
	primitives_pred[primitives_pred == 6] = 0
	primitives_pred[primitives_pred == 7] = 0
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

	segment_iou, primitive_iou, iou_b_prims = mean_IOU_primitive_segment(matching, pred_labels, target, prim_pred,
																		 primitives)
	return segment_iou, primitive_iou, matching, iou_b_prims

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

def compute_iou(gt_face_labels, face_labels, semantic_faces, semantic_faces_gt):
    weights = to_one_hot(face_labels, np.unique(face_labels).shape[0])
    s_iou, p_iou, _, _ = SIOU_matched_segments(
		gt_face_labels,
		face_labels,
		semantic_faces,
		semantic_faces_gt,
		weights,
	)
    return s_iou, p_iou