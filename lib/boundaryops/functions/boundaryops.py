from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import boundaryops_cuda

# Boundary-Guided Sampling
class BoundaryQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset, edges, boundary):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        b = edges.shape[1]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        boundaryops_cuda.boundaryquery_cuda(m, nsample, b, xyz, new_xyz, offset, new_offset, idx, dist2, edges, boundary)
        return idx, torch.sqrt(dist2)

boundaryquery = BoundaryQuery.apply