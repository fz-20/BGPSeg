import math
import numpy as np
from collections import Counter
import torch

def collate_fn(batch):
    fn, coord, normals, boundary, label, semantic, param, F, edges, dse_edges = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)

    return fn, torch.cat(coord), torch.cat(normals), torch.cat(boundary), torch.cat(label), torch.cat(semantic), torch.cat(param), torch.IntTensor(offset), torch.cat(edges), torch.cat(dse_edges)

def collate_fn_region(batch):
    fn, coord, normals, boundary, label, semantic, param, F, edges, dse_edges = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)

    F_offset, count = [], 0
    for item in F:
        # print("item shape:",item.shape)
        count += item.shape[0]
        F_offset.append(count)
    return fn, torch.cat(coord), torch.cat(normals), torch.cat(boundary), torch.cat(label), torch.cat(semantic), torch.cat(param), torch.IntTensor(offset), torch.cat(edges), torch.cat(dse_edges), torch.cat(F), torch.IntTensor(F_offset)


def data_prepare_abcprimitive(coord, normals, boundary, label, semantic, param, F, edges, dse_edges):

    coord_min = np.min(coord, 0)
    coord -= coord_min
    label -= 1  
    # set small number primitive as background
    counter = Counter(label)
    mapper = np.ones([label.max() + 1]) * -1
    keys = [k for k, v in counter.items() if v > 100]
    if len(keys):
        mapper[keys] = np.arange(len(keys))
    label = mapper[label]
    clean_primitives = np.ones_like(semantic) * -1
    valid_mask = label != -1
    clean_primitives[valid_mask] = semantic[valid_mask]
    semantic = clean_primitives.astype(int)
    label = label.astype(int)
    coord = torch.FloatTensor(coord)
    normals = torch.FloatTensor(normals)
    boundary = torch.LongTensor(boundary)
    semantic = torch.LongTensor(semantic)
    param = torch.FloatTensor(param)
    label = torch.LongTensor(label)
    F = torch.LongTensor(F)
    edges = torch.IntTensor(edges)
    dse_edges = torch.IntTensor(dse_edges)
    return coord, normals, boundary, label, semantic, param, F, edges, dse_edges

def data_prepare_abcprimitive_val(coord, normals, boundary, label, semantic, param, F, edges, dse_edges):

    coord_min = np.min(coord, 0)
    coord -= coord_min
    
    coord = torch.FloatTensor(coord)
    normals = torch.FloatTensor(normals)
    boundary = torch.LongTensor(boundary)
    semantic = torch.LongTensor(semantic)
    param = torch.FloatTensor(param)
    label = torch.LongTensor(label)
    F = torch.LongTensor(F)
    edges = torch.IntTensor(edges)
    dse_edges = torch.IntTensor(dse_edges)
    return coord, normals, boundary, label, semantic, param, F, edges, dse_edges