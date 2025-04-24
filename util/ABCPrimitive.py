import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset

from util.data_util import data_prepare_abcprimitive, data_prepare_abcprimitive_val


class ABCPrimitive_Dataset(Dataset):
    def __init__(self, split='train', data_root='trainval', loop=1):
        super().__init__()
        self.split, self.loop = split, loop
        if split == 'train':
            data_root += '/train/'
        elif split == 'val' or split == 'test':
            data_root += '/val/'
        data_list = sorted(os.listdir(data_root))
        self.data_list = [item[:-4] for item in data_list]
        self.data_root = data_root
        
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + '.npz')
        data = np.load(data_path)

        coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F'],data['edges'],data['dse_edges']

        # noise = normals * np.clip(
        #     np.random.randn(coord.shape[0], 1) * 0.01,
        #     a_min=-0.01,
        #     a_max=0.01)
        # coord_noise = coord + noise.astype(np.float32)
        # coord_min = np.min(coord_noise, 0)
        # coord_noise -= coord_min
        # coord_noise = torch.FloatTensor(coord_noise)

        if self.split == 'train':
            coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data_prepare_abcprimitive(coord, normals, boundary, label, semantic, param, F, edges, dse_edges)
        else:
            coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data_prepare_abcprimitive_val(coord, normals, boundary, label, semantic, param, F, edges, dse_edges)

        return item, coord, normals, boundary, label, semantic, param, F, edges, dse_edges

    def __len__(self):
        return round(len(self.data_idx) * self.loop)