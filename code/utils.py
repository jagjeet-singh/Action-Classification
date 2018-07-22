from torch.utils.data import Dataset
import numpy as np
import pdb
import torch

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


# class MLPDataset(Dataset):

#     def __init__(self, data_list, type='train'):

#         self.type = type
#         self.num_frames = data_list[0]['features'].shape[0]
#         self.num_classes = 51
#         self.feature_size = data_list[0]['features'].shape[1]
#         self.data_size = len(data_list)
#         self.data = torch.zeros((self.data_size*self.num_frames, self.feature_size))
#         self.classes = torch.zeros((self.data_size*self.num_frames,)).type(LongTensor)
#         for i in range(self.data_size):
#             for j in range(self.num_frames):
#                 self.data[i*self.num_frames+j] = torch.from_numpy(data_list[i]['features'][j])
#                 if type != 'test':
#                     self.classes[i*self.num_frames+j] = data_list[i]['class_num'].item()
#         # self.data = self.data.view(self.data_size*self.num_frames, self.feature_size)
#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         if self.type != 'test':
#             return self.data[idx,:], self.classes[idx]
#         else:
#             return self.data[idx,:]

class ActionClsDataset(Dataset):

    def __init__(self, data_list, type='train'):

        self.type = type
        self.num_frames = data_list[0]['features'].shape[0]
        self.num_classes = 51
        self.feature_size = data_list[0]['features'].shape[1]
        self.data_size = len(data_list)
        self.data = torch.zeros((self.data_size, self.num_frames, self.feature_size))
        self.classes = torch.zeros((self.data_size,)).type(LongTensor)
        for i in range(self.data_size):
            self.data[i,:,:] = torch.from_numpy(data_list[i]['features'])
            if type != 'test':
                self.classes[i] = data_list[i]['class_num'].item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.type != 'test':
            return self.data[idx,:,:], self.classes[idx]
        else:
            return self.data[idx,:,:]