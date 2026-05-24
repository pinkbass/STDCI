import numpy as np 
import scipy.io as scio 
from torch.utils.data import Dataset 
import os 
import os.path as osp 
from .builder import DATASETS
from stci.utils.utils import tansT_np
import math


@DATASETS.register_module
class GraySim2stepSTData(Dataset):
    def __init__(self,data_root,*args,**kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        self.mask = kwargs["mask"]
        self.frames,self.height_mask,self.width_mask = self.mask.shape           
        self.s_cr = 4


    def __getitem__(self,index):
        pic = scio.loadmat(osp.join(self.data_root,self.data_name_list[index]))
        
        if "orig" in pic:
            pic = pic['orig']
        elif "patch_save" in pic:
            pic = pic['patch_save']
        elif "p1" in pic:
            pic = pic['p1']
        elif "p2" in pic:
            pic = pic['p2']
        elif "p3" in pic:
            pic = pic['p3']

        pic = pic / 255
        H,W,N = pic.shape
        # pic_H = self.pic_H
        pic_H = math.floor(H/self.height_mask)*self.height_mask
        pic_W = math.ceil(W/self.width_mask)*self.width_mask
        d=math.floor((H-pic_H)/2)

        nh = pic_H//self.height_mask
        nw = pic_W//self.width_mask

        mask = np.tile(self.mask,(1, nh, nw))

        pic = pic[d:d+pic_H, :, :]             

        pic_gt = np.zeros([pic.shape[2] // self.frames, self.frames, pic_H, pic_W])        
        meas = np.zeros([pic_H//self.s_cr, pic_W//self.s_cr])                               
        
        for jj in range(pic.shape[2]):
            if jj % self.frames == 0:   
                meas_st_cs = np.zeros([pic_H//self.s_cr, pic_W//self.s_cr])  
                n = 0
            pic_t = np.zeros([pic_H, pic_W])
            pic_t[:,:W]= pic[:, :, jj]
            mask_t = mask[n, :, :]

            pic_gt[jj // self.frames, n, :, :] = pic_t
            n += 1   
            meas_t = np.multiply(mask_t, pic_t)                          
            Th = tansT_np(pic_H, self.s_cr)
            Tw = tansT_np(pic_W, self.s_cr)            
            meas_st = Th@meas_t@Tw.T                                    
            
            meas_st_cs = meas_st_cs + meas_st

            if jj == (self.frames-1):
                meas_st_cs = np.expand_dims(meas_st_cs, 0)
                meas = meas_st_cs
            elif (jj + 1) % self.frames == 0 and jj != (self.frames-1):
                meas_st_cs = np.expand_dims(meas_st_cs, 0)
                meas = np.concatenate((meas, meas_st_cs), axis=0)
        return meas,pic_gt                                             
    def __len__(self,):
        return len(self.data_name_list)
    
