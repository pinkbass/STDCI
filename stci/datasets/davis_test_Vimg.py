import os
import os.path as osp
import cv2
from torch.utils.data import Dataset 
from .builder import DATASETS
from .pipelines.builder import build_pipeline

@DATASETS.register_module 
class davis_test_Vimg(Dataset):                                               
    def __init__(self,data_root,*args,**kwargs):
        self.img_files = []
        self.mask = kwargs["mask"]
        self.gene_meas = build_pipeline(kwargs["gens_meas_test"])
        self.ratio,self.resize_w,self.resize_h = self.mask.shape
        
        for image_dir in os.listdir(data_root):                                  
            train_data_path = osp.join(data_root,image_dir)                    
            data_path = os.listdir(train_data_path)                             
            data_path.sort()                                                    
            for sub_index in range(self.ratio):                                
                sub_data_path = data_path[sub_index:]
                image_name_list = []
                count = 0                                                                                
                for image_name in sub_data_path:                                
                    image_name_list.append(osp.join(train_data_path,image_name))
                    if (count+1)%self.ratio==0:
                        self.img_files.append(image_name_list)
                        image_name_list = []
                    count += 1
    def __getitem__(self, index):
        imgs = []
        for i,image_path in enumerate(self.img_files[index]):                  
            img = cv2.imread(image_path)
            imgs.append(img)
        gt,meas = self.gene_meas(imgs,self.mask)
        return gt,meas

    def __len__(self):
        return len(self.img_files)
    

@DATASETS.register_module 
class davis_test_gt(Dataset):                                                  
    def __init__(self,data_root,*args,**kwargs):
        self.img_files = []
        self.mask = kwargs["mask"]
        self.gene_data = build_pipeline(kwargs["gens_meas_test"])
        self.ratio,self.resize_w,self.resize_h = self.mask.shape
        
        for image_dir in os.listdir(data_root):                                  
            train_data_path = osp.join(data_root,image_dir)                    
            data_path = os.listdir(train_data_path)                             
            data_path.sort()                                                   

            for sub_index in range(self.ratio):                                 
                sub_data_path = data_path[sub_index:]
                image_name_list = []
                count = 0                                                                                
                for image_name in sub_data_path:                                
                    image_name_list.append(osp.join(train_data_path,image_name))
                    if (count+1)%self.ratio==0:
                        self.img_files.append(image_name_list)
                        image_name_list = []
                    count += 1
    def __getitem__(self, index):
        imgs = []
        for i,image_path in enumerate(self.img_files[index]):                   
            img = cv2.imread(image_path)
            imgs.append(img)
        gt = self.gene_data(imgs,self.mask)
        return gt

    def __len__(self):
        return len(self.img_files)