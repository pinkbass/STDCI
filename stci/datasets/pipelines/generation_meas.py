import cv2 
import numpy as np 
from .builder import PIPELINES
from stci.utils.utils import tansT_np, gradient_sobel
import math

# ---------------------------------------------------------------
#                measurement with block
# ---------------------------------------------------------------  
@PIPELINES.register_module
class GeneSTGrayMeas_dataAug:
    def __init__(self,norm=255,s_cr=4): 
        self.norm=norm
        self.s_cr=s_cr
    def __call__(self, imgs, mask):
        assert isinstance(imgs,list),                      "imgs must be list"
        gt = []
        m_cr,m_h,m_w = mask.shape
        i_cr = len(imgs)                                   # return n_frames        
        i_h = imgs[0].shape[0]
        i_w = imgs[0].shape[1]
        assert m_cr==i_cr and m_h==i_h and m_w==i_w,       "Image size does not match mask size! "    # 单组测量值处理，cr帧图像作为一组图像进行处理
        meas = np.zeros_like(mask[0])
        for i,img in enumerate(imgs):       
            if len(img.shape) == 3:     
                Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            else:
                Y=img
            Y = Y.astype(np.float32)/self.norm            
            gt.append(Y)
            meas += np.multiply(mask[i, :, :], Y)          # 采用时空域模板进行调制并产生时域压缩结果
        T0 = tansT_np(mask[0].shape[-1],self.s_cr)
        meas = T0@meas@T0.T                                # 进行空域降采样，时域加和与空域加和的前后顺序并没有影响
        return np.array(gt),meas
    
# ---------------------------------------------------------------
#               measurement without block 
# ---------------------------------------------------------------
@PIPELINES.register_module
class GenSTGrayMeasNonBlock:
    def __init__(self,norm=255,s_cr=4): 
        self.norm=norm
        self.s_cr=s_cr

    def __call__(self, imgs, mask):
        assert isinstance(imgs,list),                          "imgs must be list"
        gt = []
        # --------block_size=128--------
        m_cr,m_h,m_w = mask.shape
        i_cr = len(imgs)                                       # return n_frames
        i_h,i_w,ch = imgs[0].shape
        n_h = math.ceil(i_h/m_h)
        n_w = math.ceil(i_w/m_w)
        mask_exp=np.tile(mask,(1,n_h,n_w))
        img_exp = np.zeros_like(mask_exp[0])
        # assert m_cr==i_cr and m_h==i_h and m_w==i_w,        "Image size does not match mask size! "
        meas = np.zeros_like(mask_exp[0])
        for i,img in enumerate(imgs):
            Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            Y_h,Y_w = Y.shape
            img_exp[:Y_h,:Y_w]=Y
            Y = img_exp.astype(np.float32)/self.norm            
            gt.append(Y)
            meas += np.multiply(mask_exp[i, :, :], Y)         # 采用时空域模板进行调制并产生时域压缩结果
        T_row = tansT_np(mask_exp[0].shape[-2],self.s_cr)
        T_col = tansT_np(mask_exp[0].shape[-1],self.s_cr)
        meas = T_row@meas@T_col.T                             # 进行空域降采样，时域加和与空域加和的前后顺序并没有影响
        return np.array(gt),meas    

# ---------------------------------------------------------------
#                      add noise
# ---------------------------------------------------------------
@PIPELINES.register_module
class GeneSTGrayMeas_dataAug_noisy:
    def __init__(self,norm=255,s_cr=4, snr_range=[10, 40]): 
        self.norm=norm
        self.s_cr=s_cr
        self.snr_range = snr_range
    def __call__(self, imgs, mask):
        assert isinstance(imgs,list),                      "imgs must be list"
        gt = []
        m_cr,m_h,m_w = mask.shape
        i_cr = len(imgs)                                   # return n_frames        
        i_h = imgs[0].shape[0]
        i_w = imgs[0].shape[1]
        assert m_cr==i_cr and m_h==i_h and m_w==i_w,       "Image size does not match mask size! "    # 单组测量值处理，cr帧图像作为一组图像进行处理
        meas = np.zeros_like(mask[0])
        for i,img in enumerate(imgs):       
            if len(img.shape) == 3:     
                Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            else:
                Y=img
            Y = Y.astype(np.float32)/self.norm            
            gt.append(Y)
            meas += np.multiply(mask[i, :, :], Y)          # 采用时空域模板进行调制并产生时域压缩结果
        T0 = tansT_np(mask[0].shape[-1],self.s_cr)
        meas = T0@meas@T0.T                                # 进行空域降采样，时域加和与空域加和的前后顺序并没有影响
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        signal_power = np.mean(meas ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.randn(*meas.shape) * np.sqrt(noise_power)
        meas = meas + noise
        return np.array(gt),meas
    

# ---------------------------------------------------------------
#           得到时空高分辨率图像序列并且不进行压缩
# ---------------------------------------------------------------    
@PIPELINES.register_module
class GeneSTgt_dataAug:
    def __init__(self,norm=255,s_cr=2): 
        self.norm=norm
        self.s_cr=s_cr
    def __call__(self, imgs):
        assert isinstance(imgs,list),                      "imgs must be list"
        gt = []
        for i, img in enumerate(imgs):       
            if len(img.shape) == 3:     
                Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            else:
                Y=img
            Y = Y.astype(np.float32)/self.norm            
            gt.append(Y)
        return np.array(gt)
    
# ---------------------------------------------------------------
#               用于'不分块'的整幅'gt'--1024*1024
# ---------------------------------------------------------------
@PIPELINES.register_module
class GenSTgtNonBlock:
    def __init__(self,norm=255,s_cr=4): 
        self.norm=norm
        self.s_cr=s_cr

    def __call__(self, imgs, mask):
        assert isinstance(imgs,list),                      "imgs must be list"
        gt = []
        # --------block_size=128--------
        m_cr,m_h,m_w=mask.shape
        i_cr = len(imgs)                                   # return n_frames
        i_h,i_w,ch = imgs[0].shape
        n_h = math.ceil(i_h/m_h)
        n_w = math.ceil(i_w/m_w)                           # 保证了产生的测量数据的维度
        mask_exp=np.tile(mask,(1,n_h,n_w))
        img_exp = np.zeros_like(mask_exp[0])
        for i,img in enumerate(imgs):
            Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            Y_h,Y_w = Y.shape
            img_exp[:Y_h,:Y_w]=Y
            Y = img_exp.astype(np.float32)/self.norm            
            gt.append(Y)
        return np.array(gt)
    

# ---------------------------------------------------------------
#               DSM 模拟实验
# ---------------------------------------------------------------
@PIPELINES.register_module
class GeneDSMMeas:
    def __init__(self,norm=255,s_cr=4): 
        self.norm=norm
        self.s_cr=s_cr
    def __call__(self, imgs, mask):
        assert isinstance(imgs,list),                      "imgs must be list"
        gt = []
        m_cr,m_h,m_w = mask.shape
        i_cr = len(imgs)                                   # return n_frames        
        i_h = imgs[0].shape[0]
        i_w = imgs[0].shape[1]
        assert m_cr==i_cr and m_h==i_h and m_w==i_w,       "Image size does not match mask size! "    # 单组测量值处理，cr帧图像作为一组图像进行处理
        meas = np.zeros_like(mask[0])
        for i,img in enumerate(imgs):       
            if len(img.shape) == 3:     
                Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            else:
                Y=img
            Y = Y.astype(np.float32)/self.norm            
            gt.append(Y)
            meas += np.multiply(mask[i, :, :], Y)          # 采用时空域模板进行调制并产生时域压缩结果
        T0 = tansT_np(mask[0].shape[-1],self.s_cr)
        meas = T0@meas@T0.T                                # 进行空域降采样，时域加和与空域加和的前后顺序并没有影响
        
        
        return np.array(gt),meas  