import torch
import numpy as np
import cv2
import os.path as osp
import einops
from stci.utils.demosaic import demosaicing_CFA_Bayer_Menon2007 as demosaicing_bayer
import torch.nn as nn
import pywt
import torch.nn.functional as F

def get_device_info():
    gpu_info_dict = {}
    if torch.cuda.is_available():
        gpu_info_dict["CUDA available"]=True
        gpu_num = torch.cuda.device_count()
        gpu_info_dict["GPU numbers"]=gpu_num
        infos = [{"GPU "+str(i):torch.cuda.get_device_name(i)} for i in range(gpu_num)]
        gpu_info_dict["GPU INFO"]=infos
    else:
        gpu_info_dict["CUDA_available"]=False
    return gpu_info_dict
    
def load_checkpoints(model,pretrained_dict,strict=False):
    # pretrained_dict = torch.load(checkpoints)
    if strict is True:
        try: 
            model.load_state_dict(pretrained_dict)
        except:
            print("load model error!")
    else:
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,strict=False)

def save_image(out,gt,image_name,show_flag=False):
    if len(out.shape)==4:
        out = einops.rearrange(out,"c f h w->h (f w) c")
        gt = einops.rearrange(gt,"c f h w->h (f w) c")
        result_img = np.concatenate([out,gt],axis=0)
        result_img = result_img[:,:,::-1]
    if len(out.shape)==2:
        result_img = np.concatenate([out,gt],axis=0)
    else:
        out = einops.rearrange(out,"f h w->h (f w)")
        gt = einops.rearrange(gt,"f h w->h (f w)")
        result_img = np.concatenate([out,gt],axis=0)
    result_img = result_img*255.
    cv2.imwrite(image_name,result_img)
    
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)

def save_image2(out, image_name, show_flag=False):
    # 这里的if-else结构有问题，因为如果out.shape==4时，执行完第一个if后，仍然会继续判断第二个if和else。
    # 正确的做法是使用if-elif-else结构，保证只会进入一个分支。
    if len(out.shape) == 4:
        result_img = einops.rearrange(out, "c f h w->h (f w) c")
    elif len(out.shape) == 2:
        result_img = out
    else:
        result_img = einops.rearrange(out, "f h w->h (f w)")

    result_img = result_img*255.
    cv2.imwrite(image_name,result_img)
    
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)

def save_single_image(images,image_dir,batch,name="",demosaic=False):
    images = images*255.
    if len(images.shape)==4:
        frames = images.shape[1]
    else:
        frames = images.shape[0]
    for i in range(frames):
        begin_frame = batch*frames
        if len(images.shape)==4:
            single_image = images[:,i].transpose(1,2,0)[:,:,::-1]
        else:
            single_image = images[i]
        if demosaic:
            single_image = demosaicing_bayer(single_image,pattern='BGGR')
        cv2.imwrite(osp.join(image_dir,name+"_"+str(begin_frame+i+1)+".png"),single_image)        

# 图像域映射到压缩域
def A(x,Phi,s_cr):
    temp = x*Phi
    B, T, H, W = Phi.shape
    temp = temp.reshape(B, T, H//s_cr, s_cr, W//s_cr, s_cr)
    y = temp.sum(dim=(1,3,5))
    return y.unsqueeze(1)

# 压缩域映射到图像域
def At(y,Phi,s_cr):
    # 维度扩展
    y_exp = y.repeat_interleave(repeats=s_cr, dim=-1).repeat_interleave(repeats=s_cr, dim=-2)   # 元素复制扩展
    # 乘模板
    x = y_exp*Phi
    return x

def tansT_tensor(m,r):
    n=int(m/r)
    T=torch.zeros((n,m))
    for i in range(n):
        T[i,r*i:r*(i+1)]=1
    return T

def tansT_np(m,r):
    n=int(m/r)
    T=np.zeros((n,m))
    for i in range(n):
        T[i,r*i:r*(i+1)]=1
    return T

def ST_init(y, phi, T0, radio=4):
    mask_tsum = torch.sum(phi, dim=1, keepdim=True)
    mask_stsum = T0@mask_tsum@T0.T
    mask_stsum[mask_stsum==0] = 1                                    # 计算时空压缩的mask的值作为data normlization的数值
    y_norm = torch.div(y, mask_stsum)
    y_norm = T0.T@y_norm@T0
    y_init = phi * y_norm
    return y_init
    

# 只在空域进行调制与压缩,产生空域压缩的时域多帧图像
def st_modulation_t_init(imgs, mask_spatial):                                                # imgs-[1 8 256 256]    mask_spatial--[b,cr,4,4]
    mask_s_large = mask_spatial.repeat(1, 1, imgs.shape[2]//4, imgs.shape[3]//4)             # [b,cr,h',w'] h'=h//4
    mea_temp = imgs*mask_s_large                       
    
    T0 = tansT_tensor(imgs[0].shape[-1],4).to(imgs.device)
    mea = T0@mea_temp@T0.T

    mask_spatial_sum = torch.sum(mask_spatial, dim=2, keepdim=True)
    mask_spatial_sum = torch.sum(mask_spatial_sum, dim=3, keepdim=True)
    mask_spatial_sum = mask_spatial_sum.repeat(1,1,imgs.shape[2]//4,imgs.shape[3]//4)

    mea = torch.div(mea, mask_spatial_sum)
    return mea

# def st_modulation_t_init(imgs, mask, mask_s):                                           # imgs-[1 8 256 256]    mask--[b,cr,h,w]   mask_s
#     mask_s_large = mask_s.repeat(mask.shape[0], 1, mask.shape[2]//4, mask.shape[3]//4)  # [b,cr,h',w'] h'=h//4
#     mea_temp = imgs*mask_s_large
#     T0 = tansT_tensor(mask[0].shape[-1],4).to(imgs.device)
#     mea = T0@mea_temp@T0.T
#     return mea

# 对输入在时空域进行编码压缩
def st_modulation_stcs0(imgs, mask, mask_sum):
    mea_temp = imgs*mask

    T0 = tansT_tensor(imgs[0].shape[-1], 4).to(imgs.device)
    mea = T0@mea_temp@T0.T

    mea = mea.sum(dim=1, keepdim=True)
    mea = torch.div(mea, mask_sum)

    return mea

def st_modulation_stcs(imgs_batch, mask_batch, mask_sum_batch,s_cr=4):             # imgs[b,cr,H,W]/mask[b,cr,H,W]/mask_sum[b,1,h,w]
    mask=mask_batch[0]
    mask_sum=mask_sum_batch[0]
    mea_batch=torch.zeros(imgs_batch.shape[0],1,mask_sum_batch.shape[2],mask_sum_batch.shape[3]).to(imgs_batch.device)
    for i in range(imgs_batch.shape[0]):
        imgs=imgs_batch[i,:,:,:]
        
        mea_temp = imgs*mask

        T0 = tansT_tensor(imgs[0].shape[-1], s_cr).to(imgs.device)
        mea = T0@mea_temp@T0.T

        mea = mea.sum(dim=0, keepdim=True)
        mea = torch.div(mea, mask_sum)
        mea_batch[i,:,:,:]=mea.unsqueeze(1)
    return mea_batch

# 对测量值进行时空域的初始化
def st_init(y, phi, phi_s, radio):
    phi_s = phi_s.float()
    T0 = tansT_tensor(phi[0].size(-1), radio)
    T0 = T0.to(y.device)
    y1 = y/phi_s                             # 模板归一化
    # y1=y
    y2 = T0.T@y1@T0                          # [64 64]--[256 256]
    x = phi * y2
    return x, y1                                 # return the spatiotemporal init result

# 对stci_mea只进行时域初始化--模板是2-step mask
def st_tInit(y,phi_s):
    phi_s = phi_s.float()
    y1 = y/phi_s                             # 
    return y1

def tem_init(y, phi_tem):
    """
    y.shape=[1,1,64,64]
    phi_tem.shape=[8,64,64]
    """
    phi_s = torch.sum(phi_tem,0)
    phi_s = phi_s.float()
    phi_s[phi_s==0] = 1
    y1 = y/phi_s                # 模板归一化
    x = phi_tem * y1
    return x, y1                                # return the spatiotemporal init result

def tem_init2(y, phi_stsum, phi_t):
    """
    y.shape=[1,1,64,64]
    phi_tem.shape=[8,64,64]
    """
    y1 = y/phi_stsum
    x = phi_t * y1
    return x, y1   

def spa_cs3(y, s_cr=4):
    T0 = tansT_tensor(y.shape[-1], s_cr)
    T0 = T0.to(y.device)
    T1 = tansT_tensor(y.shape[-2], s_cr)
    T1 = T1.to(y.device)
    x = T1@y@T0.T
    x = x/16.
    return x

def spa_cs(y, s_cr=4):
    T0 = tansT_tensor(y.shape[-1], s_cr)
    T0 = T0.to(y.device)
    x = T0@y@T0.T
    x = x/16.
    return x

def spa_cs2(y, s_kernel):               # 使用空域模板进行降采样--使用一个空域模板
    kernel = s_kernel
    x = F.conv2d(y, kernel, stride=4, groups=1)
    ssum = kernel.sum()
    x = x/ssum
    return x

def block_unshuffle(x):
    pixelunshuffle = nn.PixelUnshuffle(8)
    pixelshuffle = nn.PixelShuffle(4)
    x = torch.tensor(x).unsqueeze(0)    
    data_cat = []
    data_f = torch.tensor([], dtype=torch.float)
    k=0
    for i in range(8):
        data_cat = []
        x_i = x[:1,i,:,:]                        # [1 8 120 120]
        x_d = pixelunshuffle(x_i)
        if k//2 == 0:
            for j in range (0,32,8):
                data_cat.append(x_d[j:j+4,:,:])
        if k//2 == 1:
            for j in range (4,32,8):
                data_cat.append(x_d[j:j+4,:,:])
        if k//2 == 2:
            for j in range (36,64,8):
                data_cat.append(x_d[j:j+4,:,:])
        if k//2 == 3:
            for j in range (32,64,8):
                data_cat.append(x_d[j:j+4,:,:])
        data_tensor = data_cat[0]
        for i in range(1,4):
            data_tensor = torch.cat([data_tensor, data_cat[i]], dim=0)
        data_cat = pixelshuffle(data_tensor)
        data_f = torch.cat([data_f,data_cat],dim=0)
        k += 1
    data_f = data_f.squeeze(0).numpy()
    return data_f

def spatial_init(y, phi_spatial, size_l, radio):
    """
    y-----y_expand
    phi_spatial-----phi_spatial_expand
    """
    T0 = tansT_tensor(size_l, radio)
    T0 = T0.to(y.device)                                            # [M N]
    y1 = T0.T@y@T0                                                  # [64 64] --> [256 256]
    phi = phi_spatial.repeat(1, 1, 1, size_l//4, size_l//4)         # [4 4] --> [256 256]
    x = phi * y1
    return x

# 梯度信息计算
def gradient_sobel(Y):
    # Y.shape=[bs, 1, h, w]
    # bs = Y.shape[0]
    # for i in range(bs):
    #     Ysingle = Y[i,0,:,:]
    sobelx = cv2.Sobel(Y, cv2.CV_64F,1,0,ksize=3)     # sobel算子是针对numpy数据计算的
    sobely = cv2.Sobel(Y, cv2.CV_64F,0,1,ksize=3)
    abs_sobelx = np.float64(np.absolute(sobelx))
    abs_sobely = np.float64(np.absolute(sobely))
    gradient_magnitude = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    return gradient_magnitude

# 基于时空域模板对输入图像进行时空域压缩
def stcs(imgs_batch, mask, s_cr):
    """
    imgs_batch.shape[b, cr, H, W]
    mask.shape[cr, H, W]
    """
    b, t_cr, H, W = imgs_batch.shape
    mask_sblock = mask.reshape(t_cr, H // s_cr, s_cr, W // s_cr, s_cr)
    mask_stcs = mask_sblock.sum(dim=(0, 2, 4))   # [h, w]
    mask_stcs[mask_stcs == 0] = 1

    mea_temp = imgs_batch * mask.unsqueeze(0)           # [b, cr, H, W]
    mea_temp = mea_temp.reshape(b, t_cr, H // s_cr, s_cr, W // s_cr, s_cr)
    mea = mea_temp.sum(dim=(1, 3, 5))                   # [b, h, w]
    mea = mea / mask_stcs.unsqueeze(0)                   # broadcast divide

    return mea.unsqueeze(1)                              # [b, 1, h, w]

# 基于时空域模板对输入图像进行时空域压缩
def stcs_ssr(imgs_batch, mask, s_cr):
    """
    imgs_batch.shape[b, cr, H, W]
    mask.shape[cr, H, W]
    """
    t_cr, H, W = mask.shape
    mask_sblock = mask.reshape(t_cr, H//s_cr, s_cr, W//s_cr, s_cr)
    mask_stcs = mask_sblock.sum(dim=(0, 2, 4))   # [h, w]
    mask_stcs[mask_stcs==0] = 1
    mea_batch = torch.zeros(imgs_batch.shape[0], mask_stcs.shape[0], mask_stcs.shape[1])
    for i in range(imgs_batch.shape[0]):
        imgs=imgs_batch[i,:,:,:]        
        mea_temp = imgs*mask
        mea_temp = mea_temp.reshape(t_cr, H//s_cr, s_cr, W//s_cr, s_cr)
        mea = mea_temp.sum(dim=(0, 2, 4))
        # mea = torch.div(mea, mask_stcs)
        mea_batch[i,:,:]=mea.unsqueeze(0)
    return mea_batch.unsqueeze(1)

def stcs2(imgs_batch, mask_tem, s_cr):    # 输入一个未空间扩展的小规模mask_tem矩阵
    """
    imgs_batch.shape[b, cr, H, W]
    mask.shape[cr, H, W]
    """
    mask = mask_tem.repeat_interleave(repeats=s_cr, dim=1).repeat_interleave(repeats=s_cr, dim=2) 
    t_cr, H, W = mask.shape
    mask_sblock = mask.reshape(t_cr, H//s_cr, s_cr, W//s_cr, s_cr)
    mask_stcs = mask_sblock.sum(dim=(0, 2, 4))   # [h, w]
    mask_stcs[mask_stcs==0] = 1                  # 时空加和后的模板
    mea_batch = torch.zeros(imgs_batch.shape[0], mask_stcs.shape[0], mask_stcs.shape[1])
    for i in range(imgs_batch.shape[0]):
        imgs=imgs_batch[i,:,:,:]        
        mea_temp = imgs*mask
        mea_temp = mea_temp.reshape(t_cr, H//s_cr, s_cr, W//s_cr, s_cr)
        mea = mea_temp.sum(dim=(0, 2, 4))
        mea = torch.div(mea, mask_stcs)
        mea_batch[i,:,:]=mea.unsqueeze(0)
    return mea_batch.unsqueeze(1)
   

# 对测量值进行时空初始化
def st_init1(y, phi, s_cr):
    """
    y.shape=[b, 1, h, w]
    phi.shape=[cr, H, W]
    """
    y_exp = y.repeat_interleave(repeats=s_cr, dim=2).repeat_interleave(repeats=s_cr, dim=3)   # 元素复制扩展
    y_out = y_exp * phi
    # y_out_t = y_exp * phi_t
    return y_out

def fft_magnitude(x):
    x_fft = torch.fft.fftn(x,dim=(-3,-2-1))
    magnitude = torch.abs(x_fft)
    return magnitude

def fft_loss_func(reX, X):
    reX_mag = fft_magnitude(reX)
    X_mag = fft_magnitude(X)

    loss = F.l1_loss(reX_mag, X_mag)
    return loss
    