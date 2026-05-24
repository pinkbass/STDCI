import numpy as np 
import scipy.io as scio
import torch 
from stci.utils.utils import tansT_np


def generate_masks(mask_path=None,mask_shape=None,s_cr=4):
    assert mask_path is not None or mask_shape is not None
    if mask_path is None:
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        mask = scio.loadmat(mask_path)
        mask = mask['mask']
        if mask_shape is not None:
            h,w,c = mask.shape
            m_h,m_w,m_c = mask_shape[0],mask_shape[1],mask_shape[2]
            h_b = np.random.randint(0,h-m_h+1)
            w_b = np.random.randint(0,w-m_w+1)
            mask = mask[h_b:h_b+m_h,w_b:w_b+m_w,:m_c]
    mask = np.transpose(mask, [2, 0, 1])
    mask = mask.astype(np.float32)

    t_cr, H, W = mask.shape
    mask_block = mask.reshape(t_cr, H//s_cr, s_cr, W//s_cr, s_cr)
    mask_sum = mask_block.sum(axis=(0,2,4))
    mask_sum[mask_sum==0] = 1

    return mask, mask_sum

def generate_real_masks(frames=10,size_h=512,size_w=512,mask_path=None):
    mask_dict = scio.loadmat(mask_path)
    mask = mask_dict["mask"]
    h,w,f = mask.shape
    if size_h!=h or size_w!=w:
        h_begin = np.random.randint(0,h-size_h)
        w_begin = np.random.randint(0,w-size_w)
        if frames==f:
            f_begin = 0
        else:
            f_begin = np.random.randint(0,f-frames)
        mask = mask[h_begin:h_begin+size_h,w_begin:w_begin+size_w,f_begin:f_begin+frames]
    else:
        mask = mask[:,:,0:frames]
    
    mask = mask.transpose(2,0,1)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1 
    return mask,mask_s

def generate_shipai_masks(frames=8,size_h=128,size_w=128,mask_path=None):
    mask_dict = scio.loadmat(mask_path)
    mask = mask_dict["mask"]
    h,w,f = mask.shape
    if size_h!=h or size_w!=w:
        h_begin = np.random.randint(0,h-size_h)
        w_begin = np.random.randint(0,w-size_w)
        if frames==f:
            f_begin = 0
        else:
            f_begin = np.random.randint(0,f-frames)
        mask = mask[h_begin:h_begin+size_h,w_begin:w_begin+size_w,f_begin:f_begin+frames]
    else:
        mask = mask[:,:,0:frames]
    
    mask = mask.transpose(2,0,1)
    mask = mask.astype(np.float32)
    mask_s = np.sum(mask,axis=0)
    T = tansT_np(mask_s.shape[0],4)
    mask_s=T@mask_s@T.T
    mask_s[mask_s==0] = 1 
    mask_s=mask_s.astype(np.float32)
    return mask,mask_s
# ----------------------------------------------------
#                时域only--针对随机二值模板
# ----------------------------------------------------
def generate_masks_tem(mask_path=None,mask_shape=None):
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        # create random mask
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask = scio.loadmat(mask_path)                          # mask--mat
        mask = mask['mask']
        # change the shape of mask
        if mask_shape is not None:
            h,w,c = mask.shape                                  # 256,256,8/512,512,20
            m_h,m_w,m_c = mask_shape[0],mask_shape[1],mask_shape[2]
            h_b = np.random.randint(0,h-m_h+1)
            w_b = np.random.randint(0,w-m_w+1)
            mask = mask[h_b:h_b+m_h,w_b:w_b+m_w,:m_c]
    mask = np.transpose(mask, [2, 0, 1])                        # hwc -> chw
    mask = mask.astype(np.float32)                              # data type changing

    mask_s = np.sum(mask, axis=0)
    mask_s[mask_s==0] = 1    
    return mask, mask_s                                         # 返回 original mask & 通道加和后的:mask_s-temporal sum mask

# ----------------------------------------------------
#                处理时空域的随机二值模板
# ----------------------------------------------------
def generate_RandomST_masks(mask_path = None, mask_shape=None):    
    """
    random mask:[cr H W]
    """
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        # create random mask
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask_all = scio.loadmat(mask_path)                              # mask--mat
        mask = mask_all['mask_st']                                         # [256 256 8]
        
        # select the effective range for mask
        if mask_shape is not None:
            H, W, _ = mask.shape
            h, w, cr = mask_shape[0], mask_shape[1], mask_shape[2]
            h_ = np.random.randint(0, H-h+1)
            w_ = np.random.randint(0, W-w+1)
            mask = mask[h_:h_+h, w_:w_+w,:cr]
            mask_position = [h_, w_]      
    
    mask = np.transpose(mask, [2, 0, 1])                                    # hwc -> chw
    mask = mask.astype(np.float32)                                          # data type changing

    mask_sum = np.sum(mask, axis=0)                                    
    T = tansT_np(mask_sum.shape[0],4)
    mask_sum = T@mask_sum@T.T
    mask_sum[mask_sum==0] = 1
    mask_sum = mask_sum.astype(np.float32)

    return mask, mask_sum, mask_position

# ----------------------------------------------------
#      zlx采用的大分块(32)模板--只返回时空域联合模板
# ---------------------------------------------------- 
def generate_BlockRandomST_masks(mask_blocksize=32, mask_path = None, mask_shape=None, s_cr=4):    
    """
        random mask:[cr H W]
    """
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        # create random mask
        mask_st = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask_all = scio.loadmat(mask_path)                              # mask--mat
        mask = mask_all['mask_st']                                         # [256 256 8]
        if len(mask.shape) > 3:
            mask = mask[0]
        # select the effective range for mask
        if mask_shape is not None:
            n_h = mask_shape[0]//mask_blocksize
            n_w = mask_shape[1]//mask_blocksize
            cr = mask_shape[2]
            mask_block = mask[0:mask_blocksize,0:mask_blocksize,:cr]
            mask_st = np.tile(mask_block,(n_h, n_w, 1))
    
    mask_st = np.transpose(mask_st, [2, 0, 1])                                    # hwc -> chw
    mask_st = mask_st.astype(np.float32)                                          # data type changing

    t_cr, H, W = mask_st.shape
    mask_block = mask_st.reshape(t_cr, H//s_cr, s_cr, W//s_cr, s_cr)
    mask_sum = mask_block.sum(axis=(0,2,4))
    mask_sum[mask_sum==0] = 1
   
    mask_position = [0,0]
    
    return mask_st, mask_sum, mask_position

# ----------------------------------------------------
#                产生时域调制模板-birnatrdn只进行时域调制时使用的代码
# ---------------------------------------------------- 
def gen_tem_masks(mask_path = None, mask_shape=None, mask_blocksize=None):    
    """
    random mask:[cr H W]
    """
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        mask_all = scio.loadmat(mask_path)                              # mask--mat
        mask = mask_all['mask']                                         # [256 256 8]

        if mask_shape is not None:
            cut_h = mask_shape[0]
            cut_w = mask_shape[1]
            nh = cut_h//mask_blocksize
            nw = cut_w//mask_blocksize
            cr = mask_shape[2]
            mask_block = mask[0:mask_blocksize,0:mask_blocksize,:cr]
            mask_block = np.tile(mask_block, (nh, nw, 1))
            mask_block = np.transpose(mask_block, [2, 0, 1])
            mask_block = mask_block.astype(np.float32)
    return mask_block

# -------------------------------------------------------------------------------
#           2stepST：指时域模板与空域模板是独立的，两者相乘得到时空域模板
#                  针对在birnatRDN中使用两步的模板需求的模板处理
# -------------------------------------------------------------------------------
def gene_birrdn_mask(mask_path = None, mask_shape=None, mask_blocksize=None):    
    """
        measurement needed:
        spatiotemporal mask:   [256 256 8]
        reconstruction needed:
        temporal mask:         [64 64 8]
        temporal mask sum:     [64 64]
        spatial mask:          [4 4 8]
    """
    s_cr = 4
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        # create random mask
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask_all = scio.loadmat(mask_path)                           # mask--mat
        # mask = mask_all['mask_st']                                   # [256 256 8]
        mask_spatial = mask_all['mask_s']                            # [4 4 8]
        mask_spatial = mask_spatial[:,:,0]
        mask_temporal = mask_all['mask_t']                           # [64 64 8]
        mask_temporal = mask_temporal.transpose(2,0,1)        
        # select the effective range for mask
        if mask_shape is not None:
            cut_h = mask_shape[0]
            cut_w = mask_shape[1]
            nh = cut_h//mask_blocksize                               # 按照32分块的分块数
            nw = cut_w//mask_blocksize
            cr = mask_shape[2]
       
        mask_tem_blocksize = mask_blocksize//s_cr
        mask_tem_block = mask_temporal[:cr, :mask_tem_blocksize, :mask_tem_blocksize]   

        exp_matrix = np.ones((4,4))
        mask_tblock_exp = np.kron(mask_tem_block, exp_matrix)               # 对后两个维度进行复制展开
        mask_spa_block = np.tile(mask_spatial, (mask_tem_blocksize, mask_tem_blocksize))
        mask_block = mask_tblock_exp*mask_spa_block
        
        mask_block = np.tile(mask_block, (1, nh, nw))
        mask_tem_block = np.tile(mask_tem_block, (1, nh, nw))

        # mask_block = np.transpose(mask_block,[2,0,1])
        mask_block = mask_block.astype(np.float32)
        # mask_tem_block = np.transpose(mask_block,[2,0,1])
        mask_tem_block = mask_tem_block.astype(np.float32)
        mask_spatial = mask_spatial.astype(np.float32)

        mask_sum = np.sum(mask_block, axis=0)
        T = tansT_np(mask_sum.shape[0], s_cr)
        mask_sum = T@mask_sum@T.T
        mask_sum[mask_sum==0] = 1
        mask_sum = mask_sum.astype(np.float32)        

    return mask_block, mask_sum, mask_spatial, mask_tem_block

def gene_STmask_all(mask_blocksize, mask_path, mask_shape, s_cr):
    assert mask_path is not None or mask_shape is not None
    if mask_path is None:  
        # create random mask
        mask_st = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask_all = scio.loadmat(mask_path)                              # mask--mat
        mask = mask_all['mask_st'] 
        mask_spatial = mask_all['mask_s']                            # [4 4 8]
        mask_temporal = mask_all['mask_t']

        if mask_shape is not None:
            mask_block = mask[:mask_blocksize, :mask_blocksize, :]
            min_blocksize = mask_blocksize//s_cr
            mask_tem_block = mask_temporal[:min_blocksize, :min_blocksize, :]
    nh = mask_shape[0]//mask_blocksize
    mask_block = np.transpose(mask_block, [2,0,1])
    mask_tem_block = np.transpose(mask_tem_block, [2,0,1])
    mask_spa = np.transpose(mask_spatial, [2,0,1]).astype(np.float32)
    mask_st = np.tile(mask_block, (nh,nh)).astype(np.float32)
    mask_tem = np.tile(mask_tem_block, (nh,nh)).astype(np.float32)
    mask_sum = mask_st.reshape(mask_shape[-1], mask_shape[0]//s_cr, s_cr, mask_shape[1]//s_cr, s_cr)
    mask_sum = torch.from_numpy(mask_sum).sum(dim=(0,2,4))
    mask_sum[mask_sum==0] = 1
    
    return mask_st, mask_sum, mask_tem, mask_spa

# -------------------------------------------------------------------------
#                        mask_BirnatRDN：返回时/空/时空模板
# -------------------------------------------------------------------------
def generate_2stepST_masks(mask_path = None, mask_shape=None, mask_blocksize=None):    
    """
    measurement needed:
    spatiotemporal mask:   [256 256 8]
    reconstruction needed:
    temporal mask:         [64 64 8]
    temporal mask sum:     [64 64]
    spatial mask:          [4 4 8]
    """
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        # create random mask
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask_all = scio.loadmat(mask_path)                           # mask--mat
        mask = mask_all['mask_st']                                   # [256 256 8]
        mask_spatial = mask_all['mask_s']                            # [4 4 8]
        mask_temporal = mask_all['mask_t']                           # [64 64 8]
        
        # select the effective range for mask
        if mask_shape is not None:
            h_t, w_t, c = mask_temporal.shape
            m_ht, m_wt, m_c = mask_shape[0]//4, mask_shape[1]//4, mask_shape[2]
            h_bt = np.random.randint(0, h_t-m_ht+1)
            w_bt = np.random.randint(0, w_t-m_wt+1)
            mask_temporal = mask_temporal[h_bt:h_bt+m_ht, w_bt:w_bt+m_wt,:m_c]
            mask_position = [h_bt, w_bt]      

            # h,w,c = mask.shape                                        # 256,256,8/512,512,20
            m_h,m_w,m_c = mask_shape[0],mask_shape[1],mask_shape[2]
            h_b = h_bt * 4
            w_b = w_bt * 4
            mask = mask[h_b:h_b+m_h,w_b:w_b+m_w,:m_c]           

    mask = np.transpose(mask, [2, 0, 1])                                    # hwc -> chw
    mask = mask.astype(np.float32)                                          # data type changing

    mask_sum = np.sum(mask, axis=0)                                    
    T = tansT_np(mask_sum.shape[0],4)
    mask_sum = T@mask_sum@T.T
    mask_sum[mask_sum==0] = 1
    mask_sum = mask_sum.astype(np.float32)

    mask_spatial = np.transpose(mask_spatial, [2, 0, 1])             # hwc -> chw
    mask_spatial = mask_spatial.astype(np.float32)                   # data type changing
    mask_temporal = np.transpose(mask_temporal, [2, 0, 1])           # hwc -> chw
    mask_temporal = mask_temporal.astype(np.float32)                 # data type changing

    # return original mask & mask_s & spatial block mask 
    return mask, mask_sum, mask_spatial, mask_temporal, mask_position

# generate mask for TM method, the result includes the spatiotemporal mask and the summation version
def generate_tm_masks(mask_path = None, mask_shape=None):
    """
    measurement needed:
    spatiotemporal mask:   [256 256 8]
    reconstruction needed:
    temporal mask:         [64 64 8]
    temporal mask sum:     [64 64]
    spatial mask:          [4 4 8]
    """
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        # create random mask
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask_all = scio.loadmat(mask_path)                           # mask--mat
        mask = mask_all['Phi']                                       # [256 256 8]
        
        # select the effective range for mask
        if mask_shape is not None:
            c, h, w = mask.shape                                     # shape of the mask is [c h w]
            m_c, m_h, m_w = mask_shape[0], mask_shape[1], mask_shape[2]
            h_b = np.random.randint(0, h-m_h+1)
            w_b = np.random.randint(0, w-m_w+1)
            mask = mask[:m_c, h_b:h_b+m_h, w_b:w_b+m_w]              
            mask_position = [h_b, w_b]

    mask = mask.astype(np.float32)                                    # data type changing

    mask_sum = np.sum(mask, axis=0)                                    
    T = tansT_np(mask_sum.shape[0],4)
    mask_sum = T@mask_sum@T.T
    mask_sum[mask_sum==0] = 1
    mask_sum = mask_sum.astype(np.float32)

    # return original mask & mask_s 
    return mask, mask_sum, mask_position

def generate_2stage_masks(mask_path=None,mask_shape=None):
    """
    仅对时域模板进行shape transform以及加和操作
    measurement needed:
    spatiotemporal mask:   [256 256 8]
    
    reconstruction needed:
    temporal mask:         [64 64 8]
    temporal mask sum:     [64 64]
    spatial mask:          [4 4 8]
    """
    assert mask_path is not None or mask_shape is not None   
    if mask_path is None:  
        # create random mask
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        # loading mask data
        mask_all = scio.loadmat(mask_path)                           # mask--mat
        mask = mask_all['mask_st']                                   # [256 256 8]
        mask_spatial = mask_all['mask_s']                            # [4 4 8]
        mask_temporal = mask_all['mask_t']                           # [64 64 8]
        # change the shape of mask
        if mask_shape is not None:
            h,w,c = mask.shape                                       # 256,256,8/512,512,20
            m_h,m_w,m_c = mask_shape[0],mask_shape[1],mask_shape[2]
            h_b = np.random.randint(0,h-m_h+1)
            w_b = np.random.randint(0,w-m_w+1)
            mask = mask[h_b:h_b+m_h,w_b:w_b+m_w,:m_c]
    mask = np.transpose(mask, [2, 0, 1])                             # hwc -> chw
    mask = mask.astype(np.float32)                                   # data type changing

    mask_s = np.sum(mask, axis=0)                                    
    T = tansT_np(mask_s.shape[0],4)
    mask_s = T@mask_s@T.T
    mask_s[mask_s==0] = 1

    mask_spatial = np.transpose(mask_spatial, [2, 0, 1])             # hwc -> chw
    mask_spatial = mask_spatial.astype(np.float32)                   # data type changing
    mask_temporal = np.transpose(mask_temporal, [2, 0, 1])           # hwc -> chw
    mask_temporal = mask_temporal.astype(np.float32)                 # data type changing

    mask_temproal_s = np.sum(mask_temporal, axis=0)
    mask_temproal_s[mask_temproal_s==0] = 1

    # return original mask & mask_s & spatial block mask 
    return mask, mask_s, mask_spatial, mask_temporal, mask_temproal_s
