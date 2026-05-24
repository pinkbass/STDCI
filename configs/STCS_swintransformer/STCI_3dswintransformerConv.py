_base_=[
        "../_base_/six_gray_sim_data_st.py",       
        "../_base_/davis_stformerLKA.py",         
        "../_base_/default_runtime.py"            
        ]

data = dict(
    samples_per_gpu = 6,                              
    workers_per_gpu=4,                              
)

scr = 4         
if scr == 2:
    resize_h,resize_w = 96, 96
    mask_Bsize = 32
if scr == 3:
    resize_h,resize_w = 132, 132
    mask_Bsize = 33
if scr == 4:
    resize_h,resize_w = 128, 128
    mask_Bsize = 32

data_size = dict(
    sizeH = resize_h,
    sizeW = resize_w
)

path_mask = "test_datasets/mask/STMask.mat"

train_pipeline = [ 
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True,crop_ratio=1),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]

gene_meas = dict(type='GeneSTGrayMeas_dataAug')                    
train_data = dict(
    type = "Davis_stformerLKA_Data",                                  
    data_root = "/home/hxw4/Dataset/DAVIS/Full-Resolution/",
    mask_path = path_mask,      
    mask_shape = (resize_h,resize_w,8),
    mask_blocksize = mask_Bsize,
    pipeline = train_pipeline,                                   
    gene_meas = gene_meas,
    s_cr = scr
)

gens_meas_test = dict(type='GenSTGrayMeasNonBlock') 
test_data = dict(
    type="davis_test_Vimg",                                        
    mask_path= path_mask,
    data_root = "test_datasets/UVG_img_1024/",
    gens_meas_test = gens_meas_test,
)

model = dict(
    type='SwinTransformer3DConv',
    # type='SwinTransformer3DConv_P3D',     # chose model
    patch_size = (1,scr,scr),
    s_cr = scr, 
    in_chans=1,

    # ---STST-3D
    embed_dim=192, 
    depths = [2,4,6,2],
    num_heads=[3, 6, 12, 24], 

    # ---STST-P3D-A
    # embed_dim=192,
    # depths = [2,4,6,4],
    # num_heads=[3, 6, 12, 24],

    
    # ---STST-P3D-B
    # embed_dim=144, 
    # depths = [2,4,6,2],
    # num_heads=[3, 6, 12, 24], 

    window_size=(4,4,4),            
    mlp_ratio=4.                                                                                                                                                                                                                                                                                                 
)

loss_params = dict(
    ParaReLoss   = 1,
    ParaCSLoss   = 1,
    ParaBNLoss   = 0,
    ParaGANLoss  = 0,
    ParaFreLoss  = 0,    
    lambda_lpips = 1,     
    lambda_ssim  = 0,     
)

eval=dict(
    flag=True,
    interval=1
)


checkpoints = None
resume = None