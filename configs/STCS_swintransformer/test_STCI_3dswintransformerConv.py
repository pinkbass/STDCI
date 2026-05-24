
_base_=[
        "../_base_/six_gray_sim_data_st.py",       # 测试数据集
        "../_base_/default_runtime.py"             # 一些训练的配置
        ]

data = dict(
    samples_per_gpu=1,                             # batch_size
    workers_per_gpu=4,                             # GPU的数据加载器(worker)个数
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


gens_meas_test = dict(type='GenSTGrayMeasNonBlock') 
test_data = dict(
    type="GraySim2stepSTData",
    mask_path = 'test_datasets/mask/STMask.mat',
    data_root = 'test_datasets/Image_UVG_mat',
    mask_shape = (resize_h,resize_w,8),
    mask_blocksize = mask_Bsize, 
    gens_meas_test = gens_meas_test,
    s_cr = scr
)

model = dict(
    type='SwinTransformer3DConv',
    type='SwinTransformer3DConv_P3D',
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


checkpoints = '../best_psnr.pth'
