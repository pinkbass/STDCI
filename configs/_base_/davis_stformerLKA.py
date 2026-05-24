resize_h,resize_w = 256,256
train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True, crop_ratio=0.5),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w)              # 时空域中不需要resize
    ]
gene_meas = dict(type='GenerationSpatiotemporalGrayMeas')                 # 如何产生训练数据

train_data = dict(
    type="Davis_stformerLKA_Data",                                        # ./datasets/davis_stformerLKA.py -- 定义数据集相关
    data_root = "/home/hxw4/Dataset/DAVIS/Full-Resolution/",    
    mask_path="test_datasets/mask/mask_stformerLKA_256.mat",
    pipeline=train_pipeline,
    gene_meas = gene_meas,
    mask_shape = None,
)

