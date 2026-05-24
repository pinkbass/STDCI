from stci.utils.registry import Registry,build_from_cfg 

DATASETS = Registry("dataset")                                              # 创建了名为DATASETS的注册表对象，指定注册表名称为dataset

def build_dataset(cfg,default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset