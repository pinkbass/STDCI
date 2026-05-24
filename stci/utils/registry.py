import inspect
import six

'''
实现注册表以及由配置构建对象的函数(build_from_cfg)
'''

def is_str(x):
    return isinstance(x, six.string_types)    # 兼容python2与python3中的字符串类型/保证在py2/3环境中都可以正确判断字符串

class Registry(object):

    def __init__(self, name):                 # 实例化registry对象时自动调用-通过name识别不同的注册表-设置注册表的名称和模块字典
        self._name = name    
        self._module_dict = dict()            # 初始化时设置空字典，后续存储注册的模块

    @property
    def name(self):  
        return self._name                     # 使得定义的方法可以像属性一样被访问，而不需要在调用时加上括号，提供只读接口实现访问A.name
    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):                       # 提供一个get方法，用于从注册表中检索已注册的模块    
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):       # _xx中的'_'表示内部使用方法
        module_name = module_class.__name__  
        if module_name in self._module_dict:  
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class  

    def register_module(self, cls):      
        self._register_module(cls)                  # 调用内部的 _register_module 方法来执行实际的注册逻辑
        return cls

def build_from_cfg(cfg, registry, default_args=None):                                # registry：注册表对象-用于获取创建的对象的类
    assert isinstance(cfg, dict) and 'type' in cfg                                   # 确保 cfg 是一个字典，并且必须包含 'type' 键。'type' 键用于指定要实例化的类名。
    assert isinstance(default_args, dict) or default_args is None   
    # print(registry)
    args = cfg.copy()
    obj_type = args.pop('type')
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
        # print(obj_cls)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):                          # obj_type本身就是一个类对象，则直接对类对象进行实例化
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():                                     # default_args='mask'
            args.setdefault(name, value)
    return obj_cls(**args)                                                           # 实例化的具体过程，obj_cls是通过注册表或直接传入方式得到的类对象/**args位python的字典解包语法