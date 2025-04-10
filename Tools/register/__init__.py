from .registry import Registry

"""
sample use

### datasets.py
# add dataset
@DATASETS.register()
class ImageNet(ImageFolder):
    ...
    pass


### build_datasets.py
# import dataset
from .datasets import ImageNet

# import register hook
from .register import DATASETS


__all__ = [k for k in globals().keys() if not k.startswith("_")]

# load dataset by its name
def build_dataset(obj_type, *args, **kwargs):
    return DATASETS.get(obj_type)(*args, **kwargs)

"""


BACKBONE = Registry("backbone")
DATASET = Registry("dataset")
MODEL = Registry("model")
TRANSFORM = Registry("transform")

