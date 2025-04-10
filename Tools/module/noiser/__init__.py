
# 注册noise layer模块，就是要导入，让其执行装饰器
from .diff_jpeg import DiffJpeg
from .diff_binarization import DiffBinarization
from .diff_common import DiffAffine, DiffColor, DiffGaussianNoise, DiffDegree, DiffTranslate, DiffResize

