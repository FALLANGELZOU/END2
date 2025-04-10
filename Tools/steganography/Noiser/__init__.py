import os
import importlib
module_path = os.path.join(os.path.dirname(__file__), "Module")

# 导入所有Module下的模块，注册噪声层
from .Module.Common import *
from .Module.JPEGCompression import *
from .Module.Crop import *
from .Module.Filter import *
from .Module.Geometric import *
from .Module.TPSGeometric import *