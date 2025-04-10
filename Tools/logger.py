# -*- coding: utf-8 -*-
"""
@FILE    :   logger.py
@DATE    :   2023/7/4 20:47
@Author  :   Angel_zou 
@Contact :   ahacgn@gmail.com
@docs    :   
"""
from collections.abc import Mapping
import logging
import os
import colorlog
from logging.handlers import RotatingFileHandler
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import transforms
# 定义不同日志等级颜色
from torch.utils.tensorboard import SummaryWriter

toTensor = transforms.ToTensor()

log_colors_config = {
    'DEBUG': 'bold_cyan',
    'INFO': 'bold_green',
    'WARNING': 'bold_yellow',
    'ERROR': 'bold_red',
    'CRITICAL': 'red',
}


class Logger(logging.Logger):
    def __init__(self, name, file_path=None, main_worker=True):
        super().__init__(name)
        self.encoding = 'utf-8'
        self.level = "DEBUG"
        self.main_worker = main_worker
        if file_path is not None and not os.path.exists(file_path) and main_worker:
            os.makedirs(file_path)
            pass
        # 针对所需要的日志信息 手动调整颜色
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s [%(filename)s:%(''lineno)d] %(log_color)s%(levelname)s: %(message)s',
            reset=True,
            log_colors=log_colors_config,
            secondary_log_colors={
                'message': {
                    'DEBUG': 'blue',
                    'INFO': 'blue',
                    'WARNING': 'blue',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red'
                }
            },
            style='%'
        )  # 日志输出格式

        if file_path is not None and main_worker:
            # 创建一个FileHandler，用于写到本地
            rotatingFileHandler = logging.handlers.RotatingFileHandler(filename=os.path.join(file_path, "_logger.log"),
                                                                       maxBytes=1024 * 1024 * 50,
                                                                       backupCount=5)
            rotatingFileHandler.setFormatter(
                logging.Formatter('%(asctime)s [%(filename)s:%(''lineno)d] %(levelname)s:%(message)s'))
            rotatingFileHandler.setLevel(logging.DEBUG)
            self.addHandler(rotatingFileHandler)

        # 创建一个StreamHandler,用于输出到控制台
        console = colorlog.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        self.addHandler(console)
        self.setLevel(logging.DEBUG)

        if file_path is not None and main_worker:

            self.summary_path = os.path.join(file_path, "_summary")
            if not os.path.exists(self.summary_path):
                os.makedirs(self.summary_path)
                pass
            self.writer = SummaryWriter(self.summary_path)

        pass

    def add_features(self, tag, features, iter=None, format='NCHW'):
        if not self.main_worker:
            return
        self.writer.add_images(tag, features, iter, dataformats=format)
        pass

    def add_model(self, model, input_x):
        if not self.main_worker:
            return
        self.writer.add_graph(model, input_x)
        pass

    def add_scalars(self, tag, x_y_dict, iter=None):
        if not self.main_worker:
            return
        self.writer.add_scalars(tag, x_y_dict, iter)
        pass

    def add_latent_dim(self, latent_dim, label=None, representation=None, iter=None):
        """
        高维空间的低纬映射
        :param latent_dim: 潜空间的值[N, D] 其中N代表有几个数据，D代表潜空间的维度
        :param label: [N]，是list，其中的每一项代表latent_dim对应的label
        :param representation: [N, C, H, W]潜空间向量对应的实际代表图片
        :return:
        """
        if not self.main_worker:
            return
        self.writer.add_embedding(latent_dim, label, representation, global_step=iter)
        pass

    def show_summary(self, port=8080):
        if not self.main_worker:
            return
        assert self.summary_path is not None
        command = "tensorboard --logdir={} --port {}".format(self.summary_path, port)
        os.system(command)
        pass

    def add_hot(self, tag, img, iter=None):
        """
        绘制热力图
        :param tag: 标签
        :param iter: 迭代次数
        :param img: H x W
        :return:
        """
        if not self.main_worker:
            return
        iter = iter if iter is not None else 0
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        feature_map = img.numpy()

        # 获取特征图的最大值和最小值
        max_value = np.max(feature_map)
        min_value = np.min(feature_map)

        # 绘制热力图
        plt.pcolor(feature_map, cmap='hot', vmin=min_value, vmax=max_value)

        # 添加颜色条
        plt.colorbar()

        # 显示图形
        # plt.show()

        plt.savefig("./tmp.png")
        image = toTensor(Image.open("./tmp.png").convert('RGB')).float()
        self.writer.add_image(tag, image, global_step=iter)
        # os.remove("./tmp.png")
        pass
    
    def info(self, msg, *args, **kwargs):
        if not self.main_worker:
            return
        super().info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if not self.main_worker:
            return
        super().warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if not self.main_worker:
            return
        super().error(msg, *args, **kwargs)
        
if __name__ == "__main__":
    logger = Logger("shadow", "test")

    pass
