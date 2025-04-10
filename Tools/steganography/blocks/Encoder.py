import torch
import torch.nn as nn
import torch.nn.functional as F

from Tools.module.common import MLP, ConvBNRelu, ExpandNet, ReShape, ResBlock
from Tools.module.SENet import SENet
from Tools.util.utils import cal_params

"""
信息隐写编码器集合
"""

class SENet2D_ST_Encoder(nn.Module):
	'''
	向图片中插入秘密信息，返还嵌入水印的图像
	默认128x128
	'''

	def __init__(self,  
			  message_length, 
			  in_channel=3, 
			  out_channel=3, 
			  blocks=4, 
			  channels=64, 
			  diffusion_length=256,
			  msg_upsample=3
			  ):
		super().__init__()

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(diffusion_length ** 0.5)

		self.image_pre_layer = ConvBNRelu(in_channel, channels)
		self.image_first_layer = SENet(channels, channels, blocks=blocks)

		self.message_duplicate_layer = nn.Linear(message_length, self.diffusion_length)
		
		self.message_pre_layer_0 = ConvBNRelu(1, channels)
		self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=msg_upsample)
		self.message_pre_layer_2 = SENet(channels, channels, blocks=1)
		self.message_first_layer = SENet(channels, channels, blocks=blocks)

		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + in_channel, out_channel, kernel_size=1)

	def forward(self, image, message):
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)
		message_duplicate = self.message_duplicate_layer(message)
		message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)
		message_pre_0 = self.message_pre_layer_0(message_image)
		message_pre_1 = self.message_pre_layer_1(message_pre_0)
		message_pre_2 = self.message_pre_layer_2(message_pre_1)
		intermediate2 = self.message_first_layer(message_pre_2)
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)
		intermediate3 = self.after_concat_layer(concat1)
		concat2 = torch.cat([intermediate3, image], dim=1)
		output = self.final_layer(concat2)
		return output

if __name__ == "__main__":
	import numpy as np
	bs = 2
	msg_len = 30
	img_size = 128
	model = SENet2D_ST_Encoder(msg_len)
	messages = np.random.choice([0, 1], (bs, msg_len))
	msg = torch.from_numpy(messages).float()
	img = torch.randn(bs, 3, img_size, img_size)
	x = model(img, msg)
	cal_params(model, need_print=True)
	
	pass

