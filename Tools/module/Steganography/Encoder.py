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
		intermediate2 = F.interpolate(intermediate2, size=(intermediate1.size(2), intermediate1.size(3)), mode='bilinear', antialias=True)

		concat1 = torch.cat([intermediate1, intermediate2], dim=1)
		intermediate3 = self.after_concat_layer(concat1)
		concat2 = torch.cat([intermediate3, image], dim=1)
		output = self.final_layer(concat2)
		return output

# 待定
# class GenericTemplate_ST_Encoder(nn.Module):
# 	'''
# 	无载体水印生成，输出长宽默认与msg长度相同
# 	待定
# 	'''
# 	def __init__(self, msg_len, output_size=None, output_dim=1, hidden_dim=32):
# 		super().__init__()
# 		if output_size == None:
# 			output_size = msg_len
# 		self.fc = nn.Sequential(
# 			nn.LayerNorm(msg_len),
# 			MLP(msg_len, output_size, norm='ln', act='relu')
# 		)

# 		self.conv = nn.Sequential(
# 			ConvBNRelu(1, hidden_dim),
# 			SENet(hidden_dim, hidden_dim, blocks=2),
# 			ConvBNRelu(hidden_dim, hidden_dim),
# 			nn.Conv2d(hidden_dim, output_dim, 1, 1)
# 		)
# 		pass

# 	def generate_matrix(self, msg):
# 		bs, n = msg.size()
# 		feat = torch.zeros(bs, n, n).to(device=msg.device)
# 		feat[:,0] = msg
# 		for i in range(1, n):
# 			msg = torch.roll(msg, 1, dims=[-1])
# 			feat[:, i] = msg
# 			pass
# 		return feat.unsqueeze(1)
# 		pass

# 	def forward(self, msg):
# 		msg = self.fc(msg)
# 		matrix = self.generate_matrix(msg)
# 		watermark = self.conv(matrix)
# 		return watermark
# 		pass
# 	pass


class T1_ST_Encoder(nn.Module):
	def __init__(self, in_channel, msg_len, hidden_dim=64, msg_upsample=1):
		super().__init__()
		self.projector_layer = nn.Conv2d(in_channel, hidden_dim, 1)
		self.msg_encoder = [nn.Sequential(
			nn.Linear(msg_len, 256),
			ReShape(1, 16, 16),
			ResBlock(1, 16),
			ResBlock(16, hidden_dim),
		)]
		for _ in range(msg_upsample):
			self.msg_encoder.append(nn.Sequential(
				nn.Upsample(scale_factor=2),
				ConvBNRelu(hidden_dim, hidden_dim)
			))
			pass
		self.msg_encoder = nn.Sequential(*self.msg_encoder)
		self.img_encoder = nn.Sequential(
			ResBlock(hidden_dim, hidden_dim),
			SENet(hidden_dim, hidden_dim),
			ConvBNRelu(hidden_dim, hidden_dim)
		)
		self.predictor = nn.Sequential(
			ConvBNRelu(hidden_dim*2, hidden_dim),
			SENet(hidden_dim,hidden_dim),
			ResBlock(hidden_dim,hidden_dim),
			nn.Conv2d(hidden_dim, in_channel, 1)
		)
		pass
	def forward(self, img, msg):
		origin_img = img
		img = self.projector_layer(img)
		img = self.img_encoder(img)
		msg = self.msg_encoder(msg)
		if msg.size(2) != img.size(2):
			msg = F.interpolate(msg, size=(img.size(2), img.size(3)), mode='bilinear', antialias=True)
			pass
		watermark = self.predictor(torch.cat([msg, img], dim=1))
		return watermark + origin_img
		pass
	pass
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
	print(x.size())
	pass

