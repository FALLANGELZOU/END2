import torch
import torch.nn as nn

from Tools.module.common import ConvBNRelu
from Tools.module.SENet import SENet, SENet_decoder
import torch.nn.functional as F

class SENet2D_ST_Decoder(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, message_length, in_channel=3, blocks=2, channels=64, diffusion_length=256, feature_size=64):
		super().__init__()

		stride_blocks = blocks

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(self.diffusion_length ** 0.5)
		self.feature_size = feature_size
		self.first_layers = nn.Sequential(
			ConvBNRelu(in_channel, channels),
			SENet(channels, channels * 2, blocks=1),
			nn.MaxPool2d((2, 2)),
			ConvBNRelu(channels * 2, channels),
			SENet(channels, channels, blocks=1),
			nn.MaxPool2d((2, 2)),
			ConvBNRelu(channels, channels),
		)
		self.keep_layers = nn.Sequential(
			SENet(channels, channels, blocks=1),
			SENet_decoder(channels, channels * 2, blocks=3, drop_rate2=1),
			ConvBNRelu(channels * 2, channels)
		)

		self.final_layer = ConvBNRelu(channels, 1)
		self.message_layer = nn.Linear(self.diffusion_length, message_length)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		# x = F.interpolate(x, size=(self.feature_size, self.feature_size), mode='bilinear', antialias=True)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = torch.nn.functional.adaptive_max_pool2d(x, 1)
		x = x.view(x.shape[0], -1)
		x = self.message_layer(x)
		return x
	
	def get_feature(self, x):
		x = self.first_layers(x)
		x = F.interpolate(x, size=(self.feature_size, self.feature_size), mode='bilinear', antialias=True)
		return x
		pass


if __name__ == "__main__":
	x = torch.randn(2, 3, 256, 256)
	model = SENet2D_ST_Decoder(30)
	print(model(x).size())
	pass