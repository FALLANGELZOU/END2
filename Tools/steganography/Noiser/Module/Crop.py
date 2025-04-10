from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from Tools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height, image_width = image_shape[2], image_shape[3]
	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width

@NoiseLayerManager.register("Crop")
class Crop(NoiseLayer):
	"""
	crop image randomly
	"""
	def __init__(self, 
			  ratio=[0.5, 1.], 
			  keep_size=True, 
			  proportional=True,
			  sync_cover_img=True,
			  clip_out=True
			  ):
		"""随机裁剪

		Args:
			ratio ([float, float], optional): 裁剪块与原图的比例. Defaults to [0.5, 1.].
			keep_size (bool, optional): 是否保持原图尺寸. Defaults to True.
			proportional (bool, optional): 是否保持宽高比例不变. Defaults to True.
			sync_cover_img (bool, optional): cover_img是否也应用. Defaults to True.
		"""
		super().__init__()
		self.min_ratio = ratio[0]
		self.max_ratio = ratio[1]
		self.proportional = proportional
		self.keep_size = keep_size
		self.sync_cover_img = sync_cover_img
		self.clip_out = clip_out
		self.h_start = 0
		self.h_end = 0
		self.w_start = 0
		self.w_end = 0
		pass
	
	def noise(self, img, cover_img, repeat=False):
		b, c, h, w = img.size()
		if repeat is False:
			if self.proportional:
				height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio) 
				width_ratio = height_ratio
			else:
				height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)
				width_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)
		
			self.h_start, self.h_end, self.w_start, self.w_end = get_random_rectangle_inside(img.shape, height_ratio, width_ratio)
		
		img = img[:, :, self.h_start:self.h_end, self.w_start:self.w_end]
		if self.sync_cover_img:
			cover_img = cover_img[:, :, self.h_start:self.h_end, self.w_start:self.w_end]
		
		if self.keep_size:
			img = F.interpolate(img, (h, w), mode='bicubic', antialias=True)
			if self.sync_cover_img:
				cover_img = F.interpolate(cover_img, (h, w), mode='bicubic', antialias=True)
		if self.clip_out:
			img = torch.clamp(img, 0, 1)
			cover_img = torch.clamp(cover_img, 0, 1)
		return img, cover_img		
		pass

	def latest_noise(self, img, cover_img):
		return self.noise(img, cover_img, True)

@NoiseLayerManager.register("Cropout")
class Cropout(NoiseLayer):
	def __init__(self, 
		ratio=[0.05, 0.1], 
		proportional=False,
		clip_out=True
		):
		super().__init__()
		self.min_ratio = ratio[0]
		self.max_ratio = ratio[1]
		self.proportional = proportional
		self.clip_out = clip_out
		self.h_start = 0
		self.h_end = 0
		self.w_start = 0
		self.w_end = 0

	def noise(self, img, cover_img, repeat=False):
		b, c, h, w = img.size()
		if repeat is False:
			if self.proportional:
				height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio) 
				width_ratio = height_ratio
			else:
				height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)
				width_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)
		
			self.h_start, self.h_end, self.w_start, self.w_end = get_random_rectangle_inside(img.shape, height_ratio, width_ratio)
		output = img.clone()
		output[:, :, self.h_start:self.h_end, self.w_start:self.w_end] = cover_img[:, :, self.h_start:self.h_end, self.w_start:self.w_end]

		if self.clip_out:
			output = torch.clamp(output, 0, 1)
			cover_img = torch.clamp(cover_img, 0, 1)
		return output, cover_img		
		pass

	def latest_noise(self, img, cover_img):
		return self.noise(img, cover_img, True)
	pass

@NoiseLayerManager.register("Dropout")
class Dropout(NoiseLayer):
	def __init__(self, prob=[0.1, 0.3]):
		super().__init__()
		self.prob = prob
		self.c_porb = 0.1
		pass

	def noise(self, img, cover_img, repeat=False):
		if repeat is False:
			self.c_porb = (np.random.rand() * (self.prob[1]-self.prob[0]) + self.prob[0])
			pass
		rdn = torch.rand(img.shape).to(img.device)
		output = torch.where(rdn > self.c_porb, img, cover_img)
		return output, cover_img
		pass

	def latest_noise(self, img, cover_img):
		return self.noise(img, cover_img, True)
	pass