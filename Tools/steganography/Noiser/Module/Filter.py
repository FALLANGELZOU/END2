import torch
import torch.nn as nn
from kornia.filters import GaussianBlur2d
import numpy as np
from kornia.filters.gaussian import gaussian_blur2d
from Tools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager
from Tools.util.ImgUtil import img2tensor
from Tools.util.utils import vutils
from kornia.augmentation import RandomGaussianNoise

@NoiseLayerManager.register("GaussianFilter")
class GaussianFilter(NoiseLayer):
	"""
	blur image with random kernel size
	"""
	def __init__(self, kernel_size=[3, 8], sigma=[0.1, 2]):
		"""随机高斯滤波

		Args:
			kernel_size (list, optional): 高斯核大小范围. Defaults to [3, 8].
			sigma (list, optional): sigma范围越大越平滑越趋近于平均. Defaults to [0.1, 2].
		"""
		super().__init__()
		self.min_kernel = kernel_size[0]
		self.max_kernel = kernel_size[1]
		self.min_sigma = sigma[0]
		self.max_sigma = sigma[1]
		self.now_sigma = 1
		self.now_kernel = 3

	def noise(self, img, cover_img, repeat=False):
		if repeat is False:
			self.now_kernel = np.random.randint(self.min_kernel, self.max_kernel)//2*2+1
			self.now_sigma = np.random.uniform(self.min_sigma, self.max_sigma)
			# print(self.now_kernel, self.now_sigma)
		img = gaussian_blur2d(img, (self.now_kernel, self.now_kernel), (self.now_sigma, self.now_sigma))
		return img, cover_img
		pass

	def latest_noise(self, img, cover_img):
		return self.noise(img, cover_img, True)
	
@NoiseLayerManager.register("GaussianNoise")
class GaussianNoise(NoiseLayer):
	"""
	blur image with random kernel size
	"""
	def __init__(self, mean=0, std=0.01):
		"""随机高斯噪声
		"""
		super().__init__()
		self.model = RandomGaussianNoise(mean, std, p=1)


	def noise(self, img, cover_img, repeat=False):
		if repeat is False:
			return self.model(img), cover_img
		return self.model(img, params=self.model._params), cover_img
		pass

	def latest_noise(self, img, cover_img):
		return self.noise(img, cover_img, True)
	

if __name__ == "__main__":
	model = GaussianNoise()
	x = img2tensor("/home/light_sun/workspace/ImageST/test/test.png").expand(2, -1, -1, -1)
	x, y = model(x, x)
	vutils().save_image(torch.cat([x, y], dim=0), "./test/affine.png")
	pass