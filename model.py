import torch
import torch.nn as nn

from Tools.light.LightModel import LModel
from Tools.metre import PSNR
from Tools.module.SENet import SENet, SENet_decoder
from Tools.module.common import MLP, ConvBNRelu, ExpandNet, Flatten
from Tools.steganography.Noiser.Noiser import Noiser, get_default_Noiser
from Tools.steganography.utils.common import msg_acc
import torch.nn.functional as F

from Tools.util.TrainUtil import Args


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
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)
		intermediate3 = self.after_concat_layer(concat1)
		concat2 = torch.cat([intermediate3, image], dim=1)
		output = self.final_layer(concat2)
		return output

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
			SENet_decoder(channels, channels * 2, blocks=2, drop_rate2=1),
			ConvBNRelu(channels * 2, channels)
		)

		self.final_layer = ConvBNRelu(channels, 1)

		self.message_layer = nn.Linear(self.diffusion_length, message_length)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		# x = F.interpolate(x, size=(self.feature_size, self.feature_size), mode='bilinear', antialias=True)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		z = x.view(x.shape[0], -1)
		x = self.message_layer(z)
		return x, z
	
	def get_feature(self, x):
		x = self.first_layers(x)
		x = F.interpolate(x, size=(self.feature_size, self.feature_size), mode='bilinear', antialias=True)
		return x
		pass


class END2(LModel):
	"""使用BarlowTwins的损失函数约束

	Args:
		LModel (_type_): _description_
	"""
	def __init__(self, args):
		super().__init__()
		self.msg_z_len = 256
		self.encoder = SENet2D_ST_Encoder(args.msg_len, msg_upsample=3, diffusion_length=self.msg_z_len)
		self.decoder_t = SENet2D_ST_Decoder(args.msg_len, diffusion_length=self.msg_z_len)
		self.decoder_s = SENet2D_ST_Decoder(args.msg_len, diffusion_length=self.msg_z_len)
		self.w_student_loss = args.w_student_loss
		self.w_msg_loss = args.w_msg_loss
		self.w_quantity_loss = args.w_quantity_loss
		self.swap_epoch = args.swap_epoch
		self.decay = 0.999
		self.lr = args.lr
		# self.noiser = cycleGanNoiser("./vangogh.pth", args.img_size)
		# self.noiser = JpegTest(50)
		self.noiser = Noiser([
			("Identity", None),
			("Rotate", None),
			("Crop", None),
			("Translate", None),
			("Scale", None),
			("Shear", None),
			("Dropout", None),
			("Cropout", None),

			("Color", None),
			("KorniaJpeg", None),
			("GaussianFilter", None),
			("GaussianNoise", None)
		])
		self.projector = nn.Linear(self.msg_z_len, self.msg_z_len, False)
		pass


	def train_step(self, batch, optimizers, batch_idx):
		self.train()
		optimizer: torch.optim.Adam = optimizers
		optimizer.zero_grad()

		img = batch['img']
		msg = batch['msg']

		wm_img = self.encoder(img, msg)

		# 模拟不可微分噪声
  
		# cycleGan模拟
		# noised_wm_img, img = self.cyclegan_noiser(wm_img, img)
  
		# 正常的模拟
		noised_wm_img, img = self.normal_noiser(wm_img, img)	

		# teacher model
		t_predict_msg, t_z = self.decoder_t(wm_img)

		# student model
		s_predict_msg, s_z = self.decoder_s(noised_wm_img)


		# 视觉质量
		quantity_loss = F.mse_loss(wm_img, img)

		# 解码准确率
		msg_loss = F.mse_loss(s_predict_msg, msg) + F.mse_loss(t_predict_msg, msg)

		# student靠近teacher
		student_loss = self.BT_loss(t_z, s_z)

		loss = quantity_loss * self.w_quantity_loss + msg_loss * self.w_msg_loss + student_loss * self.w_student_loss
		self.loss_backward(loss)
		optimizer.step()

		# EMA更新
		s_params = self.decoder_s.state_dict()
		for name, param in self.decoder_t.state_dict().items():
			param = self.decay * param + (1-self.decay) * s_params[name]
			pass

		# 交换学习
		if batch_idx % self.swap_epoch == 0:
			self.decoder_s, self.decoder_t = self.decoder_t, self.decoder_s

		# 统计数据
		psnr = PSNR(wm_img, img)
		t_acc = msg_acc(t_predict_msg, msg)
		s_acc = msg_acc(s_predict_msg, msg)

		self.log("loss", loss.cpu().item(), prog_bar=True)
		self.log("psnr", psnr.cpu().item(), prog_bar=True)
		self.log("acc", s_acc.cpu().item(), prog_bar=True)
		self.log("metric/t_acc", t_acc.cpu().item())
		self.log("metric/quantity_loss", quantity_loss.cpu().item())
		self.log("metric/msg_loss", msg_loss.cpu().item())
		self.log("metric/student_loss", student_loss.cpu().item())


		if batch_idx == 0:
			self.log_img("img", img.detach().cpu())
			self.log_img("wm_img", wm_img.detach().cpu())
			self.log_img("noised_img", noised_wm_img.detach().cpu())
		pass


	def valid_step(self, batch, batch_idx):
		
		self.eval()
		img = batch['img']
		msg = batch['msg']

		wm_img = self.encoder(img, msg)

		# 模拟不可微分噪声
		# cycleGan模拟
		# noised_wm_img, img = self.cyclegan_noiser(wm_img, img)
		# 正常的模拟
		noised_wm_img, img = self.normal_noiser(wm_img, img)	
		
		# teacher model
		t_predict_msg, t_z = self.decoder_t(wm_img)

		# student model
		s_predict_msg, s_z = self.decoder_s(noised_wm_img)


		# 视觉质量
		quantity_loss = F.mse_loss(wm_img, img)

		# 解码准确率
		msg_loss = F.mse_loss(s_predict_msg, msg) + F.mse_loss(t_predict_msg, msg)

		# student靠近teacher
		student_loss = self.BT_loss(t_z, s_z)

		loss = quantity_loss * self.w_quantity_loss + msg_loss * self.w_msg_loss + student_loss * self.w_student_loss

		# 统计数据
		psnr = PSNR(wm_img, img)
		t_acc = msg_acc(t_predict_msg, msg)
		s_acc = msg_acc(s_predict_msg, msg)

		self.log("valid_loss", loss.cpu().item(), sync_dist=True)
		self.log("valid/psnr", psnr.cpu().item(), sync_dist=True)
		self.log("valid/t_acc", t_acc.cpu().item(), sync_dist=True)
		self.log("valid/s_acc", s_acc.cpu().item(), sync_dist=True)
		self.log("valid/quantity_loss", quantity_loss.cpu().item(), sync_dist=True)
		self.log("valid/msg_loss", msg_loss.cpu().item(), sync_dist=True)
		self.log("valid/student_loss", student_loss.cpu().item(), sync_dist=True)
		pass
	def build_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)
	
	def H_loss(self, t, s, tps=.5, tpt=.5, C=0):
		t = t.detach()
		s = F.softmax(s / tps, dim=1)
		t = F.softmax((t-C) / tpt, dim=1)
		return - (t * torch.log(s)).sum(dim=1).mean()
		pass


	def BT_loss(self, t, s):
		t = t.detach()
		t = self.projector(t)
		s = self.projector(s)
		t = torch.norm(t, p=2, dim=-1)
		s = torch.norm(s, p=2, dim=-1)
		
		return F.mse_loss(s, t) ** 2
	
	def cyclegan_noiser(self, wm_img, img):
		noised_wm_img = self.noiser(wm_img.detach()*2-1)
		noised_wm_img = noised_wm_img/2+0.5
		return noised_wm_img.detach(), img
		pass

	def normal_noiser(self, wm_img, img):
		# 阻止梯度传播
		noised_wm_img, _ = self.noiser(wm_img.detach(), img)
		return noised_wm_img.detach(), img
		pass

	pass




if __name__ == "__main__":
	args = Args().load_from_yaml("./cfg.yaml")
	model = SENet2D_ST_Encoder(30)
	decoder = SENet2D_ST_Decoder(30)
	x = torch.randn(10, 3, 128, 128)
	y = torch.randn(10, 30)
	x = model(x, y)
	m = decoder(x)

	print(x.size())
	pass