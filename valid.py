import torch
from tqdm import tqdm
from model import END2
from FastTools.metre import PSNR
from FastTools.metre_util.ssim import SSIM
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.ImgUtil import clip_psnr
from FastTools.util.TrainUtil import Args
from T1Data import T1Dataset


device = "cuda:0"
ckpt_path = "./END2_ckpt.ckpt"
args = Args().load_from_yaml("./cfg.yaml")
model = END2.load_from_checkpoint(
    ckpt_path,
    args=args
).to(device).eval()

dataset = T1Dataset(args, valid=True)
dataloader = torch.utils.data.DataLoader(dataset, 12, drop_last=True)

total_psnr = 0
total_ssim = 0
total_acc = 0
n = 0
with torch.no_grad():
    for batch in tqdm(dataloader):
        img = batch['img'].to(device)
        msg = batch['msg'].to(device)
        wm_img = model.encoder(img, msg)
        wm_img = clip_psnr(wm_img, img, 45, over_clip=False)

        noised_wm_img, _ = model.normal_noiser(wm_img, img)
        t_predict_msg, _ = model.decoder_t(noised_wm_img)
        s_predict_msg, _ = model.decoder_s(noised_wm_img)
        # predict_msg = t_predict_msg
        predict_msg = (t_predict_msg + s_predict_msg) / 2 # 这里三种情况差不多
        # predict_msg = s_predict_msg
        psnr = PSNR(wm_img, img)
        ssim = SSIM(wm_img, img)
        acc = msg_acc(predict_msg, msg)
        total_psnr += psnr.item()
        total_acc += acc.item()
        total_ssim += ssim.item()
        n += 1

        pass

print("total_psnr: {}".format(total_psnr / n))
print("total_ssim: {}".format(total_ssim / n))
print("total_acc: {}".format(total_acc / n))