from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import torch
import numpy as np
import skfmm

from Tools.util.utils import vutils

def read_font(fontfile, size=256):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font


# def render(font, char, size=(128, 128), pad=5, mode='L') -> Image:
#     width, height = font.getsize(char)
#     max_size = max(width, height)
#     if width < height:
#         start_w = (height - width) // 2 + pad
#         start_h = pad
#     else:
#         start_w = pad
#         start_h = (width - height) // 2 + pad

#     img = Image.new(mode, (max_size+(pad*2), max_size+(pad*2)), 255)
#     draw = ImageDraw.Draw(img)
#     draw.text((start_w, start_h), char, font=font)
#     img = img.resize(size, 2)
#     return img


def render(font, char, size=(128, 128), pad=5, mode='L') -> Image:
    # 创建稍大一点的画布
    draw = ImageDraw.Draw(Image.new(mode, size))
    textbbox = draw.textbbox((0, 0), char, font=font)

    width = textbbox[2] - textbbox[0]
    height = textbbox[3] - textbbox[1]
    max_size = max(width, height)
    
    # 计算文本在图像中心的位置
    start_w = (max_size - width) // 2 + pad
    start_h = (max_size - height) // 2 + pad

    img = Image.new(mode, (max_size + (pad * 2), max_size + (pad * 2)), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font)
    img = img.resize(size, 2)
    return img

def calc_sdf(img):
    """计算位图的sdf，输入需要归一化到[-1，1]
        不可微分
    Args:
        img (_type_): 大小为(h, w)
    """
    device = img.device
    img = img.cpu().numpy()
    phi = np.where(img >= 0, 1, -1)
    sdf = skfmm.distance(phi, dx = 1)
    return torch.from_numpy(sdf).to(device)
    pass

def render_sdf(sdf, r = 1.5):
    """渲染由calc_sdf计算的有向距离场
    返回图片在[-1,1]之间

    Args:
        sdf (_type_): [h, w]
    """
    normalized_sdf = sdf / r
    clamped_sdf = torch.clamp(normalized_sdf, min=-1, max=1)
    return _antialias_kernel(clamped_sdf)  # multiply by color here
    pass

def _antialias_kernel(r):
    r = -r
    output = (0.5 + 0.25 * (torch.pow(r, 3) - 3 * r))
    #   output = -0.5*r + 0.5
    return output


if __name__ == "__main__":
    from torchvision import transforms
    ts = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    font_path = "/home/luyx/env_sn/workspace/ri/data/ttf/kai.ttf"
    font = read_font(font_path)
    img = render(font, "第", (256, 256))
    img = ts(img)
    print(torch.max(img), torch.min(img))
    sdf = calc_sdf(img)
    print(torch.max(sdf), torch.min(sdf))
    raster_img = render_sdf(sdf).unsqueeze(0)
    vutils().save_image(raster_img, "./test.png")
    sdf = sdf.unsqueeze(0)
    vutils().save_image(sdf, "./sdf.png", normalize=True)
    # print(raster_img)
    pass

