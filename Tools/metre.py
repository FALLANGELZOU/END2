import torch
import skimage.metrics as metrics

def PSNR(x, y):
    """PSNR, db
        the input should be between 0 and 1
        Args:
            x (tensor): _description_
            y (tensor): _description_
        Returns:
            _type_: _description_
    """
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return torch.tensor(100).float()
    PIXEL_MAX = 1
    return (20 * torch.log10(PIXEL_MAX / torch.sqrt(mse)))
    pass

def SSIM(x, y):
   """SSIM
   Args:
       x (tensor): _description_
       y (tensor): _description_
   Returns:
       _type_: _description_
   """
   x = x.cpu().numpy()
   y = y.cpu().numpy()
   similarity = metrics.structural_similarity(x, y, data_range=1.0, multichannel=True)
   return similarity

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch=1):
        self.val = val
        self.sum += val * batch
        self.count += batch
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class MultiAvgMeter(object):
    def __init__(self):
        self.meters = {}
        pass
    def update(self, losses, batch):
        for key, value in losses.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(key)
                pass
            self.meters[key].update(value, batch)
            pass
        pass
    
    def reset(self):
        for key, value in self.meters.items():
            value.reset()
    pass