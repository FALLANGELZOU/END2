
from Tools.dataset.dataset import LDataset, read_img
import os
from torchvision import transforms

from Tools.steganography.utils.common import gen_random_msg
from Tools.util.TrainUtil import Args
from torch.utils.data import Dataset, DataLoader
class T1Dataset(LDataset):
    def __init__(self, args, valid=False) -> None:
        super().__init__(args)
        self.img_size = args.img_size
        self.msg_len = args.msg_len
        self.data_path = args.data_path
        self.data_repeat = args.data_repeat
        if valid:
            self.data_path = args.valid_data_path
        self.files = os.listdir(self.data_path)
        self.transforms = transforms.Compose([
            transforms.RandomCrop((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        pass


    def __getitem__(self, index):
        img = read_img(os.path.join(self.data_path, self.files[index % len(self.files)]))
        img = self.transforms(img)
        msg = gen_random_msg(self.msg_len)
        return {
            "img": img,
            "msg": msg
        }
        pass

    def __len__(self):
        return len(self.files) * self.data_repeat
    
    pass




