import builtins
from datetime import datetime
import math
from multiprocessing import freeze_support
import multiprocessing
import os
from types import FunctionType
from typing import Any
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm

from Tools.logger import Logger
from Tools.CustomStruct import CustomDict
from abc import ABC, abstractmethod

from Tools.util.utils import cal_params, root_folder_name, save_programme

class ModuleWrap(nn.Module):
    def __init__(self, model):
        super(ModuleWrap, self).__init__()
        self.module = model
    pass
class TrainerArgs(CustomDict):
    def __init__(self,
                 cfg=None,
                 cpu=False,
                 gpu=[0],
                 batch_size:int=128,
                 lr:float=0.001,
                 workers:int=4,
                 start_epoch:int=0,
                 max_epochs:int=250,
                 weight:str=None,
                 save_iters:int=10,
                 seed=None,
                 dist_url="tcp://127.0.0.1:25565",
                 ):
        self['name'] = "DefaultModelName"
        self["cpu"] = cpu
        self["gpu"] = gpu
        self["batch_size"] = batch_size
        self["lr"] = lr
        self["workers"] = workers
        self["start_epoch"] = start_epoch
        self["max_epochs"] = max_epochs
        self["weight"] = weight
        self["save_iters"] = save_iters
        self['seed'] = seed
        self["dist_url"] = dist_url
        if cfg is not None:
            for items in cfg.items():
                self[items[0]] = items[1]
                pass
            pass
        pass
    pass

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
class Trainer(ABC):
    """
    some args are needed as follow:
    
    name: 'default'
    
    start_epoch: 1 下标从1开始
    
    max_epoch: 250 闭区间
    
    batch_size: 64
    
    workers: 24
    
    dist_url: null
    
    output_path: ""

    world_size: 2
    
    gpu: [0,1]    
    
    """
    def __init__(self,
                 args: TrainerArgs
                 ) -> None:
        self.args = args
        self.logger = None
    def run(self):
        self._init()        
        pass
    
    def _seed(self):
        seed = self.args['seed']
        if seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            cudnn.deterministic = True
            self.logger.warn('You have chosen to seed training. '
                            'This will turn on the CUDNN deterministic setting, '
                            'which can slow down your training considerably! '
                            'You may see unexpected behavior when restarting '
                            'from checkpoints.')
        pass
    
    def _init(self):
        
        output_path = self.args["output_path"]
        save_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "__" + self.args['name']
        output_path = os.path.join(output_path, save_name)
        self.args['output_path'] = output_path
        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
        self.logger = Logger(name = self.args["name"] , file_path=self.args['output_path'])
        self.logger.info("output_path: {}".format(self.args['output_path']))
        if os.path.exists(output_path):
            programme_path = os.path.join(output_path, root_folder_name())
            self.logger.info("save programme path: {}".format(programme_path))
            save_programme(programme_path)
        
        self._seed()
        
        gpu = self.args["gpu"]
        if self.args['cpu'] == True:
            self.logger.info("Use CPU!")
            self.logger.error("Not support CPU training!")
            pass
        
        elif gpu is not None:
            if len(gpu) == 1:
                self.args['ddp'] = False
                self.logger.info("Use GPU:{} for training".format(gpu[0]))
                self.single_worker(gpu[0], self.args)
                pass
            else:
                self.logger.info("Use DDP for training")
                self.args['ddp'] = True
                gpus = ",".join([str(i) for i in gpu])
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus
                self.logger.info("Use GPU:{} for training".format(gpus))
                world_size = self.args['world_size']
                self.logger = None # 无法被序列化
                mp.spawn(self.main_worker, nprocs=world_size, args=(self.args, ))
                pass
            pass 
        else:
            self.logger.error("multiprocessing-distributed is not implemented yet")
        pass
    def main_worker(self, rank, args: Any):
        # 每个进程的内容
        logger = None
        def print_pass(*args):
            pass
        if rank != 0:
            # 屏蔽其他进程的消息
            builtins.print = print_pass
            logger = Logger(name = args["name"] ,file_path=None, main_worker=False)
        else:
            # 显示本进程的消息
            logger = Logger(name = args["name"] ,file_path=args['output_path'])
        self.logger = logger
        # init params
        args['rank'] = rank
        url = args["dist_url"] if args["dist_url"] is not None else "tcp://127.0.0.1:25565"
        args["start_epoch"] = args["start_epoch"] if args["start_epoch"] is not None else 1
        args['logger'] = logger
        
        # init ddp
        if args['ddp']:
            logger.info("Init process group:{}".format(url))
            dist.init_process_group(backend="nccl", init_method=url, world_size=args['world_size'], rank=rank)
        
        # init other setting
        self.init(args)
        
        # torch.distributed.barrier(device_ids=[rank])
        logger.info("create model")
        model = self.build_model(args)

        # init optimizer
        logger.info("build optimizer")
        optimizer = self.build_optim(model, args)

        # load weight
        if args["ckpt_path"] is not None:
            logger.info("load checkpoint:{}".format(args["ckpt_path"]))
            loc = "cuda:{}".format(rank)
            checkpoint = torch.load(args["ckpt_path"], map_location=torch.device(loc))
            self.load_checkpoint(model, optimizer, checkpoint, rank, args)
            # args["start_epoch"] = checkpoint["epoch"]
            # dist.barrier(device_ids=[rank])
            pass
                
        # barrier
        if self.args['ddp']:
            dist.barrier(device_ids=[rank])
        logger.info("model created")
        
        # init model
        if args['ddp']:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(rank)
        model.cuda(rank)
        if args['ddp']:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        else:
            model = ModuleWrap(model).cuda(rank)
        if args['print_struct']:
            logger.info(model)
        cudnn.benchmark = True

        # init dataset
        logger.info("build dataset")
        dataset = self.build_dataset(args)
        if args['ddp']:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args['batch_size'], shuffle=(sampler is None),
            num_workers=args['workers'], pin_memory=True, sampler=sampler, drop_last=True)
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args['batch_size'], shuffle=True,
                num_workers=args['workers'], drop_last=True
            )
        
        # init valid dataset
        valid_data_loader = None
        if args['valid_dataset'] == True:
            logger.info("build valid dataset")
            valid_dataset = self.build_valid_dataset(args)
            if args['ddp']:
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
                valid_data_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=args['valid_batch_size'], shuffle=(valid_sampler is None),
                    num_workers=args['valid_workers'], pin_memory=True, sampler=valid_sampler, drop_last=True)
            else:
                valid_data_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=args['valid_batch_size'], shuffle=True,
                    num_workers=args['valid_workers'], drop_last=True
                )
            pass
                
        # dispatch
        if args['infer']:
            self.logger.info("##############")
            self.logger.info("#start infer!#")
            self.logger.info("##############")
            self.infer(model.module, data_loader, valid_data_loader, rank, args)
            pass
        else:
            logger.info("trainable params:{:.2f}M".format(cal_params(model)))
            
            args['epoch'] = args['start_epoch']
            self.before_train(model.module, optimizer, args)
            self.logger.info("#################")
            self.logger.info("#start training!#")
            self.logger.info("#################")
            for epoch in range(args['start_epoch'], args['max_epoch'] + 1):
                if args['ddp']:
                    sampler.set_epoch(epoch)
                self.update(model.module, optimizer, data_loader, valid_data_loader, epoch, rank, args)  
                if (args['epoch'] % args['save_iter'] == 0 or args['epoch'] == args["max_epoch"]) and rank == 0:
                    self.save_checkpoint(model.module, optimizer, args)
                    pass
                args['epoch'] = epoch + 1
                pass
            self.after_train(model.module, optimizer, args)           
            pass

        pass
    
    def single_worker(self, rank, args: Any):
        # 每个进程的内容
        logger = None
        def print_pass(*args):
            pass
        logger = Logger(name = args["name"] ,file_path=args['output_path'])
        self.logger = logger
        # init params
        args['rank'] = rank
        url = args["dist_url"] if args["dist_url"] is not None else "tcp://127.0.0.1:25565"
        args["start_epoch"] = args["start_epoch"] if args["start_epoch"] is not None else 1
        args['logger'] = logger
        
        # init other setting
        self.init(args)
        
        # torch.distributed.barrier(device_ids=[rank])
        logger.info("create model")
        model = self.build_model(args)

        # init optimizer
        logger.info("build optimizer")
        optimizer = self.build_optim(model, args)

        # load weight
        if args["ckpt_path"] is not None:
            logger.info("load checkpoint:{}".format(args["ckpt_path"]))
            loc = "cuda:{}".format(rank)
            checkpoint = torch.load(args["ckpt_path"], map_location=torch.device(loc))
            self.load_checkpoint(model, optimizer, checkpoint, rank, args)
            # args["start_epoch"] = checkpoint["epoch"]
            # dist.barrier(device_ids=[rank])
            pass
                
        # barrier
        logger.info("model created")
        
        # init model
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = ModuleWrap(model).cuda(rank)
        if args['print_struct']:
            logger.info(model)
        cudnn.benchmark = True

        # init dataset
        logger.info("build dataset")
        dataset = self.build_dataset(args)
        data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args['batch_size'], shuffle=True,
                num_workers=args['workers'], drop_last=True
            )
        
        # init valid dataset
        valid_data_loader = None
        if args['valid_dataset'] == True:
            logger.info("build valid dataset")
            valid_dataset = self.build_valid_dataset(args)
            valid_data_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=args['valid_batch_size'], shuffle=True,
                    num_workers=args['valid_workers'], drop_last=True
                )
            pass
                
        # dispatch
        if args['infer']:
            self.logger.info("##############")
            self.logger.info("#start infer!#")
            self.logger.info("##############")
            self.infer(model.module, data_loader, valid_data_loader, rank, args)
            pass
        else:
            logger.info("trainable params:{:.2f}M".format(cal_params(model)))
            
            args['epoch'] = args['start_epoch']
            self.before_train(model.module, optimizer, args)
            self.logger.info("#################")
            self.logger.info("#start training!#")
            self.logger.info("#################")
            for epoch in range(args['start_epoch'], args['max_epoch'] + 1):
                self.update(model.module, optimizer, data_loader, valid_data_loader, epoch, rank, args)  
                if (args['epoch'] % args['save_iter'] == 0 or args['epoch'] == args["max_epoch"]):
                    self.save_checkpoint(model.module, optimizer, args)
                    pass
                args['epoch'] = epoch + 1
                pass
            self.after_train(model.module, optimizer, args)           
            pass

        pass
       
    @abstractmethod
    def build_model(self, args) -> nn.Module:
        pass
    
    @abstractmethod
    def build_optim(self, model, args):
        pass
    
    @abstractmethod
    def build_dataset(self, args):
        pass
    
    @abstractmethod
    def build_valid_dataset(self, args):
        pass

    @abstractmethod
    def load_checkpoint(self, model, optimizer, checkpoint, rank, args):
        pass
    
    @abstractmethod
    def update(self, model, optimizer, data_loader, valid_data_loader, epoch, rank, args):
        pass
    
    
    @abstractmethod
    def save_checkpoint(self, model, optimizer, args):
        pass
    
    def init(self, args):
        """
        before create model
        """
        pass
    
    def infer(self, model, data_loader, valid_loader, rank, args):
        """
        推理逻辑
        """
        pass

    def before_train(self, model, optimizer, args):
        
        pass
    
    def after_train(self, model, optimizer, args):
        
        pass
    pass









if __name__ == "__main__":

    pass