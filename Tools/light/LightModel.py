from abc import abstractmethod
from datetime import datetime
import os
from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
"""
使用Lightning框架的
"""
class LModel(L.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # 用于保存记录
        self.history = { }
        self.automatic_optimization = False
        pass

    @abstractmethod
    def train_step(self, batch, optimizers, batch_idx):
        '''
        训练step
        '''        
        pass
    
    def valid_step(self, batch, batch_idx):
        '''
        验证step
        '''
        pass
    @abstractmethod
    def build_optimizers(self):
        '''
        构建优化器，可以返回一个或多个优化器
        '''
        pass    
    

    
    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        self.train_step(batch, optimizers, batch_idx)
        pass
    
    def configure_optimizers(self):
        return self.build_optimizers()
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.valid_step(batch, batch_idx)
    
    

    
    def loss_backward(self, loss, retain_graph=False):
        """调用backward

        Args:
            loss (_type_): _description_
        """
        self.manual_backward(loss, retain_graph=retain_graph)
        pass

    def log_img(self, name, img, n_epoch=10):  
        if self.current_epoch % n_epoch == 0:
            logger = self.logger.experiment
            logger.add_images("img/{}".format(name), img.cpu(), self.current_epoch)
            pass
        pass
    pass


class LTrainer():
    def __init__(self, 
                 name, # 任务名称
                 root_dir, # 存储路径
                 gpus,
                 cfg, # 配置文件 
                 batch_size,
                 num_workers, 
                 max_epochs,
                 ckpt_path = None, # 权重文件
                 use_split_dataset: bool=False, # 是否将数据集划分为训练集和验证集
                 train_dataset_ratio: int = 1.0, # 如果启用验证集，需要规定训练集占比
                 seed = None, # 使用的种子
                 precision = None,
                 find_unused_params = True
                 ):
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.precision = precision  # 精度
        # 如果路径不存在，则创建路径
        self.root_path = os.path.join(
            root_dir,
            name) #datetime.now().strftime("%Y_%m_%d_%H:%M:%S") + "__" + name)
        
        self.val_every_n_epoch = cfg.val_epoch
        if self.val_every_n_epoch == None:
            self.val_every_n_epoch = 1
        
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        
        

        dataset = self.build_dataset(cfg)
        if use_split_dataset:
            total_size = len(dataset)
            train_size = int(total_size * train_dataset_ratio)
            valid_size = total_size - train_size
            
            dataset, val_dataset = data.random_split(dataset, [train_size, valid_size], seed)
            self.val_dataset = torch.utils.data.DataLoader(
                val_dataset,
                min(32, valid_size),
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.build_collate_fn(cfg)
                )
            pass
        else:
            val_dataset = self.build_val_dataset(cfg)
            if val_dataset is not None:
                val_batch_size = min(32, len(val_dataset))
                self.val_dataset = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=self.build_collate_fn(cfg)
                )
                pass
            else:
                self.val_dataset = None
            
        
        self.dataset = torch.utils.data.DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=self.build_collate_fn(cfg)
                )
                    
        self.model = self.build_model(cfg)
        
        callbacks = []
        checkpoint_callback = self.build_checkpoint_callback()
        if checkpoint_callback is not None:
            callbacks.append(checkpoint_callback)
        earlystopping_callback = self.build_earlystopping_callback()
        if earlystopping_callback is not None:
            callbacks.append(earlystopping_callback)
        if seed is not None:
            L.seed_everything(seed, workers=True)
        if find_unused_params == True:
            strategy="ddp_find_unused_parameters_True"
        else:
            strategy="ddp_find_unused_parameters_False"
            
        self.trainer = L.Trainer(
            accelerator="gpu", 
            devices=gpus,
            default_root_dir=self.root_path,
            enable_checkpointing=True,
            max_epochs=max_epochs,
            callbacks=callbacks,
            strategy=strategy, # ddp_find_unused_parameters_true 可以避免未使用的参数
            precision= self.precision,
            check_val_every_n_epoch=self.val_every_n_epoch,
            log_every_n_steps=10
        )
        pass
    
    
    @abstractmethod
    def build_dataset(self, cfg):
        '''
        构建数据集
        '''
        pass
    
    @abstractmethod
    def build_model(self, cfg) -> LModel:
        '''
        构建model
        '''
        pass
    
    def build_val_dataset(self, cfg):
        return None
        pass
    
    
    def build_checkpoint_callback(self):
        '''
        用来自定义保存逻辑
        '''
        callback = ModelCheckpoint(
            save_top_k= 5, # 默认保存最好的5个， 需要保存条件
            monitor='loss', # 默认使用总loss保存最好的结果
            filename="ckpt-{epoch:02d}-{loss:.4f}",
            save_last=True,
            save_on_train_epoch_end=True,
            save_weights_only=False
        )
        
        return callback
        pass
    
    def build_earlystopping_callback(self):
        # callback = EarlyStopping(
        #     monitor="val_accuracy", 
        #     min_delta=0.00, 
        #     patience=5, 
        #     verbose=False, 
        #     mode="min"
        #     )
        return None
        pass

    def build_collate_fn(self, cfg):
        return None

    
    def run(self):
        self.trainer.fit(
            self.model, 
            ckpt_path=self.ckpt_path, 
            train_dataloaders=self.dataset, 
            val_dataloaders=self.val_dataset
            
        )
        pass
    pass



if __name__ == "__main__":

    pass