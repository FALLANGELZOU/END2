

from T1Data import T1Dataset
from FastTools.light.LightModel import LModel, LTrainer
from FastTools.util.TrainUtil import Args
from model import END2
from lightning.pytorch.callbacks import ModelCheckpoint




class WMTrainer(LTrainer):
    def __init__(self, name, root_dir, gpus, cfg, batch_size, num_workers, max_epochs, ckpt_path=None, use_split_dataset: bool = False, train_dataset_ratio: int = 1, seed=None, precision=None, find_unused_params=True):
        super().__init__(name, root_dir, gpus, cfg, batch_size, num_workers, max_epochs, ckpt_path, use_split_dataset, train_dataset_ratio, seed, precision, find_unused_params)
        
        pass
    def build_dataset(self, cfg):
        return T1Dataset(cfg)
    

    def build_val_dataset(self, cfg):
        return T1Dataset(cfg, True)
    
    def build_model(self, cfg) -> LModel:
        return END2(cfg)

    def build_checkpoint_callback(self):
        
        callback = ModelCheckpoint(
            save_top_k= 5, # 默认保存最好的5个， 需要保存条件
            monitor='valid_loss', # 默认使用总loss保存最好的结果
            filename="ckpt-{epoch:02d}-{valid_loss:.4f}",
            save_last=True,
            every_n_epochs=20,
            # save_on_train_epoch_end=True,
            save_weights_only=False
        )
        
        return callback
    


    pass
    
def train():
    cfg = Args().load_from_yaml("./cfg.yaml")
    trainer = WMTrainer(
        name=cfg.name,
        root_dir=cfg.root_dir,
        gpus=cfg.gpus,
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_epochs=cfg.max_epochs,
        find_unused_params=cfg.find_unused_parameters,
        precision=None,
        ckpt_path=cfg.ckpt_path
    )
    trainer.run()
    pass