import torch as T
import torch.nn as nn
import pytorch_lightning as pl

import torch.utils.data as Tdata
from motion_capture.data.preprocessing import ImageAugmentations
from motion_capture.model.models import VQVAE


class DataModule(pl.LightningDataModule):
    
    def __init__(self, dataset, batch_size, image_augmentation, train_val_test_split, num_workers):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.image_augmentation = ImageAugmentations[image_augmentation]
        
    def collate_fn(self, batch):
        x = T.stack([self.image_augmentation(b[0]) for b in batch])
        y = {}
        for b in batch:
            for k, v in b[1].items():
                if k not in y:
                    y[k] = []
                y[k].append(v)
        for k in y:
            y[k] = T.stack(y[k])
        return x, y
        
    def setup(self):
        splits = Tdata.random_split(
            dataset = self.dataset, 
            lengths = self.train_val_test_split,
        )
        self.train_dataset = splits[0]
        self.val_dataset = splits[1]
        self.test_dataset = splits[2]
        
    def train_dataloader(self):
        return Tdata.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_train_workers,
            persistent_workers=True if self.num_train_workers > 0 else False,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return Tdata.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_val_workers,
            persistent_workers=True if self.num_val_workers > 0 else False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return Tdata.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=False,
        )

