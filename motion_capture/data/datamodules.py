import torch as T
import torch.nn as nn
import pytorch_lightning as pl

import torch.utils.data as Tdata
from motion_capture.data.preprocessing import ImageAugmentations


class DataModule(pl.LightningDataModule):
    
    def __init__(self, dataset, y_key, batch_size, image_augmentation, train_val_split, num_workers):
        super().__init__()
        self.dataset = dataset
        self.y_key = y_key
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.image_augmentation = ImageAugmentations.get(image_augmentation, ImageAugmentations["NONE"])
        
    def collate_fn(self, batch):
        x = T.stack([self.image_augmentation(b[0]) for b in batch])
        y = T.stack([b[1][self.y_key] for b in batch])
        return x, y
        
    def setup(self, stage):
        splits = Tdata.random_split(
            dataset = self.dataset, 
            lengths = self.train_val_split,
        )
        self.train_dataset = splits[0]
        self.val_dataset = splits[1]
        
    def train_dataloader(self):
        return Tdata.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers != 1 else False,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return Tdata.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers != 1 else False,
            pin_memory=True
        )