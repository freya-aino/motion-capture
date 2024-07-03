import random 
import numpy as np
import torch as T
import torch.utils.data as Tdata
import pytorch_lightning as pl

from .datasets import WFLWDataset



class BboxDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        dataset_class: Tdata.Dataset,
        dataset_kwargs: dict,
        image_key: str,
        image_shape: tuple[int, int], # width, height
        bbox_key: str,
        batch_size: int,
        train_val_test_split: tuple[float, float, float],
        num_train_workers: int,
        num_val_workers: int):
        
        super().__init__()
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.image_key = image_key
        self.image_shape = image_shape
        self.bbox_key = bbox_key
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_train_workers = num_train_workers
        self.num_val_workers = num_val_workers
        
        self.generator = T.Generator()
        
    def setup(self, stage=None):
        # Setup your dataset(s) for training, validation, test
        self.train_dataset, self.val_dataset, self.test_dataset = Tdata.random_split(
            dataset = self.dataset_class(**self.dataset_kwargs),
            lengths = self.train_val_test_split,
            generator = self.generator
        )
        
    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        x = T.stack([b[self.image_key] / 255 for b in batch])
        y = T.stack([(b[self.bbox_key] / T.tensor([self.output_full_image_shape] * 2)).flatten(-2) for b in batch])
        return x, y
    
    def train_dataloader(self):
        return Tdata.DataLoader(
            self.train_dataloader, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_train_workers,
            generator = self.generator
        )
    
    def val_dataloader(self):
        return Tdata.DataLoader(
            self.val_dataloader, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_val_workers,
            generator = self.generator
        )
    
    def test_dataloader(self):
        return Tdata.DataLoader(
            self.test_dataloader, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=False,
            generator = self.generator
        )
