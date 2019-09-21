import os
import torch as torch
import numpy as np
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir,size, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id)
        dali_device = "gpu"
        self.size = size
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True,initial_fill = 3000)#initial_fill: cache pool
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR,
                                  resize_x = float(self.size),
                                  resize_y = float(self.size))
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            crop = (227, 227),
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        #images = self.res(images)
        images = self.resize(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]

""" Module for the data loading pipeline for the model to train """


def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize

    if new_size is not None:
        image_transform = Compose([
            Resize(new_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    else:
        image_transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform


# def get_data_loader(dataset, batch_size, num_workers):
#     """
#     generate the data_loader from the given dataset
#     :param dataset: dataset for training (Should be a PyTorch dataset)
#                     Make sure every item is an Image
#     :param batch_size: batch size of the data
#     :param num_workers: num of parallel readers
#     :return: dl => dataloader for the dataset
#     """
#     from torch.utils.data import DataLoader

#     dl = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers
#     )

#     return dl
import os

class Data_loader:
    def __init__(self,dataset, size,batch_size, num_workers):
        self.batchsize = batch_size
        self.pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_workers,device_id=0,
                                    data_dir=dataset ,size = size
                                    )
        self.pip_train.build()
        self.dataloader = DALIClassificationIterator(self.pip_train, size=self.pip_train.epoch_size("Reader"))
    def __iter__(self):
            return iter(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        #return len(self.dataloader)
        return self.pip_train.epoch_size("Reader")//self.batchsize
    def reset(self):
        self.dataloader.reset()

def get_data_loader(dataset,size, batch_size, num_workers):
    dl = Data_loader(dataset,size,batch_size, num_workers)
    
    return dl
