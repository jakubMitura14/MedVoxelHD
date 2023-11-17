from __future__ import division, print_function

import argparse
import csv
import glob
import os
import re
import shutil
import tempfile
import unittest
import gspread as gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymia
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import (DiceMetric, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import *
from monai.transforms import (AsDiscrete, AsDiscreted, Compose,
                              CropForegroundd, EnsureChannelFirstd, LoadImage,
                              LoadImaged, Orientationd, RandCropByPosNegLabeld,
                              Resized, ScaleIntensityRanged, Spacingd)
from monai.utils import first, set_determinism
from torch.utils import benchmark
from torch.utils.cpp_extension import load

# import warp as wp



data_dir = "/data/OrganSegmentations"
lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
#help(lltm_cuda)
device = torch.device("cuda")



execute_single_case()





# cuda0 = torch.device('cuda:0')
# def my3dResult(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     arr= lltm_cuda.getHausdorffDistance_3Dres(a, b,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool).to(cuda0) ).cpu().detach().numpy()
#     print(np.sum(arr))
#     dset = f.create_dataset("3dResulttoLooki", data=arr)




