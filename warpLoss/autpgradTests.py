# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import mainWarpLoss
import warp as wp
import statistics


wp.init()
devicesWarp = wp.get_devices()
print_config()

data_dir = '/workspaces/Hausdorff_morphological/spleenData/Task09_Spleen/Task09_Spleen/Task09_Spleen'
root_dir='/workspaces/Hausdorff_morphological/spleenData'


def mainPartWarpLoss(index,a,bList,radius):
    a=a[1,:,:,:]
    b=bList[index][0,:,:,:]
    argss= mainWarpLoss.prepare_tensors_for_warp_loss(a, b,radius,devicesWarp[1])
    mainWarpLoss.getHausdorff.apply(*argss)
    # print(argss[0])
    # print(argss[1])
    return torch.mean(torch.cat([argss[0],argss[1]]))

def meanWarpLoss(aList, bList):
    radius = 500.0#TODO increase
    summ=torch.zeros(1, requires_grad=True).to('cuda')
    lenSum=torch.zeros(1, requires_grad=True).to('cuda')
    catRes=torch.stack(list(map(lambda tupl: mainPartWarpLoss(tupl[0],tupl[1],decollate_batch(bList.bool()),radius) ,enumerate(decollate_batch(aList.bool())))))
  
    #print(f"waarp loss  {torch.nanmean(catRes)}")

    return torch.nanmean(catRes).to('cuda')


from torch.autograd import gradcheck

onesA=torch.zeros((2,2,15,15,15), requires_grad=True).to('cuda')
onesB=torch.ones((2,2,15,15,15), requires_grad=True).to('cuda')

from torch.autograd import gradcheck
input=(onesA, onesB )
test = gradcheck(meanWarpLoss, input, eps=1e-6, atol=1e-4)
print(test)