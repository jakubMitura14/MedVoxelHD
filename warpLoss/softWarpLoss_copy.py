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
import warp as wp

##
## Benchmark based on data from dataset CT-Org https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations
## One should download the data into a folder and provide its path into data_dir variable
##    data_dir should have two subdirectories labels and volumes in volumes only images should be present in labels only labels
## Additionally result from benchmark will be saved to csv file in provided path by csvPath
##

csvPath = "/workspaces/Hausdorff_morphological/csvResB.csv"
data_dir = "/workspaces/Hausdorff_morphological/CT_ORG"



# val_transforms = Compose(
# [
#     LoadImaged(keys=["image", "label"]),
#     EnsureChannelFirstd(keys=["image", "label"]),
#     Spacingd(keys=["image", "label"], pixdim=(
#         1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#     Orientationd(keys=["image", "label"], axcodes="RAS"),
#     CropForegroundd(keys=["image", "label"], source_key="image"),
#     EnsureTyped(keys=["image", "label"]),
# ])

# train_images = sorted(
#     glob.glob(os.path.join(data_dir, "volumes", "*.nii.gz")))
# train_labels = sorted(
#     glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))

# data_dicts = [
#     {"image": image_name, "label": label_name}
#     for image_name, label_name in zip(train_images, train_labels)
# ]


# check_ds = Dataset(data=data_dicts, transform=val_transforms)
# check_loader = DataLoader(check_ds, batch_size=1)

# someDat=next(iter(check_loader))
# dat=someDat

# deviceTorch = torch.device("cuda")
# device = 'cuda'
# ii=1
# jj=2
# labelBoolTensorA =  torch.where( dat['label']==ii, 1, 0).bool().to(deviceTorch)#[:,:,100:200,100:200,100:200].contiguous()          
# summA= torch.sum(labelBoolTensorA)
# sizz= labelBoolTensorA.size()
# dim_x = sizz[0]
# dim_y = sizz[1]
# dim_z = sizz[2]


dim_x = 400
dim_y = 400
dim_z = 400
sizz= (dim_x,dim_y, dim_z )
labelBoolTensorA = torch.zeros(sizz,dtype=bool).contiguous().to('cuda') 
labelBoolTensorA[50:60,50:60,50:60]= torch.ones((10,10,10),dtype=bool).to('cuda')


randdd= torch.rand((40,40,40),dtype=float).to('cuda')

labelBoolTensorBa= torch.zeros(sizz,dtype=float).float().to('cuda')
labelBoolTensorBa[40:60,40:60,40:60]= randdd[0:20,0:20,0:20]
torch.sum(labelBoolTensorBa)


labelBoolTensorBb= torch.zeros(sizz,dtype=float).float().to('cuda')
labelBoolTensorBb[40:60,40:60,40:60]= torch.mul(randdd[0:20,0:20,0:20],2).to('cuda')
torch.sum(labelBoolTensorBb)

labelBoolTensorBc= torch.zeros(sizz,dtype=float).float().to('cuda')
labelBoolTensorBc[40:80,40:80,40:80]= torch.mul(randdd,2).to('cuda')


labelBoolTensorBd= torch.zeros(sizz,dtype=float).float().to('cuda')
labelBoolTensorBd[20:60,20:60,20:60]= torch.mul(randdd,2).to('cuda')




# labelBoolTensorA = torch.zeros(sizz,dtype=bool).contiguous().to('cuda') 
# labelBoolTensorA[5:6,5:6,5:6]= torch.ones((1,1,1),dtype=bool).to('cuda')

# labelBoolTensorBa= torch.zeros(sizz,dtype=float).float().to('cuda')
# labelBoolTensorBa[4:6,4:6,4:6]= torch.ones((2,2,2),dtype=float).to('cuda')
# torch.sum(labelBoolTensorBa)


# labelBoolTensorBb= torch.zeros(sizz,dtype=float).float().to('cuda')
# labelBoolTensorBb[4:6,4:6,4:6]= torch.mul(torch.ones((2,2,2),dtype=float),2).to('cuda')
# torch.sum(labelBoolTensorBb)

# labelBoolTensorBc= torch.zeros(sizz,dtype=float).float().to('cuda')
# labelBoolTensorBc[4:8,4:8,4:8]= torch.mul(torch.ones((4,4,4),dtype=float),2).to('cuda')

# labelBoolTensorBd= torch.zeros(sizz,dtype=float).float().to('cuda')
# labelBoolTensorBd[2:6,2:6,2:6]= torch.mul(torch.ones((4,4,4),dtype=float),2).to('cuda')




wp.init()



def prepare_tensors_for_warp_loss(y_true, y_hat,radius,device):
    """
    y_true and hould be boolean tensors with the same dimensions as y_hat (y_hat should be float tensor)
    we need to parse the arrays to format of indicies in order to be able to use warp with HashGrid
    """
    # y_true=y_true[0,0,:,:,:]
    # y_hat=y_hat[0,0,:,:,:]

    sizz= y_true.size()
    dim_x = sizz[0]
    dim_y = sizz[1]
    dim_z = sizz[2]

# grid : wp.uint64,
#                     points_in_grid: wp.array(dtype=wp.vec3),
#                     points_labelArr: wp.array(dtype=wp.types.int32),
#                     counts: wp.array(dtype=float),
#                     y_hat_arr : wp.array(dtype=float)
    num_points_gold = torch.sum(y_true).item()
    num_points_gold_false= torch.numel(y_true) - num_points_gold
    #print(f"num_points_gold {num_points_gold} num_points_gold_false {num_points_gold_false}   ")

    points_in_grid=torch.argwhere(torch.logical_not(y_true)).type(torch.float32).contiguous().to('cuda')
    points_labelArr=torch.argwhere(y_true).type(torch.float32).contiguous().to('cuda')

    counts_arr = torch.zeros(num_points_gold_false, dtype=torch.float32, requires_grad=True).to('cuda') 


    # return (points_in_grid, points_labelArr,  y_hat, counts_arr
    # ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false)
    return (points_in_grid, points_labelArr,  torch.sigmoid(y_hat), counts_arr
    ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false)




class getHausdorff_soft(torch.autograd.Function):
    """
    based on example from https://github.com/NVIDIA/warp/blob/main/examples/example_sim_fk_grad_torch.py
    """
    @staticmethod
    def forward(ctx
    ,points_in_grid, points_labelArr, y_hat, counts_arr
    ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false
    ):
        ctx.tape = wp.Tape()
        ctx.points_in_grid=wp.from_torch( points_in_grid, dtype=wp.vec3)
        ctx.points_labelArr=wp.from_torch( points_labelArr, dtype=wp.vec3)
        ctx.y_hat=wp.from_torch(y_hat , dtype=wp.types.float32)
        ctx.counts_arr=wp.from_torch(counts_arr , dtype=wp.types.float32)



        wp.synchronize()

        print(device)

        grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
        idd=grid.id
        grid.build(ctx.points_in_grid, radius)
        wp.synchronize()


        with ctx.tape:
            wp.launch(kernel=count_neighbors, dim=  num_points_gold_false, inputs= [grid.id,
                                                                            ctx.points_in_grid,
                                                                            ctx.points_labelArr,
                                                                            ctx.counts_arr,
                                                                            ctx.y_hat
                                                                            ,num_points_gold
                                                                            ,(dim_x+dim_y+dim_z)/40], device = device)#(dim_x+dim_y+dim_z)/10


        # return (wp.to_torch(ctx.counts_arr_fp),
        #         wp.to_torch(ctx.counts_arr_fn))



    @staticmethod
    def backward(ctx,points_in_grid, points_labelArr, y_hat, counts_arr
    ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false):

        # map incoming Torch grads to our output variables
        ctx.points_in_grid=wp.from_torch( points_in_grid, dtype=wp.vec3)
        ctx.points_labelArr=wp.from_torch( points_labelArr, dtype=wp.vec3)
        ctx.y_hat=wp.from_torch(y_hat , dtype=wp.types.float32)
        ctx.counts_arr=wp.from_torch(counts_arr , dtype=wp.types.float32)

        ctx.tape.backward()

        # return adjoint w.r.t. inputs
        return (wp.to_torch(ctx.tape.gradients[ctx.points_in_grid]), 
                wp.to_torch(ctx.tape.gradients[ctx.points_labelArr]),
                wp.to_torch(ctx.tape.gradients[ctx.y_hat]),
                wp.to_torch(ctx.tape.gradients[ctx.counts_arr])
                 ,None,None,None,None,None,None, None)





@wp.kernel
def count_neighbors(grid : wp.uint64,
                    points_in_grid: wp.array(dtype=wp.vec3),
                    points_labelArr: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=float),
                    y_hat_arr : wp.array(dtype=float,ndim=3),
                    points_labelArr_len: wp.types.int32,
                    max_dist: float

                    ):
    """
    grid- the data structure with set of points
    points_in_grid - points that were used to build up a grid
    points_labelArr - array with points where gold standard evaluates to true 
    points_labelArr_len - length of points_labelArr_len
    y_hat_arr - array with algorithm output
    counts - array for storing output
    max_dist - maximuym expected distance - used to scale down the values
    """
    # tid = wp.tid()
    # # order threads by cell
    i = wp.hash_grid_point_id(grid, wp.tid())
    # # query point    
    # p = points_in_grid[i]
    p = points_in_grid[i]
    dist = float(1000000)
    weight = float(0)

    # # construct query around point p
    # neighbors = wp.hash_grid_query(grid, p, radius)

    for point_label_ind in range(0,points_labelArr_len):
        point_label=points_labelArr[point_label_ind]
        # compute distance to point
        d = wp.length(p - point_label)
        if((dist) >d):
            dist=d
            weight= y_hat_arr[int(p[0]),int(p[1]),int(p[2])]
        # if(i==0):
        #     print(p[0])
        #     print(p[1])
        #     print(p[2])
        #     print("diiizt ")
        #     print(d)
        #     print("weeight ")
        #     print(y_hat_arr[int(p[0]),int(p[1]),int(p[2])])



    counts[i] = (dist)*weight#*float(max_dist) # TODO(scale down the distance)



radius=500.0
print(wp.get_devices())


def getSumHaus(labelBoolTensorA, labelBoolTensorB):
    radius=500.0
    points_in_grid, points_labelArr,  y_hat, counts_arr ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false= prepare_tensors_for_warp_loss(labelBoolTensorA,labelBoolTensorB,radius, wp.get_devices()[1])
    getHausdorff_soft.apply(points_in_grid, points_labelArr,  y_hat, counts_arr ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false)
    res= torch.sub(torch.mean(counts_arr)
                ,torch.div(torch.sum(y_hat[labelBoolTensorA])  , (num_points_gold/((dim_x+dim_y+dim_z)/10) ) ))
    #res= torch.mean(counts_arr)
    #print(torch.div(torch.sum(y_hat[labelBoolTensorA]),num_points_gold))
    #torch.div(torch.sum(y_hat[labelBoolTensorA]),num_points_gold))
    # print(torch.sum(argss[3]))
    print(res)

# points_in_grid, points_labelArr,  torch.sigmoid(y_hat), counts_arr
#     ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false


print("labelBoolTensorBa")
getSumHaus(labelBoolTensorA,labelBoolTensorBa )
print("labelBoolTensorBb")

getSumHaus(labelBoolTensorA,labelBoolTensorBb )    
print("labelBoolTensorBc")

getSumHaus(labelBoolTensorA,labelBoolTensorBc )    
print("labelBoolTensorBd")

getSumHaus(labelBoolTensorA,labelBoolTensorBd )    

labelBoolTensorBd