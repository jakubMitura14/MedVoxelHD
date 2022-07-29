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



val_transforms = Compose(
[
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(
        1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

train_images = sorted(
    glob.glob(os.path.join(data_dir, "volumes", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]


check_ds = Dataset(data=data_dicts, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)

someDat=next(iter(check_loader))
dat=someDat

device = torch.device("cuda")
ii=1
jj=2
labelBoolTensorA =  torch.where( dat['label']==ii, 1, 0).bool().to(device)           
summA= torch.sum(labelBoolTensorA)
labelBoolTensorB =torch.where( dat['label']==jj, 1, 0).bool().to(device)
summB= torch.sum(labelBoolTensorB)

wp.init()

@wp.kernel
def count_neighbors(grid : wp.uint64,
                    radius: wp.types.float32,
                    points_to_check: wp.array(dtype=wp.vec3),
                    refArray: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=wp.types.float32),
                    ):
    tid = wp.tid()
    #print(tid)
    # order threads by cell
    #i = wp.hash_grid_point_id(grid, tid)
    #print(i)
    # # current point    
    p = points_to_check[tid]

    minn = float(1000000)
    # construct query around point p
    neighbors = wp.hash_grid_query(grid, p, radius)
    for index in neighbors:
        # compute distance to point
        d = wp.length(p - refArray[index])
        #if (d <= radius):
        minn= wp.min(d,minn)
        #print(d)
    counts[tid] = minn
    

def prepare_tensors_for_warp_loss(y_true, y_hat,radius,device):
    """
    y_true and y_hatshould be boolean tensors with the same dimensions
    we need to parse the arrays to format of indicies in order to be able to use warp with HashGrid
    """
    y_true=y_true[0,0,:,:,:]
    y_hat=y_hat[0,0,:,:,:]

    sizz= y_true.size()
    dim_x = sizz[0]
    dim_y = sizz[1]
    dim_z = sizz[2]


    #true positives
    eqq =torch.logical_and(y_true,y_hat)  #torch.eq(labelBoolTensorA,summB)
    #false negatives labelBoolTensorA - gold stadard
    fn=  torch.logical_and(torch.logical_not(eqq),y_true)
    #false positives labelBoolTensorA - gold stadard
    fp=  torch.logical_and(torch.logical_not(eqq),y_hat)
    ####first we will look around of all fp points     
    num_points_fp = torch.sum(fp).item()
    #to sparse - to get indicies - that transpose for correct dim
    fpIndicies = torch.from_numpy(np.argwhere(fp.cpu().numpy())).type(torch.float32).contiguous().to('cuda') 
    goldIndicies =  torch.from_numpy(np.argwhere(y_true.cpu().numpy())).type(torch.float32).contiguous().to('cuda') 
    counts_arr_fp = torch.zeros(num_points_fp, dtype=torch.float32).to('cuda')     
 
    num_points_fn = torch.sum(fn).item()
    fnIndicies = torch.from_numpy(np.argwhere(fn.cpu().numpy())).type(torch.float32).contiguous().to('cuda') 
    segmIndicies =  torch.from_numpy(np.argwhere(y_hat.cpu().numpy())).type(torch.float32).contiguous().to('cuda') 
    counts_arr_fn = torch.zeros(num_points_fn, dtype=torch.float32).to('cuda') 


    # fpIndicies = torch.t(fp.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
    # goldIndicies =  torch.t(y_true.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
    # counts_arr_fp = torch.zeros(num_points_fp, dtype=torch.float32).to('cuda')     
 
    # num_points_fn = torch.sum(fn).item()
    # fnIndicies = torch.t(fn.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
    # segmIndicies =  torch.t(y_hat.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
    # counts_arr_fn = torch.zeros(num_points_fn, dtype=torch.float32).to('cuda') 

    return (fpIndicies,goldIndicies,counts_arr_fp,fnIndicies,segmIndicies, counts_arr_fn
    ,radius,device,dim_x,dim_y,dim_z,num_points_fp, num_points_fn
    )
    ####secondly we will look around of all fn points 




class getHausdorff(torch.autograd.Function):
    """
    based on example from https://github.com/NVIDIA/warp/blob/main/examples/example_sim_fk_grad_torch.py
    """



    @staticmethod
    def forward(ctx
    ,fpIndicies,goldIndicies,counts_arr_fp,fnIndicies,segmIndicies, counts_arr_fn
    ,radius,device,dim_x,dim_y,dim_z,num_points_fp, num_points_fn
    ):
        ctx.tape = wp.Tape()
        ctx.fpIndicies=wp.from_torch( fpIndicies, dtype=wp.vec3)
        ctx.goldIndicies=wp.from_torch( goldIndicies, dtype=wp.vec3)
        ctx.counts_arr_fp=wp.from_torch(counts_arr_fp , dtype=wp.types.float32)
        ctx.fnIndicies=wp.from_torch(fnIndicies , dtype=wp.vec3)
        ctx.segmIndicies=wp.from_torch( segmIndicies, dtype=wp.vec3)
        ctx.counts_arr_fn=wp.from_torch( counts_arr_fn, dtype=float)
        wp.synchronize()

        grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
        grid.build(ctx.goldIndicies, radius)
        gridB = wp.HashGrid(dim_x, dim_y, dim_z, device)
        gridB.build(ctx.segmIndicies, radius)


        wp.synchronize()


        with ctx.tape:
            wp.launch(kernel=count_neighbors, dim=  num_points_fp, inputs=[grid.id
                    ,radius,
                    ctx.fpIndicies,
                    ctx.goldIndicies,
                    ctx.counts_arr_fp
                    ], device=device)

            wp.launch(kernel=count_neighbors, dim=  num_points_fn, inputs=[gridB.id
                                ,radius,
                                ctx.fnIndicies,
                                ctx.segmIndicies,
                                ctx.counts_arr_fn
                                ], device=device)



        return (wp.to_torch(ctx.counts_arr_fp),
                wp.to_torch(ctx.counts_arr_fn))



    @staticmethod
    def backward(ctx,fpIndicies,goldIndicies,counts_arr_fp,fnIndicies,segmIndicies, counts_arr_fn,radius
    ,device,dim_x,dim_y,dim_z,num_points_fp, num_points_fn):

        # map incoming Torch grads to our output variables
        ctx.fpIndicies=wp.from_torch( fpIndicies, dtype=wp.vec3)
        ctx.goldIndicies=wp.from_torch( goldIndicies, dtype=wp.vec3)
        ctx.counts_arr_fp=wp.from_torch(counts_arr_fp , dtype=wp.types.float32)
        ctx.fnIndicies=wp.from_torch(fnIndicies , dtype=wp.vec3)
        ctx.segmIndicies=wp.from_torch( segmIndicies, dtype=wp.vec3)
        ctx.counts_arr_fn=wp.from_torch( counts_arr_fn, dtype=wp.types.float32)

        ctx.tape.backward()

        # return adjoint w.r.t. inputs
        return (wp.to_torch(ctx.tape.gradients[ctx.fpIndicies]), 
                wp.to_torch(ctx.tape.gradients[ctx.goldIndicies]),
                wp.to_torch(ctx.tape.gradients[ctx.counts_arr_fp]),
                wp.to_torch(ctx.tape.gradients[ctx.fnIndicies]),
                wp.to_torch(ctx.tape.gradients[ctx.segmIndicies]),
                wp.to_torch(ctx.tape.gradients[ctx.counts_arr_fn])
                 ,None,None,None,None,None,None, None, None)



# radius = 500.0
# devices = wp.get_devices()

# print(devices)

# argss= prepare_tensors_for_warp_loss(labelBoolTensorA, labelBoolTensorB,radius,devices[1])
# ress=getHausdorff.apply(*argss)
# print(argss)
# print(ress)








# sizz= labelBoolTensorA.size()

# dim_x = sizz[0]
# dim_y = sizz[1]
# dim_z = sizz[2]


# # labelBoolTensorA = torch.zeros(dim_x,dim_y,dim_z, dtype=torch.bool)
# # labelBoolTensorB = torch.zeros(dim_x,dim_y,dim_z, dtype=torch.bool)

# # labelBoolTensorA[0,0,0]=True
# # #labelBoolTensorA[0,0,1]=True
# # labelBoolTensorA[0,0,30]=True

# # labelBoolTensorB[0,0,0]=True
# # labelBoolTensorB[0,0,4]=True
# # labelBoolTensorB[0,0,5]=True
# # labelBoolTensorB[0,0,2]=True
# # labelBoolTensorB[0,0,16]=True

# #true positives
# eqq =torch.logical_and(labelBoolTensorA,labelBoolTensorB)  #torch.eq(labelBoolTensorA,summB)
# #false negatives labelBoolTensorA - gold stadard
# fn=  torch.logical_and(torch.logical_not(eqq),labelBoolTensorA)
# #false positives labelBoolTensorA - gold stadard
# fp=  torch.logical_and(torch.logical_not(eqq),labelBoolTensorB)
# #combined



# import numpy as np

# import warp as wp
# from warp.tests.test_base import *
# wp.init()



# devices = wp.get_devices()
# device=devices[1]
# print(device)
# radius = 500.0




# @wp.kernel
# def count_neighbors(grid : wp.uint64,
#                     radius: wp.types.float32,
#                     points_to_check: wp.array(dtype=wp.vec3),
#                     refArray: wp.array(dtype=wp.vec3),
#                     counts: wp.array(dtype=wp.types.float32),
#                     ):
#     tid = wp.tid()
#     #print(tid)
#     # order threads by cell
#     #i = wp.hash_grid_point_id(grid, tid)
#     #print(i)
#     # # current point    
#     p = points_to_check[tid]

#     minn = float(1000000)
#     # construct query around point p
#     neighbors = wp.hash_grid_query(grid, p, radius)
#     for index in neighbors:
#         # compute distance to point
#         d = wp.length(p - refArray[index])
#         #if (d <= radius):
#         minn= wp.min(d,minn)
#         #print(d)
#     counts[tid] = minn
    


# #fn, fp, labelBoolTensorA , labelBoolTensorB

# ####first we will look around of all fp points 
# num_points = torch.sum(fp).item()
# print(f" num_points {num_points} ")

# print("fp  ")
# print( torch.t(fp.to_sparse().indices()).type(torch.float32).contiguous())
# print(" goldIndicies ")
# print(torch.t(labelBoolTensorA.to_sparse().indices()).type(torch.float32).contiguous() )


# fpIndicies = wp.from_torch( torch.t(fp.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
#                              , dtype=wp.vec3)
# goldIndicies = wp.from_torch( torch.t(labelBoolTensorA.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
#                              , dtype=wp.vec3)
# counts_arr_fp = wp.zeros(num_points, dtype=wp.types.float32, device=device)

# grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
# grid.build(goldIndicies, radius)
# wp.synchronize()
# #wp.launch(kernel=count_neighbors, dim=num_points, inputs=[grid.id
# wp.launch(kernel=count_neighbors, dim=  num_points, inputs=[grid.id
#                     ,radius,
#                     fpIndicies,
#                     goldIndicies,
#                     counts_arr_fp
#                     ], device=device)
# wp.synchronize()


# num_points = torch.sum(fn).item()

# fnIndicies = wp.from_torch( torch.t(fn.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
#                              , dtype=wp.vec3)
# segmIndicies = wp.from_torch( torch.t(labelBoolTensorB.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
#                              , dtype=wp.vec3)
# counts_arr_fn = wp.zeros(num_points, dtype=wp.types.float32, device=device)

# gridB = wp.HashGrid(dim_x, dim_y, dim_z, device)
# gridB.build(segmIndicies, radius)
# wp.synchronize()
# #wp.launch(kernel=count_neighbors, dim=num_points, inputs=[grid.id
# wp.launch(kernel=count_neighbors, dim=  num_points, inputs=[gridB.id
#                     ,radius,
#                     fnIndicies,
#                     segmIndicies,
#                     counts_arr_fn
#                     ], device=device)
# wp.synchronize()



# counts =counts_arr_fp.numpy()
# print(counts)
# print(f"suum: {np.max(counts)}")
# counts =counts_arr_fn.numpy()
# print(counts)


# print(f"suum: {np.max(counts)}")

# print("ffff ")













# indiciesB=labelBoolTensorB.to_sparse().indices().type(torch.int32)
# warpp_indicies_b=wp.from_torch(indiciesB, dtype=wp.types.int32)

# points_arr = wp.array(warpp_indicies_b, dtype=wp.vec3, device=device)
# counts_arr = wp.zeros(num_points, dtype=wp.types.float32, device=device)



# grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
# grid.build(points_arr, radius)
# wp.synchronize()

# wp.launch(kernel=count_neighbors, dim=num_points, inputs=[grid.id, radius, points_arr, counts_arr], device=device)
# wp.synchronize()

# counts = counts_arr.numpy()

# ####now the same with fn
# num_points = torch.sum(fn)



# print(f"maax: {np.max(counts)}")

# def test_hashgrid_query(device,labelBoolTensorA, labelBoolTensorB, eqq, fn, fp,radius):
        
#     grid = wp.HashGrid(dim_x, dim_y, dim_z, device)

#     points = [[1.0, 2.0, 2.0]
#                ,[1.0, 22.0, 2.0]
#                 ,[1.0,0.0,2.0]]


#     points_arr = wp.array(points, dtype=wp.vec3, device=device)
#     counts_arr = wp.zeros(len(points), dtype=wp.types.float32, device=device)


#     grid.build(points_arr, radius)
#     wp.synchronize()

#     wp.launch(kernel=count_neighbors, dim=len(points), inputs=[grid.id, radius, points_arr, counts_arr], device=device)
#     wp.synchronize()

#     counts = counts_arr.numpy()


#     print(f"suum: {np.max(counts)}")

# devices = wp.get_devices()
# print(devices[1])
# test_hashgrid_query(devices[1],labelBoolTensorA, labelBoolTensorB, eqq, fn, fp,radius)