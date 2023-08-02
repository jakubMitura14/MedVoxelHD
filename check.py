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
#import warpLoss.mainWarpLoss



wp.init()
devicesWarp = wp.get_devices()



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
    # y_true=y_true[0,0,:,:,:]
    # y_hat=y_hat[0,0,:,:,:]

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

    return (counts_arr_fp,counts_arr_fn,fpIndicies,goldIndicies,fnIndicies,segmIndicies
    ,radius,device,dim_x,dim_y,dim_z,num_points_fp, num_points_fn
    )
    ####secondly we will look around of all fn points 




class getHausdorff(torch.autograd.Function):
    """
    based on example from https://github.com/NVIDIA/warp/blob/main/examples/example_sim_fk_grad_torch.py
    """



    @staticmethod
    def forward(ctx
    ,counts_arr_fp,counts_arr_fn,fpIndicies,goldIndicies,fnIndicies,segmIndicies
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







##
## Benchmark based on data from dataset CT-Org https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations
## One should download the data into a folder and provide its path into data_dir variable
##    data_dir should have two subdirectories labels and volumes in volumes only images should be present in labels only labels
## Additionally result from benchmark will be saved to csv file in provided path by csvPath
##

csvPath = "/workspaces/Hausdorff_morphological/csvResC.csv"
data_dir = "/workspaces/Hausdorff_morphological/CT_ORG"

from torch.utils.cpp_extension import load

lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
#help(lltm_cuda)
device = torch.device("cuda")

#robust monai calculation
def hdToTestRobust(a, b,  WIDTH,  HEIGHT,  DEPTH):
    hd = HausdorffDistanceMetric(percentile=0.90)
    hd(y_pred=a, y=b)  # callable to add metric to the buffer
    metric = hd.aggregate().item()
    return metric
#not robust monai calculation
def hdToTest(a, b,  WIDTH,  HEIGHT,  DEPTH):
    hd = HausdorffDistanceMetric()
    hd(y_pred=a, y=b)  # callable to add metric to the buffer
    metric = hd.aggregate().item()
    return metric


#calculate monai SurfaceDistanceMetric
def avSurfDistToTest(a, b,  WIDTH,  HEIGHT,  DEPTH):
    sd = SurfaceDistanceMetric(symmetric=True)
    sd(y_pred=a, y=b)  # callable to add metric to the buffer
    metric = sd.aggregate().item()
    return metric


#my robust  version
def myRobustHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
    return lltm_cuda.getHausdorffDistance(a[0,0,:,:,:], b[0,0,:,:,:],  WIDTH,  HEIGHT,  DEPTH,0.90, torch.ones(1, dtype =bool) )

#my not robust  version
def myHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
    return lltm_cuda.getHausdorffDistance(a[0,0,:,:,:], b[0,0,:,:,:],  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) )

# median version
def mymedianHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
    return torch.mean(lltm_cuda.getHausdorffDistance_FullResList(a[0,0,:,:,:], b[0,0,:,:,:],  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).type(torch.FloatTensor)  ).item()

def meanWarpLoss(a, b,  WIDTH,  HEIGHT,  DEPTH):
    radius = 500.0#TODO increase
    a=a[0,0,:,:,:]
    b=b[0,0,:,:,:]
    #print(devicesWarp)

    argss= prepare_tensors_for_warp_loss(a, b,radius,devicesWarp[1])
    ress=getHausdorff.apply(*argss)
    # argss= warpLoss.mainWarpLoss.prepare_tensors_for_warp_loss(a, b,radius,devicesWarp[1])
    # ress=warpLoss.mainWarpLoss.getHausdorff.apply(*argss)
    sumA=torch.sum(ress[0]).item()
    sumB=torch.sum(ress[1]).item()
    lenSum=torch.numel(argss[0])+torch.numel(ress[1])
    print(f"waarp loss  {(sumA+sumB)/lenSum}")
    return (sumA+sumB)/lenSum
    


#for benchmarking  testNameStr the name of function defined above
#a,b input to benchmarking
#numberOfRuns - on the basis of how many iterations the resulting time will be established
# return median benchmarkTime
def pytorchBench(a,b,testNameStr, numberOfRuns,  WIDTH,  HEIGHT,  DEPTH):
    t0 = benchmark.Timer(
                stmt=testNameStr+'(a, b,WIDTH,HEIGHT,DEPTH )',
                setup='from __main__ import '+testNameStr,
                globals={'a':a , 'b':b , 'WIDTH':WIDTH , 'HEIGHT':HEIGHT , 'DEPTH':DEPTH })
    return (t0.timeit(numberOfRuns)).median



def saveBenchToCSV(labelBoolTensorA,labelBoolTensorB,sizz,df, noise,distortion,translations ):
                    try:                
                        #oliviera tuple return both result and benchamrking time
                        olivieraTuple = lltm_cuda.benchmarkOlivieraCUDA(labelBoolTensorA, labelBoolTensorB,sizz[2], sizz[3],sizz[4])
                        numberOfRuns=2#the bigger the more reliable are benchmarks but also slower
                        #get benchmark times

                        warpLosss=pytorchBench(labelBoolTensorA, labelBoolTensorB,"meanWarpLoss",numberOfRuns,   sizz[2], sizz[3],sizz[4])

                        hdToTestRobustTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"hdToTestRobust",numberOfRuns,   sizz[2], sizz[3],sizz[4])
                        hdToTestTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"hdToTest", numberOfRuns,  sizz[2], sizz[3],sizz[4])
                        avSurfDistToTestTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"avSurfDistToTest", numberOfRuns,  sizz[2], sizz[3],sizz[4])

                        myRobustHdTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"myRobustHd", numberOfRuns,  sizz[2], sizz[3],sizz[4])
                        myHdTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"myHd",  numberOfRuns, sizz[2], sizz[3],sizz[4])
                        mymedianHdTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"mymedianHd", numberOfRuns,  sizz[2], sizz[3],sizz[4])
                        olivieraTime = olivieraTuple[1]
                        #get values from the functions
                        warpLosssVal=meanWarpLoss(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])


                        hdToTestRobustValue= hdToTestRobust(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])
                        hdToTestValue= hdToTest(labelBoolTensorA, labelBoolTensorB, sizz[2], sizz[3],sizz[4])
                        avSurfDistToTestValue= avSurfDistToTest(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])

                        myRobustHdValue= myRobustHd(labelBoolTensorA, labelBoolTensorB,  sizz[2], sizz[3],sizz[4])
                        myHdValue= myHd(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])
                        mymeanHdValue= mymedianHd(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])
                        olivieraValue= olivieraTuple[0]
                        #constructing row for panda data frame
                        series = {'warpLossTime' :warpLosss
                                    ,'warpLossValue' :warpLosssVal
                                    ,'hdToTestRobustTime': hdToTestRobustTime
                                  ,'hdToTestTime': hdToTestTime
                                  ,'avSurfDistToTestTime':avSurfDistToTestTime
                                  ,'myRobustHdTime':myRobustHdTime
                                  ,'myHdTime': myHdTime
                                  ,'mymedianHdTime':mymedianHdTime
                                  ,'olivieraTime': olivieraTime
                                  ,'hdToTestRobustValue': hdToTestRobustValue
                                  ,'hdToTestValue':hdToTestValue
                                  ,'myRobustHdValue': myRobustHdValue
                                  ,'myHdValue': myHdValue
                                  ,'mymeanHdValue': mymeanHdValue
                                  ,'olivieraValue': olivieraValue
                                  ,'avSurfDistToTestValue': avSurfDistToTestValue
                                  ,'myRobustHdTime': myRobustHdTime
                                  ,'hdToTestValue ':hdToTestValue 
                                  ,'WIDTH' :sizz[2]
                                  ,'HEIGHT':sizz[3]
                                  ,'DEPTH' :sizz[4]
                                  ,'noise' :noise
                                  ,'distortion':distortion
                                  ,'translations':translations }
                        df=df.append(series, ignore_index = True)
                    except:
                        print("An exception occurred")
                    return df

#iterating over given data set
def iterateOver(dat,df,noise,distortion ):
        print("**********************   \n  ")
        #making sure that we are dealing only with data with required metadata for spacing and orientation
        if(dat["image_meta_dict"]["qform_code"]>0 and  dat["image_meta_dict"]["sform_code"]>0):

            # we iterate over all masks and look for pairs of diffrent masks to compare
            for ii in range(1,7):
                for jj in range(1,7):
                    sizz = dat['image'].shape        
                    labelBoolTensorA =  torch.where( dat['label']==ii, 1, 0).bool().to(device)            
                    summA= torch.sum(labelBoolTensorA)
                    labelBoolTensorB =torch.where( dat['label']==jj, 1, 0).bool().to(device)
                    summB= torch.sum(labelBoolTensorB)
                    print("summA %s ii %s jj %s " % (summA.item(),ii,jj ))

                    if(summA.item()>0 and summB.item()):
                        if((ii!=jj)>0):
                            dfb=saveBenchToCSV(labelBoolTensorA,labelBoolTensorB,sizz,df,noise,distortion,0 )
                            if dfb.size> df.size:
                                df=dfb
                                df.to_csv(csvPath)
                        else:#now adding translations in z direction
                            pass
                            #for translationNumb in range(1,30,5):
                            #    translated=torch.zeros_like(labelBoolTensorA)
                            #    translated[:,:,:,:,translationNumb:sizz[4]]= labelBoolTensorA[:,:,:,:,0:(sizz[4]-translationNumb)]
                            #    dfb=saveBenchToCSV(labelBoolTensorA,translated,sizz,df,noise,distortion,translationNumb )
                            #    if dfb.size> df.size:
                            #        df=dfb
                            #        df.to_csv(csvPath)
        return df



def benchmarkMitura():
    """
    main function responsible for iterating over dataset executing algorithm and its reference 
    and storing benchmark results in csv through pandas dataframe
    """

    set_determinism(seed=0)
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
    val_transformsWithNoise = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
        RandGaussianNoised(keys=["image", "label"], prob=1.0)
    ])

    val_transformsWithRandomdeformations = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
        RandAffined(keys=["image", "label"], prob=1.0)
    ])
    

    print("aaa 1 ")
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

    check_dsWithNoise = Dataset(data=data_dicts, transform=val_transformsWithNoise)
    check_loaderWithNoise = DataLoader(check_ds, batch_size=1)

    check_dsWithDistortions = Dataset(data=data_dicts, transform=val_transformsWithRandomdeformations)
    check_loaderWithDistortions = DataLoader(check_ds, batch_size=1)

    index=0;

    #pandas data frame to save results
    df = pd.DataFrame()
    # df = pd.DataFrame( columns = ['noise','distortion','hdToTestRobustTime','hdToTestTime','avSurfDistToTestTime','myRobustHdTime','myHdTime'
    #                               ,'mymedianHdTime','olivieraTime','hdToTestRobustValue','hdToTestValue '
    #                               ,'myRobustHdValue','myHdValue','mymeanHdValue','olivieraValue'
    #                               ,'avSurfDistToTestValue','WIDTH', 'HEIGHT', 'DEPTH'])
    print("aaa 2 ")

    for dat in check_loader:
            df=iterateOver(dat,df,0,0)
    
    try:
        for dat in check_loader:
            df=iterateOver(dat,df,0,0)
    except:
        print("An exception occurred")   
        
    try:
        for dat in check_loaderWithDistortions: 
            df=iterateOver(dat,df,0,1)
    except:
        print("An exception occurred")
    try:
        for dat in check_loaderWithNoise:
            df=iterateOver(dat,df,1,0) 
    except:
        print("An exception occurred")

benchmarkMitura()


# cuda0 = torch.device('cuda:0')
# def my3dResult(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     arr= lltm_cuda.getHausdorffDistance_3Dres(a, b,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool).to(cuda0) ).cpu().detach().numpy()
#     print(np.sum(arr))
#     dset = f.create_dataset("3dResulttoLooki", data=arr)




