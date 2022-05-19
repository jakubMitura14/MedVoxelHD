import torch
import h5py
import numpy as np
from torch.utils.cpp_extension import load
import csv
import pandas as pd

lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)
import lltm
device = torch.device("cuda")
print("bbbb")
cuda0 = torch.device('cuda:0')
f = h5py.File("D:\\data\\hdf5Data\\smallLiverDataSet.hdf5", "r+")
algoOuttD=torch.from_numpy(f['algoOuttD'][:].astype(bool)).to(cuda0)
golddD=torch.from_numpy(f['golddD'][:].astype(bool)).to(cuda0)

WIDTH = 336;
HEIGHT = 250;
DEPTH = 371;



#print(f['algoOuttD'][2])
#print(np.sum(algoOuttD))
#print(np.sum(algoOuttD))
#print(np.sum(golddD))

# import gspread as gspread
# import pymia
# import csv
# import pandas as pd

 # median version
def mymedianHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
    return np.savetxt('resForHistogram.csv' ,lltm_cuda.getHausdorffDistance_FullResList(a, b,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy(), delimiter=",")
        
def myRobustHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
    return lltm_cuda.getHausdorffDistance(a[:], b[:],  WIDTH,  HEIGHT,  DEPTH,0.90, torch.ones(1, dtype =bool) )

mymedianHd(algoOuttD, golddD,  WIDTH,  HEIGHT,  DEPTH)

#print("aaa")

#print(np.mean(lltm_cuda.getHausdorffDistance_FullResList(algoOuttD, golddD,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()))
#print(np.mean(lltm_cuda.getHausdorffDistance_FullResList(algoOuttD, golddD,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()))
#print(np.mean(lltm_cuda.getHausdorffDistance_FullResList(algoOuttD, golddD,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()))
#print(np.mean(lltm_cuda.getHausdorffDistance_FullResList(algoOuttD, golddD,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()))



#print("ccc")

def my3dResult(a, b,  WIDTH,  HEIGHT,  DEPTH):
    arr= lltm_cuda.getHausdorffDistance_3Dres(a, b,  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()
    print(np.sum(arr))
    dset = f.create_dataset("3dResulttoLookD", data=arr)

arr= lltm_cuda.getHausdorffDistance_3Dres(algoOuttD,golddD,  DEPTH,  HEIGHT,  WIDTH,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()
print(np.sum(arr))


#my3dResult(algoOuttD,golddD,  WIDTH,  HEIGHT,  DEPTH)        
#my3dResult(algoOuttD,golddD,  DEPTH ,  HEIGHT,WIDTH  )        

#a=torch.from_numpy(np.zeros((70,80,90)).astype(bool)).to(cuda0)
#b=torch.from_numpy(np.zeros((70,80,90)).astype(bool)).to(cuda0)

#a[1,0,0]=True
#b[11,0,0]=True

###a= a.to(cuda0)
###b= b.to(cuda0)
###my3dResult(a,b,  70,  70,  70)        
##f.close()
#arr= lltm_cuda.getHausdorffDistance_3Dres(a, b,  90,80,70 ,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()

#print(arr[1,0,0])
#print(arr[11,0,0])
#print(np.sum(arr))
#print("sss ")

#a=torch.from_numpy(np.zeros((70,80,90)).astype(bool)).to(cuda0)
#b=torch.from_numpy(np.zeros((70,80,90)).astype(bool)).to(cuda0)
#a[0,1,0]=True
#b[0,11,0]=True
#arr= lltm_cuda.getHausdorffDistance_3Dres(a, b,  90,80,70 ,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()
#print(arr[0,1,0])
#print(arr[0,11,0])
#print(np.sum(arr))

#print("sss ")
#a=torch.from_numpy(np.zeros((70,80,90)).astype(bool)).to(cuda0)
#b=torch.from_numpy(np.zeros((70,80,90)).astype(bool)).to(cuda0)
#a[0,0,1]=True
#b[0,0,11]=True
#arr= lltm_cuda.getHausdorffDistance_3Dres(a, b, 90,80,70 ,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()
#print(arr[0,0,1])
#print(arr[0,0,11])
#print(np.sum(arr))


f.close()

#a=torch.from_numpy(np.zeros((300,70,70)).astype(bool)).to(cuda0)
#b=torch.from_numpy(np.zeros((70,80,90)).astype(bool)).to(cuda0)


#a[0,0,1]=True
#b[0,0,11]=True
#arr= lltm_cuda.getHausdorffDistance_3Dres(a, b,  70,80,90 ,1.0, torch.ones(1, dtype =bool) ).cpu().detach().numpy()
#print(arr[0,0,1])
#print(arr[0,0,11])
#print(np.sum(arr))





# from __future__ import division
# from __future__ import print_function

# import numpy as np

# from monai.utils import first, set_determinism
# from monai.transforms import (
#     AsDiscrete,
#     AsDiscreted,
#     EnsureChannelFirstd,
#     Compose,
#     CropForegroundd,
#     LoadImaged,
#     Orientationd,
#     RandCropByPosNegLabeld,
#     ScaleIntensityRanged,
#     Spacingd,

# )
# from monai.networks.nets import UNet
# from monai.networks.layers import Norm
# from monai.metrics import DiceMetric
# from monai.losses import DiceLoss
# from monai.inferers import sliding_window_inference
# from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
# from monai.config import print_config
# from monai.apps import download_and_extract
# import torch
# import matplotlib.pyplot as plt
# import tempfile
# import shutil
# import os
# import glob

# import gspread as gspread
# import pymia
# import csv
# import pandas as pd

# from monai.transforms import *

# from monai.transforms import (
#     LoadImage, LoadImaged, EnsureChannelFirstd,
#     Resized,  Compose
# )
# from monai.config import print_config
# import re





# import argparse
# import numpy as np
# import torch
# import unittest



# from torch.utils.cpp_extension import load
# lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
# help(lltm_cuda)



# import lltm_baseline
# import lltm

# import lltm_cuda
# device = torch.device("cuda")



# from monai.metrics import HausdorffDistanceMetric
# from monai.metrics import SurfaceDistanceMetric
# from torch.utils import benchmark
# from parameterized import parameterized


# csvPath = "D:\\dataSets\\csvFromBenchmark\\csvRes.csv"
# data_dir = "D:\\dataSets\\CTORG\\"

# #robust monai calculation
# def hdToTestRobust(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     hd = HausdorffDistanceMetric(percentile=0.90)
#     hd(y_pred=a, y=b)  # callable to add metric to the buffer
#     metric = hd.aggregate().item()
#     return metric
# #not robust monai calculation
# def hdToTest(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     hd = HausdorffDistanceMetric()
#     hd(y_pred=a, y=b)  # callable to add metric to the buffer
#     metric = hd.aggregate().item()
#     return metric


# #calculate monai SurfaceDistanceMetric
# def avSurfDistToTest(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     sd = SurfaceDistanceMetric(symmetric=True)
#     sd(y_pred=a, y=b)  # callable to add metric to the buffer
#     metric = sd.aggregate().item()
#     return metric


# #my robust  version
# def myRobustHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     return lltm_cuda.getHausdorffDistance(a[0,0,:,:,:], b[0,0,:,:,:],  WIDTH,  HEIGHT,  DEPTH,0.90, torch.ones(1, dtype =bool) )

# #my not robust  version
# def myHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     return lltm_cuda.getHausdorffDistance(a[0,0,:,:,:], b[0,0,:,:,:],  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) )

# # median version
# def mymedianHd(a, b,  WIDTH,  HEIGHT,  DEPTH):
#     return torch.mean(lltm_cuda.getHausdorffDistance_FullResList(a[0,0,:,:,:], b[0,0,:,:,:],  WIDTH,  HEIGHT,  DEPTH,1.0, torch.ones(1, dtype =bool) ).type(torch.FloatTensor)  ).item()





# #for benchmarking  testNameStr the name of function defined above
# #a,b input to benchmarking
# #numberOfRuns - on the basis of how many iterations the resulting time will be established
# # return median benchmarkTime
# def pytorchBench(a,b,testNameStr, numberOfRuns,  WIDTH,  HEIGHT,  DEPTH):
#     t0 = benchmark.Timer(
#                 stmt=testNameStr+'(a, b,WIDTH,HEIGHT,DEPTH )',
#                 setup='from __main__ import '+testNameStr,
#                 globals={'a':a , 'b':b , 'WIDTH':WIDTH , 'HEIGHT':HEIGHT , 'DEPTH':DEPTH })
#     return (t0.timeit(numberOfRuns)).median


# #timeMonaiRobust= pytorchBench(a,b,"hdToTestRobust", numberOfRuns)
# #timeMonaiNonRobust= pytorchBench(a,b,"hdToTest", numberOfRuns

# def saveBenchToCSV(labelBoolTensorA,labelBoolTensorB,sizz,df, noise,distortion,translations ):
#                     #try:                
#                         #oliviera tuple return both result and benchamrking time
#                         olivieraTuple = lltm_cuda.benchmarkOlivieraCUDA(labelBoolTensorA, labelBoolTensorB,sizz[2], sizz[3],sizz[4])
#                         numberOfRuns=2#the bigger the more reliable are benchmarks but also slower
#                         #get benchmark times

#                         hdToTestRobustTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"hdToTestRobust",numberOfRuns,   sizz[2], sizz[3],sizz[4])
#                         hdToTestTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"hdToTest", numberOfRuns,  sizz[2], sizz[3],sizz[4])
#                         avSurfDistToTestTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"avSurfDistToTest", numberOfRuns,  sizz[2], sizz[3],sizz[4])

#                         myRobustHdTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"myRobustHd", numberOfRuns,  sizz[2], sizz[3],sizz[4])
#                         myHdTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"myHd",  numberOfRuns, sizz[2], sizz[3],sizz[4])
#                         mymedianHdTime= pytorchBench(labelBoolTensorA, labelBoolTensorB,"mymedianHd", numberOfRuns,  sizz[2], sizz[3],sizz[4])
#                         olivieraTime = olivieraTuple[1]
#                         #get values from the functions
#                         hdToTestRobustValue= hdToTestRobust(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])
#                         hdToTestValue= hdToTest(labelBoolTensorA, labelBoolTensorB, sizz[2], sizz[3],sizz[4])
#                         avSurfDistToTestValue= avSurfDistToTest(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])

#                         myRobustHdValue= myRobustHd(labelBoolTensorA, labelBoolTensorB,  sizz[2], sizz[3],sizz[4])
#                         myHdValue= myHd(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])
#                         mymeanHdValue= mymedianHd(labelBoolTensorA, labelBoolTensorB,   sizz[2], sizz[3],sizz[4])
#                         olivieraValue= olivieraTuple[0]
#                         #constructing row for panda data frame
#                         series = {'hdToTestRobustTime': hdToTestRobustTime
#                                   ,'hdToTestTime': hdToTestTime
#                                   ,'avSurfDistToTestTime':avSurfDistToTestTime
#                                   ,'myRobustHdTime':myRobustHdTime
#                                   ,'myHdTime': myHdTime
#                                   ,'mymedianHdTime':mymedianHdTime
#                                   ,'olivieraTime': olivieraTime
#                                   ,'hdToTestRobustValue': hdToTestRobustValue
#                                   ,'hdToTestValue':hdToTestValue
#                                   ,'myRobustHdValue': myRobustHdValue
#                                   ,'myHdValue': myHdValue
#                                   ,'mymeanHdValue': mymeanHdValue
#                                   ,'olivieraValue': olivieraValue
#                                   ,'avSurfDistToTestValue': avSurfDistToTestValue
#                                   ,'myRobustHdTime': myRobustHdTime
#                                   ,'hdToTestValue ':hdToTestValue 
#                                   ,'WIDTH' :sizz[2]
#                                   ,'HEIGHT':sizz[3]
#                                   ,'DEPTH' :sizz[4]
#                                   ,'noise' :noise
#                                   ,'distortion':distortion
#                                   ,'translations':translations }
#                         df=df.append(series, ignore_index = True)
#                     #except:
#                         #print("An exception occurred")
#                         return df

# #iterating over given data set
# def iterateOver(dat,df,noise,distortion ):
#         print("**********************   \n  ")
#         #making sure that we are dealing only with data with required metadata for spacing and orientation
#         if(dat["image_meta_dict"]["qform_code"]>0 and  dat["image_meta_dict"]["sform_code"]>0):

#             # we iterate over all masks and look for pairs of diffrent masks to compare
#             for ii in range(1,7):
#                 for jj in range(1,7):
#                     sizz = dat['image'].shape        
#                     labelBoolTensorA =  torch.where( dat['label']==ii, 1, 0).bool().to(device)            
#                     summA= torch.sum(labelBoolTensorA)
#                     labelBoolTensorB =torch.where( dat['label']==jj, 1, 0).bool().to(device)
#                     summB= torch.sum(labelBoolTensorB)
#                     print("summA %s ii %s jj %s " % (summA.item(),ii,jj ))

#                     if(summA.item()>0 and summB.item()):
#                         if((ii!=jj)>0):
#                             dfb=saveBenchToCSV(labelBoolTensorA,labelBoolTensorB,sizz,df,noise,distortion,0 )
#                             if dfb.size> df.size:
#                                 df=dfb
#                                 df.to_csv(csvPath)
#                         else:#now adding translations in z direction
#                             pass
#                             #for translationNumb in range(1,30,5):
#                             #    translated=torch.zeros_like(labelBoolTensorA)
#                             #    translated[:,:,:,:,translationNumb:sizz[4]]= labelBoolTensorA[:,:,:,:,0:(sizz[4]-translationNumb)]
#                             #    dfb=saveBenchToCSV(labelBoolTensorA,translated,sizz,df,noise,distortion,translationNumb )
#                             #    if dfb.size> df.size:
#                             #        df=dfb
#                             #        df.to_csv(csvPath)
#         return df



# def benchmarkMitura():
#     set_determinism(seed=0)
#     val_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(keys=["image", "label"], pixdim=(
#             1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         CropForegroundd(keys=["image", "label"], source_key="image"),
#         EnsureTyped(keys=["image", "label"]),
#     ])
#     val_transformsWithNoise = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(keys=["image", "label"], pixdim=(
#             1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         CropForegroundd(keys=["image", "label"], source_key="image"),
#         EnsureTyped(keys=["image", "label"]),
#         RandGaussianNoised(keys=["image", "label"], prob=1.0)
#     ])

#     val_transformsWithRandomdeformations = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(keys=["image", "label"], pixdim=(
#             1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         CropForegroundd(keys=["image", "label"], source_key="image"),
#         EnsureTyped(keys=["image", "label"]),
#         RandAffined(keys=["image", "label"], prob=1.0)
#     ])
    



#     train_images = sorted(
#         glob.glob(os.path.join(data_dir, "volumes 0-49", "*.nii.gz")))

#     train_labels = sorted(
#         glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))

#     data_dicts = [
#         {"image": image_name, "label": label_name}
#         for image_name, label_name in zip(train_images, train_labels)
#     ]
#     check_ds = Dataset(data=data_dicts, transform=val_transforms)
#     check_loader = DataLoader(check_ds, batch_size=1)

#     check_dsWithNoise = Dataset(data=data_dicts, transform=val_transformsWithNoise)
#     check_loaderWithNoise = DataLoader(check_ds, batch_size=1)

#     check_dsWithDistortions = Dataset(data=data_dicts, transform=val_transformsWithRandomdeformations)
#     check_loaderWithDistortions = DataLoader(check_ds, batch_size=1)

#     index=0;

#     #pandas data frame to save results
#     df = pd.DataFrame( columns = ['noise','distortion','hdToTestRobustTime','hdToTestTime','avSurfDistToTestTime','myRobustHdTime','myHdTime'
#                                   ,'mymedianHdTime','olivieraTime','hdToTestRobustValue','hdToTestValue '
#                                   ,'myRobustHdValue','myHdValue','mymeanHdValue','olivieraValue'
#                                   ,'avSurfDistToTestValue','WIDTH', 'HEIGHT', 'DEPTH'])
#     try:
#         for dat in check_loader:
#             df=iterateOver(dat,df,0,0)
#     except:
#         print("An exception occurred")   
        
#     try:
#         for dat in check_loaderWithDistortions: 
#             df=iterateOver(dat,df,0,1)
#     except:
#         print("An exception occurred")
#     try:
#         for dat in check_loaderWithNoise:
#             df=iterateOver(dat,df,1,0) 
#     except:
#         print("An exception occurred")

# #benchmarkMitura()




# from typing import Any, Optional

# import torch

# from monai._extensions.loader import load_module
# from monai.metrics.utils import do_metric_reduction
# from monai.metrics import CumulativeIterationMetric


# __all__ = ["MorphologicalHausdorffDistanceMetric"]


# class MorphologicalHausdorffDistanceMetric(CumulativeIterationMetric):

#     def aggregate(self):
#         """
#         Execute reduction logic for the output of `compute_hausdorff_distance`.

#         """
#         data = self.get_buffer()
#         if not isinstance(data, torch.Tensor):
#             raise ValueError("the data to aggregate must be PyTorch Tensor.")

#         # do metric reduction
#         f, not_nans = do_metric_reduction(data, self.reduction)
#         return (f, not_nans) if self.get_not_nans else f

#     def __init__(self, compare_values: torch.Tensor, percent: float = 1.0, to_invert_dims=False) -> None:
#         super().__init__()
#         self.percent = percent
#         self.to_invert_dims = to_invert_dims
#         self.compare_values = compare_values
#         self.compiled_extension = lltm_cuda

#     def _compute_tensor(self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None):
#         """
#         Compute the Hausdorff distance.
#         Important!!
#         Size of y and y_pred

#         Args:
#             y_pred: input data to compute, It must be 3 dimensional
#             y: ground truth to compute mean the distance. It must be 3 dimensional,
#             Dimensionality needs to be identical as in y_pred
#         """

#         if y.shape != y_pred.shape:
#             raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")
#         sizz = y.shape
#         if self.to_invert_dims:
#             return self.compiled_extension.getHausdorffDistance(
#                 y_pred, y, sizz[2], sizz[1], sizz[0], self.percent, self.compare_values
#             )
#         else:
#             return self.compiled_extension.getHausdorffDistance(
#                 y_pred, y, sizz[0], sizz[1], sizz[2], self.percent, self.compare_values
#             )


              
# dimA = 50

# dimAA = 170
# dimBB = 190
# dimCC = 200

# # testing single points diffrent dims
# # dim1
# compare_values = torch.ones(1)
# a = torch.zeros(dimA, dimA, dimA)
# b = torch.zeros(dimA, dimA, dimA)
# a[0, 0, 0] = 1
# b[10, 0, 0] = 1

# # dim2
# a1 = torch.zeros(dimAA, dimBB, dimCC)
# b1 = torch.zeros(dimAA, dimBB, dimCC)
# a1[0, 0, 0] = 1
# b1[0, 15, 0] = 1

# # dim3
# a2 = torch.zeros(dimAA, dimBB, dimCC)
# b2 = torch.zeros(dimAA, dimBB, dimCC)
# a2[0, 0, 10] = 1
# b2[0, 0, 150] = 1

# # testing whole llines and compare_values set to 2
# compare_values_b = torch.ones(1)
# compare_values_b[0] = 2
# a3 = torch.zeros(dimAA, dimBB, dimCC)
# b3 = torch.zeros(dimAA, dimBB, dimCC)
# a3[:, 0, 10] = 2
# b3[:, 0, 150] = 2

# a4 = torch.zeros(dimAA, dimBB, dimCC)
# b4 = torch.zeros(dimAA, dimBB, dimCC)
# a4[10, 0, :] = 2
# b4[120, 0, :] = 2


# a5 = torch.zeros(dimAA, dimBB, dimCC)
# b5 = torch.zeros(dimAA, dimBB, dimCC)
# a5[10, :, 0] = 2
# b5[120, :, 0] = 2


# # testing whole planes
# a6 = torch.zeros(dimAA, dimBB, dimCC)
# b6 = torch.zeros(dimAA, dimBB, dimCC)
# a6[10, :, :] = 2
# b6[120, :, :] = 2


# a7 = torch.zeros(dimAA, dimBB, dimCC)
# b7 = torch.zeros(dimAA, dimBB, dimCC)
# a7[:, 0, :] = 2
# b7[:, 110, :] = 2

# a8 = torch.zeros(dimAA, dimBB, dimCC)
# b8 = torch.zeros(dimAA, dimBB, dimCC)
# # a8[:, :, 20] = 2
# # b8[:,:, 130] = 2


# a8[1, 1, 20] = 2
# b8[1, 1, 130] = 2
# a8[2, 2, 20] = 2
# b8[2, 2, 130] = 2

# # multi points
# a9 = torch.zeros(dimAA, dimBB, dimCC)
# b9 = torch.zeros(dimAA, dimBB, dimCC)

# a9[0, 20, 0] = 2
# a9[0, 0, 30] = 2
# a9[40, 0, 0] = 2
# b9[0, 0, 0] = 2

# TEST_CASES = [
#     [[a, b, 1.0, compare_values], 10],
#     [[a1, b1, 1.0, compare_values], 15],
#     [[a2, b2, 1.0, compare_values], 140],
#     [[a3, b3, 1.0, compare_values_b], 140],
#     [[a4, b4, 1.0, compare_values_b], 110],
#     [[a5, b5, 1.0, compare_values_b], 110],
#     [[a6, b6, 1.0, compare_values_b], 110],
#     [[a7, b7, 1.0, compare_values_b], 110],
#     [[a8, b8, 1.0, compare_values_b], 110],  # testing robust
#     [[a6, b6, 0.9, compare_values_b], 110],
#     [[a7, b7, 0.85, compare_values_b], 110],
#     [[a8, b8, 0.8, compare_values_b], 110],  # multi points
#     [[a9, b9, 1.0, compare_values_b], 40]
# ]



# class TestHausdorffDistanceMorphological(unittest.TestCase):
#     @parameterized.expand(TEST_CASES)
#     def test_value(self, input_data, expected_value):
#     #    if(not version_leq(f"{torch.version.cuda}", "10.100") and not version_leq(f"{torch.version.cuda}", "10.200")):
#             [y_pred, y, percentt, compare_values] = input_data
#             hd_metric = MorphologicalHausdorffDistanceMetric(
#                 compare_values.to(device), percentt, True
#             )  # True only for tests
#             result = hd_metric._compute_tensor(y_pred.to(device), y.to(device))
#             print(result)
#             np.testing.assert_allclose(expected_value, result, rtol=1e-7)



# unittest.main()

# #res3DNeeded
# #    torch::Tensor resultListPointerLocalTensor;
# #    torch::Tensor resultListPointerMetaTensor;
# #    fbArgs.resultListPointerIterNumbTensor
# #        int32_t* resultListPointerMeta;
# #    int32_t* resultListPointerLocal;
# #    int32_t* resultListPointerIterNumb;


# #TODO additional benchmarks against pymia, scipy - https://docs.scipy.org/doc/scipy/search.html?q=hausdorff; py - hausdorff - https://github.com/mavillan/py-hausdorff; itk - https://discourse.itk.org/t/computing-95-hausdorff-distance/3832/7




#     #int32_t* resultListPointerMeta;
#     #int32_t* resultListPointerLocal;
#     #int32_t* resultListPointerIterNumb;
