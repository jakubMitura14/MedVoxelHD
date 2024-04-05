"""
provide convient interface for user to use
"""
from torch.utils.cpp_extension import load
import torch
lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)



def getHausdorffDistance(labelBoolTensorA,labelBoolTensorB,robustness_percent):
    """
    return a classic Hausdorf distance as Integer
    labelBoolTensorA and labelBoolTensorB are two boolean 3D tensors with the same size
    robustness_percent is a float number between 0 and 1 indicate how much voxel wise results should be kept
        for example if robustness_percent = 0.5, then only 50% of the voxels with the lowest distance will be kept
    """
    sizz=labelBoolTensorA.shape
    WIDTH,  HEIGHT,  DEPTH= sizz[0], sizz[1],sizz[2]
    return lltm_cuda.getHausdorffDistance(labelBoolTensorA , labelBoolTensorB,  WIDTH,  HEIGHT,  DEPTH,robustness_percent, torch.ones(1, dtype =bool) )


def getHausdorffDistance3Dres(labelBoolTensorA,labelBoolTensorB,robustness_percent):
    """
    return a 3D tensor of the same shape as input with per voxel hausdorff distance 
    labelBoolTensorA and labelBoolTensorB are two boolean 3D tensors with the same size
    robustness_percent is a float number between 0 and 1 indicate how much voxel wise results should be kept
        for example if robustness_percent = 0.5, then only 50% of the voxels with the lowest distance will be kept
    """
    sizz=labelBoolTensorA.shape
    WIDTH,  HEIGHT,  DEPTH= sizz[0], sizz[1],sizz[2]
    return lltm_cuda.getHausdorffDistance_3Dres(labelBoolTensorA , labelBoolTensorB,  WIDTH,  HEIGHT,  DEPTH,robustness_percent, torch.ones(1, dtype =bool) )


def getHausdorffDistanceFullResList(labelBoolTensorA,labelBoolTensorB,robustness_percent):
    """
    return a 1D tensor indicating per voxel hausdorff distance for futher analysis 
    labelBoolTensorA and labelBoolTensorB are two boolean 3D tensors with the same size
    robustness_percent is a float number between 0 and 1 indicate how much voxel wise results should be kept
        for example if robustness_percent = 0.5, then only 50% of the voxels with the lowest distance will be kept
    """
    sizz=labelBoolTensorA.shape
    WIDTH,  HEIGHT,  DEPTH= sizz[0], sizz[1],sizz[2]
    return lltm_cuda.getHausdorffDistance_FullResList(labelBoolTensorA , labelBoolTensorB,  WIDTH,  HEIGHT,  DEPTH,robustness_percent, torch.ones(1, dtype =bool) )

def getAverageHausdorffDistance(labelBoolTensorA,labelBoolTensorB,robustness_percent):
    """
    return a floating point number that is average of all the voxel wise hausdorff distance
    labelBoolTensorA and labelBoolTensorB are two boolean 3D tensors with the same size
    robustness_percent is a float number between 0 and 1 indicate how much voxel wise results should be kept
        for example if robustness_percent = 0.5, then only 50% of the voxels with the lowest distance will be kept
    """
    sizz=labelBoolTensorA.shape
    WIDTH,  HEIGHT,  DEPTH= sizz[0], sizz[1],sizz[2]
    return torch.mean(lltm_cuda.getHausdorffDistance_FullResList(labelBoolTensorA , labelBoolTensorB,  WIDTH,  HEIGHT,  DEPTH,robustness_percent, torch.ones(1, dtype =bool) ).type(torch.FloatTensor) ).item()

