<h1> <img src="logo_hausdorff.jpeg" alt="MedVoxelHD" width="60"> MedVoxelHD </h1>


[documentation](https://jakubmitura14.github.io/MedVoxelHD/)
CUDA c++ pytorch extension for mathemathical morphology based Hausdorff distance. Repository contain dockerfile, enviroment can be also created more conviniently with vscode remote development containers using files in this repository.

## Prerequisites
In order to build docker container one need to have NVIDIA GPU hardware and NVIDIA Container Toolkit installed [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). As the example data needs also to be dowloaded and docker container e build it is advisable to have at least 30gb of space on the hard drive.

## Installation
First one need to build docker container. Easiest way to execute the code is to 
1) clone the repository
2) download example dataset from [link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890) - only Images are needed
3) install NVIDIA Container Toolkit if not already done [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
4) open the folder with repository using visual studio code
5) edit devcontainer.json mount path /media/jm/hddData1/datasets/ct_org to the path where you downloaded the CT-ORG dataset 
6) install remote development extension in vs code
7) click remote explorer next click plus at the top left and open in remote container
8) the rest of the configuration should happen automatically
 
## Usage

In order to execute the code using Python and PyTorch the python file need to be able to localize the lltm_cuda.cpp and lltm_cuda_kernel.cu files. The easiest ways to achieve it is via locating the python file in the same directory as the 2 files mentioned.
In the python file the extension needs to be first loaded. For example :
```
from torch.utils.cpp_extension import load
lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
```
There are couple ways how to execute computations of the Hausdorff distance. The most basic one return just a single number. For example :
```
lltm_cuda.getHausdorffDistance(a[0,0,:,:,:], b[0,0,:,:,:],  WIDTH,  HEIGHT,  DEPTH,0.90, torch.ones(1, dtype =bool) )
where first two entries are 3 dimensional boolean cuda pytorch tensors, that has the same shape.  WIDTH,  HEIGHT,  DEPTH Indicate the shape of a input tensor, next is the robustness percent telling how much of the points are analyzed if set to 0.9 as above 10% of the points that are most distant from other mask will be ignored, the last entry indicates wheather we are looking for True or False voxels in most cases best to set as in example above
```
Additionally we have two additional functions with the same arguments:
1) getHausdorffDistance_FullResList will give all of the distances between two masks so for example histogram can be created from this
2) getHausdorffDistance_3Dres will return 3 dimensional array indicating what is the contribution of each voxel to the overall HD distance. It makes easy to visualize where areas that are most problematic in segmentation are present.

## Example
Description of the example and more detailed explanation of the algorithm is in the [documentation](https://jakubmitura14.github.io/MedVoxelHD/)

## Acknowledgments
Logo was created using generative machine learning model dall-e 3


