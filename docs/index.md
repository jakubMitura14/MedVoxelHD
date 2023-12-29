<img src="https://github.com/jakubMitura14/MedVoxelHD/blob/master/logo_hausdorff.jpeg?raw=true" style="display: block; margin: auto;" width="200" />

CUDA c++ pytorch extension for mathemathical morphology based Hausdorff distance. Repository contain dockerfile, enviroment can be also created more conviniently with vscode remote development containers using files in this repository.

## Prerequisites
In order to build docker container one need to have NVIDIA GPU hardware and NVIDIA Container Toolkit installed [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). As the dockerfile will download also sample data it is advised to have at least 30gb of space available on the disk.

CUDA c++ pytorch extension for mathemathical morphology based Hausdorff distance. Repository contain dockerfile, enviroment can be also created more conviniently with vscode remote development containers using files in this repository.

## Installation
First one need to build docker container. Easiest way to execute the code is to 
<ol>
  <li>clone the repository</li>
  <li>install NVIDIA Container Toolkit if not already done [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)</li>
  <li>open the folder with repository using visual studio code</li>
  <li>opeinstall remote development extension in vs code</li>
  <li>click remote explorer next click plus at the top left and open in remote container</li>
  <li>the rest of the configuration should happen automatically</li>
</ol>

 
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
The example of usage is presented in the jupyter notebook file that is present in this repository and is named case_analysis.ipynb. The file is showing how to compare the image of the gold standard liver segmentation to its dilatated version. This simulates comparison of gold standard with algorithm output.

## Algorithm details


To achieve the best performance, multiple optimizations are performed:
<ol>
  <li> Dilatations and the memory load of the required data are performed concurrently using cuda::pipeline </li>
  <li>Validations are performed only if the value of already-discovered results is lower than the total possible number of results (this information is stored in the metadata) </li>
  <li> Utilization of the work queue led to increased occupation, as each thread block has an approximately equal number of data blocks to be processed although not all data blocks must be analyzed in a given iteration</li>
  <li> Data blocks are analyzed only when both true and false voxels are present in a given data block; only in these cases can dilatations lead to any change.</li>
</ol>
  
The most significant improvements were enabled by representing the vector of boolean values as bits in a 32-bit number. This led to:
<ol>

  <li>A reduction in the required global memory relative to Boolean representation by a factor of eight</li>
  <li>A reduction of up to 32 times the number of memory loads (one memory load of a 32-bit object instead of 32 loads of eight-bit Boolean)</li>
  <li>The possibility of representing some required operations by bitwise operations</li>
  <li>Dilatations and validation can be performed as bitwise operations</li>
  <li>A reduction in the bandwidth required to fetch the data.</li>
</ol>



## Acknowledgments
Logo was created using generative machine learning model dall-e 3