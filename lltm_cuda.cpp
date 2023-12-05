#include <torch/extension.h>

#include <vector>

// CUDA  declarations
int getHausdorffDistance_CUDA(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim
    , const int zDim,const float robustnessPercent
    , at::Tensor numberToLookFor);

at::Tensor getHausdorffDistance_CUDA_FullResList(at::Tensor goldStandard,
    at::Tensor algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, at::Tensor numberToLookFor);

at::Tensor getHausdorffDistance_CUDA_3Dres(at::Tensor goldStandard,
    at::Tensor algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, at::Tensor numberToLookFor);


std::tuple<int, double>  benchmarkOlivieraCUDA(
   torch::Tensor goldStandard,
   torch::Tensor algoOutput
   , const  int xDim, const int yDim
   , const int zDim);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int getHausdorffDistance(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim, const int zDim
    , const float robustnessPercent
    , at::Tensor numberToLookFor) {

    CHECK_INPUT(goldStandard);
    CHECK_INPUT(algoOutput);


   return  getHausdorffDistance_CUDA(goldStandard, algoOutput, xDim, yDim, zDim, robustnessPercent, numberToLookFor);


}



at::Tensor getHausdorffDistance_FullResList(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim, const int zDim
    , const float robustnessPercent 
    , at::Tensor numberToLookFor) {

    CHECK_INPUT(goldStandard);
    CHECK_INPUT(algoOutput);

    return  getHausdorffDistance_CUDA_FullResList(goldStandard, algoOutput, xDim, yDim, zDim, robustnessPercent, numberToLookFor);

}



at::Tensor getHausdorffDistance_3Dres(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim, const int zDim
    , const float robustnessPercent
    , at::Tensor numberToLookFor) {

    CHECK_INPUT(goldStandard);
    CHECK_INPUT(algoOutput);

    return  getHausdorffDistance_CUDA_3Dres(goldStandard, algoOutput, xDim, yDim, zDim, robustnessPercent, numberToLookFor);

}



std::tuple<int, double>  benchmarkOlivieraCUDAOnlyBool(
   torch::Tensor goldStandard,
   torch::Tensor algoOutput
   , const  int xDim, const int yDim
   , const int zDim) {
   return  benchmarkOlivieraCUDA(goldStandard, algoOutput, xDim, yDim, zDim);


}

//Binding functions with template parameters
//https://github.com/pybind/pybind11/pull/3665/files
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("getHausdorffDistance", &getHausdorffDistance, "Basic version of Hausdorff distance");
    m.def("benchmarkOlivieraCUDA", &benchmarkOlivieraCUDA, "Algorithm by Oliviera - just for comparison sake - accept only boolean arrays  ");
    m.def("getHausdorffDistance_FullResList", &getHausdorffDistance_FullResList, " return additionally full result list indicating in which dilatation iterations results were recorded ");
    m.def("getHausdorffDistance_3Dres", &getHausdorffDistance_CUDA_3Dres, " return 3d res");
}
