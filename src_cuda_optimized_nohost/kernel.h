#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include "NoPointerFunctions.h"

using namespace vision;

void startKernel(void *forest, int numTrees, Sample<FeatureType> &sample, FeatureType *features, uint32_t featuresSize, int16_t height, int16_t width, FeatureType *features_integral, uint32_t featuresIntegralSize, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result);

void startKernel(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t treeSize, uint32_t *histograms, uint32_t histSize, FeatureType *features, uint32_t featuresSize, int16_t height, int16_t width, FeatureType *features_integral, uint32_t featuresIntegralSize, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result);

__global__ void initKernel(int16_t height, int16_t width, size_t numLabels, unsigned int *out_result);

__global__ void kernel(Sample<FeatureType> sample, NodeGPU *tree, uint32_t *histograms, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *result);

#endif