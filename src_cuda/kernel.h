#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "NoPointerFunctions.h"

using namespace vision;

void startKernel(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t treeSize, uint32_t *histograms, uint32_t histSize, FeatureType *features, uint32_t featuresSize, int16_t height, int16_t width, FeatureType *features_integral, uint32_t featuresIntegralSize, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result);

__global__ void kernel(Sample<FeatureType> sample, NodeGPU *tree, uint32_t *histograms, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *result);

#endif