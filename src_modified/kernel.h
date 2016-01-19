#ifndef KERNEL_H
#define KERNEL_H

#include "NoPointerFunctions.h"

using namespace vision;

void kernel(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t *histograms, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *result);

#endif