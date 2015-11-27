#ifndef NO_POINTER_FUNCTIONS_H
#define NO_POINTER_FUNCTIONS_H

#include "TNodeGPU.h"

namespace vision
{
	typedef float FeatureType;

	FeatureType getValueNoPtr(FeatureType *features, uint8_t channel, int16_t x, int16_t y, int16_t height, int16_t width);

	FeatureType getValueIntegralNoPtr(FeatureType *features_integral, uint8_t channel, int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t h, int16_t w);

	SplitResult splitNoPtr(const SplitData<FeatureType> &splitData, Sample<FeatureType> &sample, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral);

	uint32_t predictNoPtr(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t *histograms, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral);
}

#endif