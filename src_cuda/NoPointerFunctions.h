#ifndef NO_POINTER_FUNCTIONS_H
#define NO_POINTER_FUNCTIONS_H

namespace vision
{
	typedef float FeatureType;

	FeatureType getValueNoPtr(FeatureType *features, uint8_t channel, int16_t x, int16_t y, int16_t height, int16_t width);

	FeatureType getValueIntegralNoPtr(FeatureType *features_integral, uint8_t channel, int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t h, int16_t w);

	SplitResult splitNoPtr(const SplitData<FeatureType> &splitData, Sample<FeatureType> &sample, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral);

	vector<uint32_t>::const_iterator predictNoPtr(Sample<FeatureType> &sample, TNode<SplitData<FeatureType>, Prediction> *allNodesArray, FeatureType *features, FeatureType *features_integral, int idRoot, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral);
}

#endif