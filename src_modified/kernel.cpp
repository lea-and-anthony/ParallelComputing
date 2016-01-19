#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "kernel.h"

void kernel(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t *histograms, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *result)
{
	// The prediction itself.
	// The given Sample object s contains the imageId and the pixel coordinates.
	// p is an iterator to a vector over labels (attribut hist of class Prediction)
	// This labels correspond to a patch centered on position s
	// (this is the structured version of a random forest!)
	uint32_t histIterator = predictNoPtr(sample, tree, histograms, features, features_integral, height, width, height_integral, width_integral);

	for (int y = (int)sample.y - lPYOff; y <= (int)sample.y + lPYOff; ++y)
	{
		for (int x = (int)sample.x - lPXOff; x <= (int)sample.x + lPXOff; ++x, ++histIterator)
		{
			if (histograms[histIterator] >= numLabels)
			{
				std::cerr << "Invalid label in prediction: " << histograms[histIterator] << "\n";
				//asm("trap;");
				exit(1);
			}

			if (x >= 0 && x < width && y >= 0 && y < height)
			{
				result[histograms[histIterator] * height * width + y * width + x]++;
			}
		}
	}
}

