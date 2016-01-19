#include <iostream>
#include "NoPointerFunctions.h"

using namespace vision;

	struct Point
	{
		int x;
		int y;
	};

	__device__ FeatureType getValueNoPtr(FeatureType *features, uint8_t channel, int16_t x, int16_t y, int16_t height, int16_t width)
	{
		return features[y + x*height + channel*height*width];
	}

	__device__ FeatureType getValueIntegralNoPtr(FeatureType *features_integral, uint8_t channel, int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t h, int16_t w)
	{
		// TODO : Check if bUseIntegralImages is true before we go here
		return (FeatureType)features_integral[y2 + x2*h + channel*h*w] -
			features_integral[y2 + x1*h + channel*h*w] -
			features_integral[y1 + x2*h + channel*h*w]+
			features_integral[y1 + x1*h + channel*h*w];
	}

	// #if USE_RANDOM_BOXES
	// begin random probe box splits modification
	// split function using the randomly selected box parameters
	__device__ SplitResult splitNoPtr(const SplitData<FeatureType> &splitData, Sample<FeatureType> &sample, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral)
	{
		sample.value = getValueNoPtr(features, splitData.channel0, sample.x, sample.y, height, width);
		SplitResult centerResult = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;

#ifdef VERBOSE_PREDICTION
		std::cerr << "CPU-split: x=" << sample.x << " y=" << sample.y << " val=" << sample.value << " centerResult=" << (centerResult == SR_LEFT ? "L" : "R") << " fType=" << (int)splitData.fType;
#endif

		if (splitData.fType == 0) // single probe (center only)
		{
#ifdef VERBOSE_PREDICTION
			std::cerr << "\n";
#endif
			return centerResult;
		}

		// for cases when we have non-centered probe types
		Point pt1, pt2, pt3, pt4;

		pt1.x = sample.x + splitData.dx1 - splitData.bw1;
		pt1.y = sample.y + splitData.dy1 - splitData.bh1;

		pt2.x = sample.x + splitData.dx1 + splitData.bw1 + 1; // remember -> integral images have size w+1 x h+1
		pt2.y = sample.y + splitData.dy1 + splitData.bh1 + 1;

		int16_t w = width;
		int16_t h = height;

#ifdef VERBOSE_PREDICTION
		std::cerr << " pt1=" << pt1 << " pt2=" << pt2;
#endif

		if (pt1.x < 0 || pt2.x < 0 || pt1.y < 0 || pt2.y < 0 ||
			pt1.x > w || pt2.x > w || pt1.y > h || pt2.y > h) // due to size correction in getImgXXX we dont have to check \geq
		{
#ifdef VERBOSE_PREDICTION
			std::cerr << "\n";
#endif
			return centerResult;
		}
		else
		{
			FeatureType valueIntegral = getValueIntegralNoPtr(features_integral, splitData.channel0, pt1.x, pt1.y, pt2.x, pt2.y, height_integral, width_integral);
			if (splitData.fType == 1) // single probe (center - offset)
			{
				int16_t norm1 = (pt2.x - pt1.x) * (pt2.y - pt1.y);
				sample.value -= valueIntegral / norm1;
#ifdef VERBOSE_PREDICTION
				std::cerr << "new-val1= " << sample.value;
#endif
			}
			else                      // pixel pair probe test
			{
				pt3.x = sample.x + splitData.dx2 - splitData.bw2;
				pt3.y = sample.y + splitData.dy2 - splitData.bh2;

				pt4.x = sample.x + splitData.dx2 + splitData.bw2 + 1;
				pt4.y = sample.y + splitData.dy2 + splitData.bh2 + 1;

#ifdef VERBOSE_PREDICTION
				std::cerr << " pt3=" << pt3 << " pt4=" << pt4;
#endif

				if (pt3.x < 0 || pt4.x < 0 || pt3.y < 0 || pt4.y < 0 ||
					pt3.x > w || pt4.x > w || pt3.y > h || pt4.y > h)
				{
#ifdef VERBOSE_PREDICTION
					std::cerr << "\n";
#endif
					return centerResult;
				}

				int16_t norm1 = (pt2.x - pt1.x) * (pt2.y - pt1.y);
				int16_t norm2 = (pt4.x - pt3.x) * (pt4.y - pt3.y);

				FeatureType valueIntegral2 = getValueIntegralNoPtr(features_integral, splitData.channel1, pt3.x, pt3.y, pt4.x, pt4.y, height_integral, width_integral);
				if (splitData.fType == 2)    // sum of pair probes
				{
					sample.value = valueIntegral / norm1 + valueIntegral2 / norm2;
				}
				else if (splitData.fType == 3)  // difference of pair probes
				{
					sample.value = valueIntegral / norm1 - valueIntegral2 / norm2;
				}
				//else
					//std::cout << "ERROR: Impossible case in splitData in StrucClassSSF::split(...)" << std::endl;

#ifdef VERBOSE_PREDICTION
				std::cerr << " new-val23= " << sample.value;
#endif

			}
		}

		SplitResult res = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;
#ifdef VERBOSE_PREDICTION
		std::cerr << " result=" << (res == SR_LEFT ? "L" : "R") << "\n";
#endif
		return res;
	}

	__device__ uint32_t predictNoPtr(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t *histograms, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral)
	{
		NodeGPU *curNode = tree;
		SplitResult sr = SR_LEFT;
#pragma warning (push)
#pragma warning (disable:4309)
		while (curNode->idxLeft != NodeGPU::NO_IDX && !(sr == SR_INVALID))
#pragma warning (pop)
		{
			sr = splitNoPtr(curNode->splitData, sample, features, features_integral, height, width, height_integral, width_integral);
			switch (sr)
			{
			case SR_LEFT:
				curNode = &tree[curNode->idxLeft];
				break;
			case SR_RIGHT:
				curNode = &tree[curNode->idxRight];
				break;
			default:
				break;
			}
		}

		return curNode->idxHist;
	}