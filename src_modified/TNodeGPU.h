#ifndef T_NODE_GPU_H
#define T_NODE_GPU_H

#include <stdint.h>

namespace vision
{
	template <typename FeatureType> struct Sample
	{
		int16_t x, y;
		uint16_t imageId;
		FeatureType value;
	};

	template<typename FeatureType> struct SplitData
	{
		SplitData() : dx1(0), dx2(0), dy1(0), dy2(0), channel0(0), channel1(0), fType(0), thres(0)
		{}

		int16_t dx1, dx2;
		int16_t dy1, dy2;
		int8_t bw1, bh1, bw2, bh2;
		uint8_t channel0;    // number of feature channels is restricted by 255
		uint8_t channel1;
		uint8_t fType;           // CW: split type
		FeatureType thres;       // short is sufficient in range
	};

	enum SplitResult
	{
		SR_LEFT = 0, SR_RIGHT = 1, SR_INVALID = 2
	};
}

template<class SplitData>
struct TNodeGPU
{
	static const uint32_t NO_IDX = (uint32_t) -1;

	SplitData splitData;
	uint32_t idxHist;
	uint32_t sizeHist;
	uint32_t idxLeft;
	uint32_t idxRight;
	uint32_t idx;

};

typedef TNodeGPU<vision::SplitData<float>> NodeGPU;

#endif