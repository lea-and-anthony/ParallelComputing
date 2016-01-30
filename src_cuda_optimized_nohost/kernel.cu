#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "kernel.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuKernelExecErrChk() { gpuCheckKernelExecutionError( __FILE__, __LINE__); }

// function from main_test_simple
void getFlattenedTree(void *forest_void, int numTree, NodeGPU **out_tree, uint32_t **out_histograms, uint32_t *out_treeSize, uint32_t *out_histSize);
bool getUseRandomBoxesFromTree(void *forest_void, int numTree);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

inline void gpuCheckKernelExecutionError(const char *file, int line)
{
	/**
	Check for invalid launch argument, then force the host to wait
	until the kernel stops and checks for an execution error.
	The synchronisation can be eliminated if there is a subsequent blocking
	API call like cudaMemcopy. In this case the cudaMemcpy call can return
	either errors which occurred during the kernel execution or those from
	the memory copy itself. This can be confusing for the beginner, so it is
	recommended to use explicit synchronisation after a kernel launch during
	debugging to make it easier to understand where problems might be arising.
	*/
	gpuAssert(cudaPeekAtLastError(), file, line);
	gpuAssert(cudaDeviceSynchronize(), file, line);
}

bool transferMemory(void** dest, void* src, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(dest, size);
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMalloc", cudaStatus);
		*dest = NULL;
		return false;
	}
	cudaStatus = cudaMemcpy(*dest, src, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMemcpy", cudaStatus);
		cudaFree(dest);
		*dest = NULL;
		return false;
	}
	return true;
}

unsigned char* startKernel(void *forest, int numTrees, Sample<FeatureType> &sample, FeatureType *features, uint32_t featuresSize, int16_t width, int16_t height, FeatureType *features_integral, uint32_t featuresIntegralSize, int16_t width_integral, int16_t height_integral, size_t numLabels, int lPXOff, int lPYOff)
{
	// Memory transfer for features
	FeatureType *featuresGPU = NULL;
	bool success = transferMemory((void**)&featuresGPU, (void*)features, featuresSize*sizeof(FeatureType));
	if (!success)
	{
		return NULL;
	}

	// Memory transfer for features_integral
	FeatureType *features_integralGPU = NULL;
	success = transferMemory((void**)&features_integralGPU, (void*)features_integral, featuresIntegralSize*sizeof(FeatureType));
	if (!success)
	{
		cudaFree(featuresGPU);
		return NULL;
	}

	// Memory transfer for out_result
	// [numLabel * maxRow * maxCol + numRow * maxCol + numCol]
	unsigned int *resultGPU = NULL;
	gpuErrchk(cudaMalloc((void**)&resultGPU, numLabels*width*height*sizeof(unsigned int)));

	// Memory allocation for mapResult
	unsigned char *mapResultGPU = NULL;
	gpuErrchk(cudaMalloc((void**)&mapResultGPU, width*height*sizeof(unsigned char)));

	int minGridSize = 0;
	int SIZE_BLOCK = 10;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &SIZE_BLOCK, (void*)kernel);
	SIZE_BLOCK = sqrt(SIZE_BLOCK);
	//std::cout << "SIZE_BLOCK = " << SIZE_BLOCK << std::endl;
	dim3 dimBlock(SIZE_BLOCK, SIZE_BLOCK);
	dim3 dimGrid((int)ceil(width * 1.0f / SIZE_BLOCK), (int)ceil(height * 1.0f / SIZE_BLOCK));

	// Initialization
	initKernel << <dimGrid, dimBlock >> > (height, width, numLabels, resultGPU, mapResultGPU);

	// Flatten tree
	NodeGPU *treeGPU = NULL;
	uint32_t *histogramsGPU = NULL;
	uint32_t *treeOffsetGPU = NULL;
	uint32_t *histOffsetGPU = NULL;
	NodeGPU **tree = new NodeGPU*[numTrees];
	uint32_t **histograms = new uint32_t*[numTrees];
	uint32_t *treeSize = new uint32_t[numTrees];
	uint32_t *histSize = new uint32_t[numTrees];
	uint32_t *treeOffset = new uint32_t[numTrees];
	uint32_t *histOffset = new uint32_t[numTrees];
	uint32_t totalTreeSize = 0;
	uint32_t totalHistSize = 0;

	for (int t = 0; t < numTrees; ++t)
	{
		if (!getUseRandomBoxesFromTree(forest, t))
		{
			continue;
		}

		// Flatten tree
		getFlattenedTree(forest, t, &tree[t], &histograms[t], &treeSize[t], &histSize[t]);
		totalTreeSize += treeSize[t];
		totalHistSize += histSize[t];
		if (t > 0)
		{
			treeOffset[t] = treeOffset[t - 1] + treeSize[t];
			histOffset[t] = histOffset[t - 1] + histSize[t];
		}
		else
		{
			treeOffset[t] = 0;
			histOffset[t] = 0;
		}
	}
	gpuKernelExecErrChk();

	gpuErrchk(cudaMalloc((void**)&treeGPU, totalTreeSize * sizeof(NodeGPU)));
	gpuErrchk(cudaMalloc((void**)&histogramsGPU, totalHistSize * sizeof(uint32_t)));

	for (int t = 0; t < numTrees; ++t)
	{
		gpuErrchk(cudaMemcpy(&treeGPU[treeOffset[t]], tree[t], treeSize[t] * sizeof(NodeGPU), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(&histogramsGPU[histOffset[t]], histograms[t], histSize[t] * sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	gpuErrchk(cudaMalloc((void**)&treeOffsetGPU, numTrees * sizeof(uint32_t)));
	gpuErrchk(cudaMalloc((void**)&histOffsetGPU, numTrees * sizeof(uint32_t)));
	gpuErrchk(cudaMemcpy(treeOffsetGPU, treeOffset, numTrees * sizeof(uint32_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(histOffsetGPU, histOffset, numTrees * sizeof(uint32_t), cudaMemcpyHostToDevice));

	// Kernel launch
	dimGrid.z = numTrees;
	kernel << <dimGrid, dimBlock >> > (sample, treeGPU, treeOffsetGPU, histogramsGPU, histOffsetGPU, featuresGPU, width, height, features_integralGPU, width_integral, height_integral, numLabels, lPXOff, lPYOff, resultGPU);

	gpuKernelExecErrChk();

	// Argmax of result ===> mapResult
	dimGrid.z = 1;
	mapResultKernel << <dimGrid, dimBlock >> > (resultGPU, width, height, numLabels, mapResultGPU);

	gpuKernelExecErrChk();

	unsigned char* mapResult = new unsigned char[width*height];
	gpuErrchk(cudaMemcpy(mapResult, mapResultGPU, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Memory free
	cudaFree(treeGPU);
	cudaFree(histogramsGPU);
	cudaFree(treeOffsetGPU);
	cudaFree(histOffsetGPU);
	cudaFree(featuresGPU);
	cudaFree(features_integralGPU);
	cudaFree(resultGPU);
	cudaFree(mapResultGPU);

	for (int t = 0; t < numTrees; ++t)
	{
		free(tree[t]);
		free(histograms[t]);
	}

	delete[] tree;
	delete[] histograms;
	delete[] treeSize;
	delete[] histSize;
	delete[] treeOffset;
	delete[] histOffset;

	return mapResult;
}

__global__ void initKernel(int16_t height, int16_t width, size_t numLabels, unsigned int *out_result, unsigned char* mapResult)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x >= width)
	{
		return;
	}
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (y >= height)
	{
		return;
	}

	for (size_t i = 0; i < numLabels; i++)
	{
		// [numLabel * maxRow * maxCol + numRow * maxCol + numCol]
		out_result[i*width*height + y*width + x] = 0;
	}
	
	mapResult[y*width + x] = numLabels;
}

__global__ void kernel(Sample<FeatureType> sample, NodeGPU *tree, uint32_t *treeOffset, uint32_t *histograms, uint32_t *histOffset, FeatureType *features, int16_t width, int16_t height, FeatureType *features_integral, int16_t width_integral, int16_t height_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result)
{
	sample.x = blockIdx.x*blockDim.x + threadIdx.x;
	if (sample.x >= width)
	{
		return;
	}
	sample.y = blockIdx.y*blockDim.y + threadIdx.y;
	if (sample.y >= height)
	{
		return;
	}

	int numTree = blockIdx.z;

	uint32_t histIterator = predictNoPtr(sample, &tree[treeOffset[numTree]], &histograms[histOffset[numTree], features, features_integral, height, width, height_integral, width_integral);
	histIterator += histOffset[numTree];

	for (int y = (int)sample.y - lPYOff; y <= (int)sample.y + lPYOff; ++y)
	{
		for (int x = (int)sample.x - lPXOff; x <= (int)sample.x + lPXOff; ++x, ++histIterator)
		{
			if (histograms[histIterator] >= numLabels)
			{
				//std::cerr << "Invalid label in prediction: " << histograms[histIterator] << "\n";
				asm("trap;");
			}

			if (x >= 0 && x < width && y >= 0 && y < height)
			{
				atomicAdd(out_result + (histograms[histIterator] * height * width + y * width + x), 1);
			}
		}
	}
}

__global__ void mapResultKernel(unsigned int *result, int16_t width, int16_t height, size_t numLabels, unsigned char* out_mapResult)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x >= width)
	{
		return;
	}

	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (y >= height)
	{
		return;
	}

	size_t maxIdx = 0;
	// [numLabel * maxRow * maxCol + numRow * maxCol + numCol]
	unsigned int resultMaxIdx = result[maxIdx * height * width + y * width + x];
	for (size_t j = 1; j < numLabels; ++j)
	{
		unsigned int resultJ = result[j * height * width + y * width + x];
		if (resultJ > resultMaxIdx)
		{
			maxIdx = j;
			resultMaxIdx = resultJ;
		}
	}
	out_mapResult[y*width + x] = (unsigned char)maxIdx;
}
