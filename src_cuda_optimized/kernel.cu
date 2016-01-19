#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "kernel.h"

// function from main_test_simple
void getFlattenedTree(void *forest_void, int numTree, NodeGPU **out_tree, uint32_t **out_histograms, uint32_t *out_treeSize, uint32_t *out_histSize);
bool getUseRandomBoxesFromTree(void *forest_void, int numTree);

bool transferMemory(void** dest, void* src, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(dest, size);
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMalloc", cudaStatus);
		return false;
	}
	cudaStatus = cudaMemcpy(*dest, src, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMemcpy", cudaStatus);
		cudaFree(dest);
		return false;
	}
	return true;
}

void startKernel(void *forest, int numTrees, Sample<FeatureType> &sample, FeatureType *features, uint32_t featuresSize, int16_t height, int16_t width, FeatureType *features_integral, uint32_t featuresIntegralSize, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result)
{
	void** treeHist = new void*[numTrees * 2];
	void** treeHistGPU = new void*[numTrees * 2];

	// Memory transfer for features
	FeatureType *featuresGPU = NULL;
	hostGetDevicePointer((void**)&featuresGPU, (void*)features);

	// Memory transfer for features_integral
	FeatureType *features_integralGPU = NULL;
	hostGetDevicePointer((void**)&features_integralGPU, (void*)features_integral);

	// Memory transfer for out_result
	/*for (unsigned int i = 0; i < resultSize; i++)
	{
		result[i] = 0;
	}*/
	unsigned int *out_resultGPU = NULL;
	cudaError_t cudaStatus;
	cudaStatus  = cudaMalloc((void**)&out_resultGPU, numLabels*width*height*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMalloc", cudaStatus);
		return;
	}

	int minGridSize = 0;
	int SIZE_BLOCK = 10;
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &SIZE_BLOCK, (void*)kernel);
	dim3 dimBlock(SIZE_BLOCK, SIZE_BLOCK);
	dim3 dimGrid((int)ceil(width * 1.0f / SIZE_BLOCK), (int)ceil(height * 1.0f / SIZE_BLOCK));
	initKernel << <dimGrid, dimBlock >> > (height, width, numLabels, out_resultGPU);

	for (int t = 0; t < numTrees; ++t)
	{
		if (!getUseRandomBoxesFromTree(forest, t))
		{
			continue;
		}

		// Flatten tree
		NodeGPU *tree;
		uint32_t *histograms;
		uint32_t treeSize, histSize;
		getFlattenedTree(forest, t, &tree, &histograms, &treeSize, &histSize);
		treeHist[2 * t] = (void*)tree;
		treeHist[2 * t + 1] = (void*)histograms;

		// GPU kernel
		// Memory transfer for tree
		NodeGPU *treeGPU = NULL;
		hostGetDevicePointer((void**)&treeGPU, (void*)tree);

		// Memory transfer for histograms
		uint32_t *histogramsGPU = NULL;
		hostGetDevicePointer((void**)&histogramsGPU, (void*)histograms);

		// Kernel launch
		cudaDeviceSynchronize();
		kernel << <dimGrid, dimBlock >> > (sample, treeGPU, histogramsGPU, featuresGPU, features_integralGPU, height, width, height_integral, width_integral, numLabels, lPXOff, lPYOff, out_resultGPU);
	}

	cudaDeviceSynchronize();

	// Kernel end
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaGetLastError", cudaStatus);
		for (int t = 0; t < 2 * numTrees; ++t)
			hostFree(treeHist[t]);

		cudaFree((void*)out_resultGPU);
		return;
	}

	// Memory transfer
	cudaStatus = cudaMemcpy(out_result, out_resultGPU, width*height*numLabels*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMemcpy", cudaStatus);
		for (int t = 0; t < 2 * numTrees; ++t)
			hostFree(treeHist[t]);

		cudaFree((void*)out_resultGPU);
		return;
	}

	// Memory free
	for (int t = 0; t < 2*numTrees; ++t)
		hostFree(treeHist[t]);

	cudaFree((void*)out_resultGPU);
}
/*
void startKernel(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t treeSize, uint32_t *histograms, uint32_t histSize, FeatureType *featuresGPU, uint32_t featuresSize, int16_t height, int16_t width, FeatureType *features_integralGPU, uint32_t featuresIntegralSize, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result)
{
	// Memory transfer for tree
	NodeGPU *treeGPU = NULL;
	hostGetDevicePointer((void**)&treeGPU, (void*)tree);

	// Memory transfer for histograms
	uint32_t *histogramsGPU = NULL;
	hostGetDevicePointer((void**)&histogramsGPU, (void*)histograms);

	// Memory transfer for out_result
	unsigned int *out_resultGPU = NULL;
	cudaError_t cudaStatus;
	/*cudaStatus  = cudaMalloc((void**)&out_resultGPU, numLabels*width*height*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMalloc", cudaStatus);
		return;
	}
	const int SIZE_BLOCK = 32;
	dim3 dimBlock(SIZE_BLOCK, SIZE_BLOCK);
	dim3 dimGrid((int)ceil(width * 1.0f / SIZE_BLOCK), (int)ceil(height * 1.0f / SIZE_BLOCK));
	initKernel << <dimBlock, dimGrid >> > (height, width, numLabels, out_resultGPU);
	cudaDeviceSynchronize();*/
/*
	bool success = transferMemory((void**)&out_resultGPU, (void*)out_result, width*height*numLabels*sizeof(unsigned int));
	if (!success)
	{
		return;
	}

	// Kernel launch
	const int SIZE_BLOCK = 32;
	dim3 dimBlock(SIZE_BLOCK, SIZE_BLOCK);
	dim3 dimGrid((int)ceil(width * 1.0f / SIZE_BLOCK), (int)ceil(height * 1.0f / SIZE_BLOCK));
	kernel <<<dimBlock, dimGrid >>> (sample, treeGPU, histogramsGPU, featuresGPU, features_integralGPU, height, width, height_integral, width_integral, numLabels, lPXOff, lPYOff, out_resultGPU);
	cudaDeviceSynchronize();

	// Kernel end
	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaGetLastError", cudaStatus);
		cudaFree((void*)out_resultGPU);
		return;
	}
	
	// Memory transfer
	cudaStatus = cudaMemcpy(out_result, out_resultGPU, width*height*numLabels*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMemcpy", cudaStatus);
		cudaFree((void*)out_resultGPU);
		return;
	}

	cudaFree((void*)out_resultGPU);
	return;
}
*/

__global__ void initKernel(int16_t height, int16_t width, size_t numLabels, unsigned int *out_result)
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
}

__global__ void kernel(Sample<FeatureType> sample, NodeGPU *tree, uint32_t *histograms, FeatureType *features, FeatureType *features_integral, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result)
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

	uint32_t histIterator = predictNoPtr(sample, tree, histograms, features, features_integral, height, width, height_integral, width_integral);

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
				//out_result[histograms[histIterator] * height * width + y * width + x]++;
			}
		}
	}
}
