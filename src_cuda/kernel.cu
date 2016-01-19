#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "kernel.h"

void cudaDisplayError(char *functionName, cudaError_t cudaStatus)
{
	std::cerr << std::endl << functionName << " failed!" << std::endl << cudaGetErrorString(cudaStatus) << std::endl;
}

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

void startKernel(Sample<FeatureType> &sample, NodeGPU *tree, uint32_t treeSize, uint32_t *histograms, uint32_t histSize, FeatureType *features, uint32_t featuresSize, int16_t height, int16_t width, FeatureType *features_integral, uint32_t featuresIntegralSize, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *out_result)
{
	bool success = false;

	// Memory transfer for tree
	NodeGPU *treeGPU = NULL;
	success = transferMemory((void**)&treeGPU, (void*)tree, treeSize*sizeof(NodeGPU));
	if (!success)
	{
		return;
	}

	// Memory transfer for histograms
	uint32_t *histogramsGPU = NULL;
	success = transferMemory((void**)&histogramsGPU, (void*)histograms, histSize*sizeof(uint32_t));
	if (!success)
	{
		cudaFree(treeGPU);
		return;
	}

	// Memory transfer for features
	FeatureType *featuresGPU = NULL;
	success = transferMemory((void**)&featuresGPU, (void*)features, featuresSize*sizeof(FeatureType));
	if (!success)
	{
		cudaFree(treeGPU);
		cudaFree(histogramsGPU);
		return;
	}

	// Memory transfer for features_integral
	FeatureType *features_integralGPU = NULL;
	success = transferMemory((void**)&features_integralGPU, (void*)features_integral, featuresIntegralSize*sizeof(FeatureType));
	if (!success)
	{
		cudaFree(treeGPU);
		cudaFree(histogramsGPU);
		cudaFree(featuresGPU);
		return;
	}
	
	// Memory transfer for out_result
	unsigned int *out_resultGPU = NULL;
	success = transferMemory((void**)&out_resultGPU, (void*)out_result, width*height*numLabels*sizeof(unsigned int));
	if (!success)
	{
		cudaFree(treeGPU);
		cudaFree(histogramsGPU);
		cudaFree(featuresGPU);
		cudaFree(features_integralGPU);
		return;
	}

	// Kernel launch
	const int SIZE_BLOCK = 32;
	dim3 dimBlock(SIZE_BLOCK, SIZE_BLOCK);
	dim3 dimGrid((int)ceil(width * 1.0f / SIZE_BLOCK), (int)ceil(height * 1.0f / SIZE_BLOCK));
	kernel << <dimGrid, dimBlock >> > (sample, treeGPU, histogramsGPU, featuresGPU, features_integralGPU, height, width, height_integral, width_integral, numLabels, lPXOff, lPYOff, out_resultGPU);
	cudaDeviceSynchronize();

	// Kernel end
	cudaError_t cudaStatus = cudaGetLastError();
	cudaFree(treeGPU);
	cudaFree(histogramsGPU);
	cudaFree(featuresGPU);
	cudaFree(features_integralGPU);

	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaGetLastError", cudaStatus);
		cudaFree(out_resultGPU);
		return;
	}

	// Memory transfer
	cudaStatus = cudaMemcpy(out_result, out_resultGPU, width*height*numLabels*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cudaDisplayError("cudaMemcpy", cudaStatus);
		cudaFree(out_resultGPU);
		return;
	}
	cudaFree(out_resultGPU);

	return;
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
				// out_result[histograms[histIterator] * height * width + y * width + x]++;
			}
		}
	}
}
