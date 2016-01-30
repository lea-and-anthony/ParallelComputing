// =========================================================================================
// 
// =========================================================================================
//    Original message from Peter Kontschieder and Samuel Rota Bulò:
//    Structured Class-Label in Random Forests. This is a re-implementation of
//    the work we presented at ICCV'11 in Barcelona, Spain.
//
//    In case of using this code, please cite the following paper:
//    P. Kontschieder, S. Rota Bulò, H. Bischof and M. Pelillo.
//    Structured Class-Labels in Random Forests for Semantic Image Labelling. In (ICCV), 2011.
//
//    Implementation by Peter Kontschieder and Samuel Rota Bulò
//    October 2013
//
// =========================================================================================

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <sys/stat.h>

#include "Global.h"
#include "ConfigReader.h"
#include "ImageData.h"
#include "ImageDataFloat.h"
#include "SemanticSegmentationForests.h"
#include "StrucClassSSF.h"
#include "label.h"
#include "NoPointerFunctions.h"
#include "kernel.h"

using namespace std;
using namespace vision;

typedef TNode<SplitData<FeatureType>, Prediction> Node;

/***************************************************************************
 USAGE
 ***************************************************************************/

void usage (char *com) 
{
    std::cerr<< "usage: " << com << " <configfile> <inputimage> <outputimage> <n.o.trees> <tree-model-prefix>\n"
        ;
    exit(1);
}

/***************************************************************************
 Writes profiling output (milli-seconds since last call)
 ***************************************************************************/

clock_t LastProfilingClock;

inline float profiling (const char *s, clock_t *whichClock=NULL) 
{
	if (whichClock==NULL)
		whichClock=&LastProfilingClock;

    clock_t newClock=clock();
    float res = (float) (newClock-*whichClock) / (float) CLOCKS_PER_SEC;
	if (s != NULL)
#ifdef SHUT_UP
		std::cout << s << " : " << res << std::endl;
#else
		std::cout << "Time : " << s << " : " << res << std::endl;
#endif
    *whichClock = newClock;
    return res;
}

inline float profilingTime (const char *s, time_t *whichClock) 
{
    time_t newTime=time(NULL);
    float res = (float) (newTime-*whichClock);
	if (s != NULL)
#ifdef SHUT_UP
		std::cout << s << " : " << res << std::endl;
#else
		std::cout << "Time(real) : " << s << " : " << res << std::endl;
#endif
    return res;
}

/***************************************************************************
 Test a simple image
 ***************************************************************************/
void testStructClassForest(StrucClassSSF<float> *forest, ConfigReader *cr, TrainingSetSelection<float> *pTS)
{
    int iImage;
    cv::Point pt;
    cv::Mat matConfusion;
	char strOutput[200];
    
    // Process all test images
    // result goes into ====> result[].at<>(pt)
    for (iImage = 0; iImage < pTS->getNbImages(); ++iImage)
    {
    	// Create a sample object, which contains the imageId
        Sample<float> sample;

#ifndef SHUT_UP
        std::cout << "Testing image nr. " << iImage+1 << "\n";
#endif

		sample.imageId = iImage;
		cv::Rect box(0, 0, pTS->getImgWidth(sample.imageId), pTS->getImgHeight(sample.imageId));
        cv::Mat mapResult = cv::Mat::ones(box.size(), CV_8UC1) * cr->numLabels;

        // ==============================================
        // THE CLASSICAL CPU SOLUTION
        // ==============================================

        profiling(NULL);
        int lPXOff = cr->labelPatchWidth / 2;
    	int lPYOff = cr->labelPatchHeight / 2;

        // Initialize the result matrices
		unsigned int maxLabel = cr->numLabels;
		unsigned int maxRow = box.height;
		unsigned int maxCol = box.width;
		unsigned int resultSize = maxLabel*maxRow*maxCol;
		// [numLabel * maxRow * maxCol + numRow * maxCol + numCol]
		unsigned int *result = new unsigned int[resultSize];

		// Flatten features
		FeatureType *features, *features_integral;
		int nbChannels;
		int16_t width_integral, height_integral;
		pTS->getFlattenedFeatures(iImage, &features, &nbChannels);
		pTS->getFlattenedIntegralFeatures(iImage, &features_integral, &width_integral, &height_integral);

		// Obtain forest predictions
		// Iterate over all trees
		startKernel((void*)forest, cr->numTrees, sample, features, box.width*box.height*nbChannels, box.height, box.width, features_integral, width_integral*height_integral*nbChannels, height_integral, width_integral, (size_t)cr->numLabels, lPXOff, lPYOff, result);

		free(features);
		free(features_integral);

        // Argmax of result ===> mapResult
        size_t maxIdx;
		for (pt.y = 0; pt.y < box.height; ++pt.y)
		{
			for (pt.x = 0; pt.x < box.width; ++pt.x)
			{
				maxIdx = 0;
				for (int j = 1; j < cr->numLabels; ++j)
				{
					unsigned int resultJ = result[j * maxRow * maxCol + pt.y * maxCol + pt.x];
					unsigned int resultMaxIdx = result[maxIdx * maxRow * maxCol + pt.y * maxCol + pt.x];
					maxIdx = (resultJ > resultMaxIdx) ? j : maxIdx;
				}
				mapResult.at<uint8_t>(pt) = (uint8_t)maxIdx;
			}
		}
		delete[] result;

		stringstream ss;
		ss << (iImage + 1);
		string profilingLabel = "Prediction for image " + ss.str();
		profiling(profilingLabel.c_str());

        // Write segmentation map
        sprintf(strOutput, "%s/segmap_1st_stage%04d.png", cr->outputFolder.c_str(), iImage);
        if (cv::imwrite(strOutput, mapResult)==false)
        {
			cout << "Failed to write to " << strOutput << endl;
            return;
        }

        // Write RGB segmentation map
        cv::Mat imgResultRGB;
        convertLabelToRGB(mapResult, imgResultRGB);

        sprintf(strOutput, "%s/segmap_1st_stage_RGB%04d.png", cr->outputFolder.c_str(), iImage);
        if (cv::imwrite(strOutput, imgResultRGB)==false)
        {
			cout << "Failed to write to " << strOutput << endl;
            return;
        } 
    }
}

void getFlattenedTree(void *forest_void, int numTree, NodeGPU **out_tree, uint32_t **out_histograms, uint32_t *out_treeSize, uint32_t *out_histSize)
{
	StrucClassSSF<float> *forest = (StrucClassSSF<float> *) forest_void;
	forest[numTree].getRoot()->getFlattenedTree(out_tree, out_histograms, out_treeSize, out_histSize);
}

bool getUseRandomBoxesFromTree(void *forest_void, int numTree)
{
	StrucClassSSF<float> *forest = (StrucClassSSF<float> *) forest_void;
	return forest[numTree].bUseRandomBoxes;
}

/***************************************************************************
 MAIN PROGRAM
 ***************************************************************************/

int main(int argc, char* argv[])
{
    string strConfigFile;
    ConfigReader cr;
    ImageData *idata = new ImageDataFloat();
    TrainingSetSelection<float> *pTrainingSet;
    bool bTestAll = false;
    int optNumTrees=-1;
    char *optTreeFnamePrefix=NULL;
	char buffer[2048];
    int optNoImages=1;

    srand((unsigned int)time(0));
    setlocale(LC_NUMERIC, "C");
    profiling(NULL);

#ifndef NDEBUG
    std::cout << "******************************************************\n"
    	<< "DEBUG MODE!!!!!\n"
		<< "******************************************************\n";
#endif

    if (argc!=4)
        usage(*argv);
    else
    {
        strConfigFile = argv[1];
        optNumTrees = atoi(argv[2]);
        optTreeFnamePrefix = argv[3];
    }

    if (cr.readConfigFile(strConfigFile)==false)
    {
        cout<<"Failed to read config file "<<strConfigFile<<endl;
        return -1;
	}

    // Load image data
    idata->bGenerateFeatures = true;

    if (idata->setConfiguration(cr)==false)
    {
        cout<<"Failed to initialize image data with configuration"<<endl;
        return -1;
    }

    if (bTestAll==true)
    {
        std::cout << "Set contains all images. Not supported.\n";
        exit(1);
    }
    else {
        
        // CW Create a dummy training set selection with a single image number
        pTrainingSet = new TrainingSetSelection<float>(9, idata);
        
        for (int i=0; i<optNoImages; ++i)
        	((TrainingSetSelection<float> *)pTrainingSet)->vectSelectedImagesIndices.push_back(i);
    }

#ifndef SHUT_UP
    cout<<pTrainingSet->getNbImages()<<" test images"<<endl;
#endif

    // Load forest
    StrucClassSSF<float> *forest = new StrucClassSSF<float>[optNumTrees];

    profiling("Init + feature extraction");

	cr.numTrees = optNumTrees;
#ifndef SHUT_UP
	cout << "Loading " << cr.numTrees << " trees: \n";
#endif

    for(int iTree = 0; iTree < optNumTrees; ++iTree)
    {
		sprintf(buffer, "%s%d.txt", optTreeFnamePrefix, iTree + 1);
#ifndef SHUT_UP
        std::cout << "Loading tree from file " << buffer << "\n";
#endif

        forest[iTree].bUseRandomBoxes = true;
        forest[iTree].load(buffer);
        forest[iTree].setTrainingSet(pTrainingSet);
	}
#ifndef SHUT_UP
    cout << "done!" << endl;
#endif
    profiling("Tree loading");
    
    testStructClassForest(forest, &cr, pTrainingSet);

    // delete tree;
    delete pTrainingSet;
	delete idata;
    delete [] forest;


#ifndef SHUT_UP
    std::cout << "Terminated successfully.\n";
#endif

#ifdef _WIN32
	//system("PAUSE");
#endif

	cudaDeviceReset();
    return 0;
}
