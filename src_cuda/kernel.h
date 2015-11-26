template <typename FeatureTyp> struct Sample;
typedef float FeatureType;
struct Prediction;
template<typename FeatureType> struct SplitData;
template<class SplitData, class Prediction> class TNode;
enum SplitResult;

#include <vector>
#include "NoPointerFunctions.h"

using namespace vision;

void kernel(Sample<FeatureType> &sample, TNode<SplitData<FeatureType>, Prediction> *allNodesArray, FeatureType *features, FeatureType *features_integral, int idRoot, int16_t height, int16_t width, int16_t height_integral, int16_t width_integral, size_t numLabels, int lPXOff, int lPYOff, unsigned int *result);
