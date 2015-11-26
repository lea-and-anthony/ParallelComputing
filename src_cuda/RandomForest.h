// =========================================================================================
//    Structured Class-Label in Random Forests. This is a re-implementation of
//    the work we presented at ICCV'11 in Barcelona, Spain.
//
//    In case of using this code, please cite the following paper:
//    P. Kontschieder, S. Rota Bul�, H. Bischof and M. Pelillo.
//    Structured Class-Labels in Random Forests for Semantic Image Labelling. In (ICCV), 2011.
//
//    Implementation by Peter Kontschieder and Samuel Rota Bul�
//    October 2013
//
// =========================================================================================

#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdint.h>

#include "Global.h"

using namespace std;

namespace vision
{
	// =====================================================================================
	//        Class:  TNode
	//  Description:
	// =====================================================================================

	template<class SplitData, class Prediction> class TNode
	{
	public:
		// ====================  LIFECYCLE     =======================================
		static TNode<SplitData, Prediction>* CreateTNode(uint32_t start, uint32_t end)
		{
			TNode<SplitData, Prediction> *node = new TNode<SplitData, Prediction>(start, end);
			// TODO : access idx if multithread to know where it is stored
			allNodesVector.push_back(node);
			return node;
		}

		TNode() :
			idxLeft(NO_IDX), idxRight(NO_IDX), start(0), end(0), depth(0)
		{
		}

		~TNode()
		{
		}

		// ====================  ACCESSORS     =======================================
		bool isLeaf() const
		{
			return idxLeft == NO_IDX;
		}

		uint32_t getStart() const
		{
			return start;
		}

		uint32_t getEnd() const
		{
			return end;
		}

		uint32_t getNSamples() const
		{
			return end - start;
		}

		uint16_t getDepth() const
		{
			return depth;
		}

		const SplitData &getSplitData() const
		{
			return splitData;
		}

		const Prediction &getPrediction() const
		{
			return prediction;
		}

		TNode<SplitData, Prediction>* getLeft(TNode *allNodesArray) const
		{
			if (idxLeft != NO_IDX && allNodesArray != NULL)
			{
				return &allNodesArray[idxLeft];
			}
			return NULL;
		}

		TNode<SplitData, Prediction>* getLeft() const
		{
			if (idxLeft == NO_IDX)
			{
				return NULL;
			}
			return allNodesVector[idxLeft];
		}

		const TNode<SplitData, Prediction>* getLeftConst() const
		{
			return getLeft();
		}

		TNode<SplitData, Prediction>* getRight(TNode *allNodesArray) const
		{
			if (idxRight != NO_IDX && allNodesArray != NULL)
			{
				return &allNodesArray[idxRight];
			}
			return NULL;
		}

		TNode<SplitData, Prediction>* getRight() const
		{
			if (idxRight == NO_IDX)
			{
				return NULL;
			}
			return allNodesVector[idxRight];
		}

		const TNode<SplitData, Prediction>* getRightConst() const
		{
			return getRight();
		}

		// ====================  MUTATORS      =======================================

		void setSplitData(SplitData splitData)
		{
			this->splitData = splitData;
		}
		void setPrediction(Prediction prediction)
		{
			this->prediction = prediction;
		}

		void setDepth(uint16_t depth)
		{
			this->depth = depth;
		}

		void setEnd(uint32_t end)
		{
			this->end = end;
		}

		void setStart(uint32_t start)
		{
			this->start = start;
		}

		void split(uint32_t start, uint32_t middle)
		{
			assert(start >= this->start && middle >= start && middle <= end);
			if (idxLeft == NO_IDX)
			{
				idxLeft = CreateTNode(start, middle)->idx;
				idxRight = CreateTNode(middle, end)->idx;
				allNodesVector[idxLeft]->setDepth(depth + 1);
				allNodesVector[idxRight]->setDepth(depth + 1);
			}
			else
			{
				allNodesVector[idxLeft]->setStart(start);
				allNodesVector[idxLeft]->setEnd(middle);
				allNodesVector[idxRight]->setStart(middle);
			}
		}

		// ====================  OPERATORS     =======================================

		// ====================  METHODS       =======================================
		static void allNodesVectorToArray(TNode *allNodesArray)
		{
			for (size_t i = 0; i < allNodesVector.size(); i++)
			{
				memcpy(allNodesArray + i, allNodesVector[i], sizeof(TNode));
			}
		}

	protected:

		// ====================  DATA MEMBERS  =======================================

	private:
		TNode(uint32_t start, uint32_t end) :
			idxLeft(NO_IDX), idxRight(NO_IDX), start(start), end(end), depth(0)
		{
			static uint32_t iNode = 0;
			idx = iNode++;

			// cout<<endl<<"New node "<<idx<<" "<<hex<<this<<dec<<endl;
		}
		// ====================  METHODS       =======================================

		// ====================  DATA MEMBERS  =======================================
		static const uint32_t NO_IDX = (uint32_t) - 1;

	public:
		static vector<TNode*> allNodesVector;
		uint32_t idxLeft, idxRight;
		SplitData splitData;
		Prediction prediction;
		uint32_t start, end;
		uint16_t depth;
		uint32_t idx;

	};

	// -----  end of class TNode  -----

	template<class Sample, class Label>
	struct LabelledSample
	{
		Sample sample;
		Label label;
	};

	enum SplitResult
	{
		SR_LEFT = 0, SR_RIGHT = 1, SR_INVALID = 2
	};

	typedef vector<SplitResult> SplitResultsVector;

	// =====================================================================================
	//        Class:  RandomTree
	//  Description:
	// =====================================================================================
	template<class SplitData, class Sample, class Label, class Prediction, class ErrorData>
	class RandomTree
	{
	public:
		typedef LabelledSample<Sample, Label> LSample;
		typedef vector<LSample> LSamplesVector;
		// ====================  LIFECYCLE     =======================================
		RandomTree() :
			root(NULL)
		{
		}

		virtual ~RandomTree()
		{
			delete root;
			root = NULL;
		}

		// ====================  ACCESSORS     =======================================

		void save(string filename, bool includeSamples = false) const
		{
			ofstream out(filename.c_str());
			if (out.is_open() == false)
			{
				cout << "Failed to open " << filename << endl;
				return;
			}
			writeHeader(out);
			out << endl;
			write(root, out);
			out << includeSamples << " ";
			if (includeSamples)
				write(samples, out);
		}

		Prediction predict(Sample &sample) const
		{
			assert(root != NULL);
			TNode<SplitData, Prediction> *curNode = root;
			SplitResult sr = SR_LEFT;
			while (!curNode->isLeaf() && sr != SR_INVALID)
				switch (sr = split(curNode->getSplitData(), sample))
			{
				case SR_LEFT:
					curNode = curNode->getLeft();
					break;
				case SR_RIGHT:
					curNode = curNode->getRight();
					break;
				default:
					break;
			}

			return curNode->getPrediction();
		}


		// ====================  MUTATORS      =======================================

		void train(const LSamplesVector &trainingSamples, int nTrials, bool interleavedTraining = false)
		{
			samples.clear();
			samples.resize(trainingSamples.size());
			splitResults.resize(trainingSamples.size());
			copy(trainingSamples.begin(), trainingSamples.end(), samples.begin());
			root = new TNode<SplitData, Prediction>(0, (int)samples.size());
			ErrorData errorData;
			Prediction rootPrediction;
			initialize(root, errorData, rootPrediction);
			root->setPrediction(rootPrediction);

			vector<TNode<SplitData, Prediction> *> nodeList[2], *curNodeList;
			nodeList[0].reserve(samples.size());
			nodeList[1].reserve(samples.size());

			int nodeListPtr = 0;
			curNodeList = &nodeList[nodeListPtr];

			curNodeList->push_back(root);

			cout << "nTrials = " << nTrials << endl;

			while (curNodeList->size() > 0)
			{
				if (interleavedTraining)
				{
					//double initialError = getError(errorData);
					for (int t = 0; t < nTrials; ++t)
					{
						for (size_t i = 0; i < curNodeList->size(); ++i)
						{
							TNode<SplitData, Prediction> *node = (*curNodeList)[i];
							tryImprovingSplit(errorData, node);
						}
					}
					/*
					double finalError = getError(errorData);
					cout << "l " << level << " errorDelta: " << (finalError - initialError);
					cout << endl;
					*/
				}
				else
				{
					//        double initialError = getError(errorData);
					for (size_t i = 0; i < curNodeList->size(); ++i)
					{
						for (int t = 0; t < nTrials; ++t)
						{
							TNode<SplitData, Prediction> *node = (*curNodeList)[i];
							// cout<<"Node "<<node->idx<<" try "<<t<<endl;
							tryImprovingSplit(errorData, node);
						}
					}
					/*
					double finalError = getError(errorData);
					cout << "l " << level << " errorDelta: " << (finalError - initialError);
					cout << endl;
					*/
				}

				int nextList = (++nodeListPtr) % 2;
				nodeList[nextList].clear();
				for (size_t i = 0; i < curNodeList->size(); ++i)
				{
					TNode<SplitData, Prediction> *node = (*curNodeList)[i];
					if (!node->isLeaf())
					{
#ifdef _DEBUG
						cout << setprecision(4) << (float)(node->getLeft()->getEnd() - node->getStart()) / (node->getEnd() - node->getStart()) << "(" <<
							(node->getLeft()->getEnd() - node->getLeft()->getStart()) << ") / " <<
							setprecision(4) << (float)(node->getRight()->getEnd() - node->getRight()->getStart()) / (node->getEnd() - node->getStart()) << "(" <<
							node->getRight()->getEnd() - node->getRight()->getStart() << ")" << endl;
#endif
						nodeList[nextList].push_back(node->getLeft());
						nodeList[nextList].push_back(node->getRight());
					}
					else
					{
						if (updateLeafPrediction(node, cLeftPrediction))
							node->setPrediction(cLeftPrediction);
					}
				}

				nodeListPtr = nextList;
				curNodeList = &nodeList[nextList];

			}
		}

		void load(string filename)
		{
			ifstream in(filename.c_str());
			readHeader(in);
			root = TNode<SplitData, Prediction>::CreateTNode(0, 0);
			read(root, in);
			bool includeSamples;
			in >> includeSamples;
			if (includeSamples)
				read(this->samples, in);
		}

		// ====================  OPERATORS     =======================================

	protected:
		// ====================  METHODS       =======================================

		//virtual SplitData generateSplit(const TNode<SplitData, Prediction> *node) const=0;

		virtual SplitResult split(const SplitData &splitData, Sample &sample) const = 0;

		virtual bool split(const TNode<SplitData, Prediction> *node, SplitData &splitData,
			Prediction &leftPrediction, Prediction &rightPrediction) = 0;

		virtual void initialize(const TNode<SplitData, Prediction> *node, ErrorData &errorData,
			Prediction &prediction) const = 0;

		virtual void updateError(ErrorData &newError, const ErrorData &errorData,
			const TNode<SplitData, Prediction> *node, Prediction &newLeft,
			Prediction &newRight) const = 0;

		virtual double getError(const ErrorData &error) const = 0;

		// non-pure virtual function which allows to modify predictions after all node split trials are made
		virtual bool updateLeafPrediction(const TNode<SplitData, Prediction> *node, Prediction &newPrediction) const
		{
			return false;
		}

		const LSamplesVector &getLSamples() const
		{
			return samples;
		}

		LSamplesVector &getLSamples()
		{
			return samples;
		}

		SplitResultsVector &getSplitResults()
		{
			return splitResults;
		}

		const SplitResultsVector &getSplitResults() const
		{
			return splitResults;
		}

		TNode<SplitData, Prediction>* getRoot() const
		{
			return root;
		}

		virtual void writeHeader(ostream &out) const = 0;
		virtual void readHeader(istream &in) = 0;

		virtual void write(const Sample &sample, ostream &out) const = 0;
		virtual void read(Sample &sample, istream &in) const = 0;

		virtual void write(const Prediction &prediction, ostream &out) const = 0;
		virtual void read(Prediction &prediction, istream &in) const = 0;

		virtual void write(const Label &label, ostream &out) const = 0;
		virtual void read(Label &label, istream &in) const = 0;

		virtual void write(const SplitData &splitData, ostream &out) const = 0;
		virtual void read(SplitData &splitData, istream &in) const = 0;

		// ====================  DATA MEMBERS  =======================================

	protected:
		// ====================  METHODS       =======================================

		bool tryImprovingSplit(ErrorData &errorData, TNode<SplitData, Prediction> *node)
		{
			bool improved = false;

			if (split(node, cSplitData, cLeftPrediction, cRightPrediction))
			{

				double initialError = getError(errorData);
				ErrorData newErrorData;
				updateError(newErrorData, errorData, node, cLeftPrediction, cRightPrediction); //do not move this afterwards
				double deltaError = getError(newErrorData) - initialError;
				if (node->isLeaf() || deltaError < 0)
				{
					int start, middle;

					doSplit(node, start, middle);
					node->setSplitData(cSplitData);
					node->split(start, middle);
					node->getLeft()->setPrediction(cLeftPrediction);
					node->getRight()->setPrediction(cRightPrediction);
					errorData = newErrorData;
					improved = true;
				}
			}

			return improved;
		}

		void doSplit(const TNode<SplitData, Prediction> *node, int &pInvalid, int &pLeft)
		{
			pLeft = node->getStart();
			pInvalid = node->getStart();

			int pRight = node->getEnd() - 1;
			while (pLeft <= pRight)
			{
				LSample s;
				switch (splitResults[pLeft])
				{
				case SR_RIGHT:
					s = samples[pRight];
					samples[pRight] = samples[pLeft];
					samples[pLeft] = s;
					splitResults[pLeft] = splitResults[pRight];
					splitResults[pRight] = SR_RIGHT; //not necessary
					--pRight;
					break;
				case SR_INVALID:
					s = samples[pInvalid];
					samples[pInvalid] = samples[pLeft];
					samples[pLeft] = s;

					splitResults[pLeft] = splitResults[pInvalid];
					splitResults[pInvalid] = SR_INVALID;

					++pInvalid;
					++pLeft;
					break;
				case SR_LEFT:
					++pLeft;
					break;
				}
			}
		}

		virtual void write(const TNode<SplitData, Prediction> *node, ostream &out) const
		{
			out << (node->isLeaf() ? "L " : "N ");
			out << node->getStart() << " " << node->getEnd() << " " << node->getDepth() << " ";
			write(node->getSplitData(), out);
			out << " ";
			write(node->getPrediction(), out);
			out << endl;
			if (!node->isLeaf())
			{
				write(node->getLeftConst(), out);
				write(node->getRightConst(), out);
			}
		}

		virtual void read(TNode<SplitData, Prediction> *node, istream &in) const
		{
			char type;
			in >> type;
			if (type != 'L' && type != 'N')
			{
				cout << "ERROR: Unknown node type: " << type << endl;
				exit(-1);
			}
			bool isLeaf = type == 'L';

			int start, end, depth;
			in >> start;
			in >> end;
			in >> depth;
			node->setStart(start);
			node->setEnd(end);
			node->setDepth(depth);

			SplitData splitData;
			read(splitData, in);
			node->setSplitData(splitData);

			Prediction prediction;
			read(prediction, in);
			node->setPrediction(prediction);

			if (!isLeaf)
			{
				node->split(node->getStart(), node->getStart());
				read(node->getLeft(), in);
				read(node->getRight(), in);
			}
		}

		void write(const LSamplesVector &lSamples, ostream &out) const
		{
			out << lSamples.size() << " ";
			for (int i = 0; i < lSamples.size(); ++i)
			{
				write(lSamples[i].sample, out);
				out << " ";
				write(lSamples[i].label, out);
				out << " ";
			}
		}

		void read(LSamplesVector &lSamples, istream &in) const
		{
			int nSamples;
			in >> nSamples;
			lSamples.resize(nSamples);
			for (int i = 0; i < nSamples; ++i)
			{
				read(lSamples[i].sample, in);
				read(lSamples[i].label, in);
			}
		}

		// ====================  DATA MEMBERS  =======================================

		TNode<SplitData, Prediction> *root;
		LSamplesVector samples;
		SplitResultsVector splitResults;

		SplitData cSplitData;
		Prediction cLeftPrediction, cRightPrediction;
	};
	// -----  end of class RandomTree  -----

}
#endif