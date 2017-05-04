/*Copyright 2007,2008 Alex Graves

This file is part of nnl.

nnl is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

nnl is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with nnl.  If not, see <http://www.gnu.org/licenses/>.*/

#include <iostream>
#include <fstream>
#include <limits>
#include "LabellingOutputLayer.h"
#include "NnMath.h"
#include "ActivationFunctions.h"
#include "Typedefs.h"
#include "Helpers.h"

//#define ALL_FVS
//#define BLANK_UNIT


#ifdef BLANK_UNIT
static int blankUnit;
#endif
//global debug switch
extern bool sequenceDebugOutput;

LabellingOutputLayer::LabellingOutputLayer(const string& nam, const vector<string>& lab, int numDims, vector<string>& criteria):
		name(nam),
		labels(lab),
#ifdef BLANK_UNIT
		size((int)lab.size() + 1)
#else
		size((int)lab.size())
#endif
{
#ifdef BLANK_UNIT
	blankUnit = size - 1;
#endif
	criteria.push_back("labelErrorRate");
	criteria.push_back("labMlError");
	criteria.push_back("seqErrorRate");
	forwardVars.resize(size);
	backwardVars.resize(size);
}

LabellingOutputLayer::~LabellingOutputLayer(void)
{
}

void LabellingOutputLayer::feedForward(const vector<int>& dims, int seqLength, const DataSequence& seq)
{
	seqDims = dims;
	const double logTwo = safeLog(2);
	actBuffer.resize(seqLength * size);
	logActBuffer.resize(actBuffer.size());
	logOneMinusActBuffer.resize(actBuffer.size());
	errorBuffer.resize(actBuffer.size());
	fill(actBuffer.begin(), actBuffer.end(), 0);
	double* actBegin = &actBuffer.front();
	double* logActBegin = &logActBuffer.front();
	double* logOneMinusActBegin = &logOneMinusActBuffer.front();
	int offset = 0;
	for (int i = 0; i < seqLength; ++i, logOneMinusActBegin += size, actBegin += size, logActBegin += size)
	{
		//sum acts from network
		double* actEnd = actBegin + size;
		for (VPBCCI it = inputs.begin(); it != inputs.end(); ++it)
		{
			(*it)->feedForward(actBegin, actEnd, i);
		}
		
		//pass them through squashing function
		//TODO work out more efficient way of calculating log acts
		for (int i = 0; i < size; ++i)
		{
			double x = actBegin[i];
			actBegin[i] = Logistic::fn(x);
#ifndef BLANK_UNIT		
			logActBegin[i] = safeLog(actBegin[i]);//-logAdd(0, -x);
			logOneMinusActBegin[i] = safeLog(1 - actBegin[i]);//logMinus(0, -2*x);
#endif
		}
#ifdef BLANK_UNIT
		for (int i = 0; i < blankUnit; ++i)
		{
			double prob = actBegin[i] * actBegin[blankUnit];
			logActBegin[i] = safeLog(prob);//-logAdd(0, -x);
			logOneMinusActBegin[i] = safeLog(1 - prob);//logMinus(0, -2*x);
		}
#endif
	}
}

void LabellingOutputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(actBuffer, path + name + "_activations", size, &seqDims, &labels);
	printBufferToFile(errorBuffer, path + name + "_errors", size, &seqDims, &labels);
}

void LabellingOutputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" LabellingOutputLayer size " << size << endl;
}

const string& LabellingOutputLayer::getName() const
{
	return name;
}

void LabellingOutputLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{
	*errBegin = &errorBuffer[offset * size];
	*errEnd = *errBegin + size;
}

void LabellingOutputLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &actBuffer[offset * size];
	*actEnd = *actBegin + size;
}

int LabellingOutputLayer::inputSize() const
{
	return size;
}

int LabellingOutputLayer::outputSize() const
{
	return size;
}

void LabellingOutputLayer::updateDerivs(int offset) 
{
	const double* errBegin = 0;
	const double* errEnd = 0;
	getErrors(&errBegin, &errEnd, offset);
	InputConnsImpl::updateDerivs(errBegin, errEnd, offset);
}

//TODO FIX 4 SLICES
void LabellingOutputLayer::setErrToOutputDeriv(int timestep, int outputNum)
{
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	int offset = (timestep * size) + outputNum;
	errorBuffer[offset] = Logistic::deriv(actBuffer[offset]);
}

double LabellingOutputLayer::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	//RUN THE FORWARD BACKWARD ALGORITHM
	int T = (int)actBuffer.size() / size;
	const vector<int>& labelCounts = seq.labelCounts;
#ifdef BLANK_UNIT
	if (labelCounts.size() != blankUnit)
#else
	if (labelCounts.size() != size)
#endif
	{
		cerr <<  "ERROR LabellingOutputLayer::injectSequenceErrors labelCounts.size() ";
		cerr << (int)labelCounts.size() << " !=  size " << size << ", exiting" << endl;
		exit(0);
	}
	double logProb = 0;
#ifdef BLANK_UNIT
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	for (int k = 0; k < blankUnit; ++k)
#else
	for (int k = 0; k < size; ++k)
#endif
	{
		double kLogProb;
		vector<double>& fvs = forwardVars[k];
		vector<double>& bvs = backwardVars[k];
		int numLabels = labelCounts[k] + 1;
		if (numLabels > 1)
		{
			fvs.resize(T * numLabels);
			fill(fvs.begin(), fvs.end(), -doubleMax);
			fvs[0] = logOneMinusActBuffer[k];
			fvs[1] = logActBuffer[k];
			for (int t = 1; t < T; ++t)
			{
				int minN = max(0, numLabels - T + t);
				int maxN = min(t+2, numLabels);
				double* fvBegin = &fvs[t * numLabels];
				double* oldFvBegin = fvBegin - numLabels;
				double* acts = &logActBuffer[t * size];
				double* oneMinusActs = &logOneMinusActBuffer[t * size];
				for (int n = minN; n < maxN; ++n)
				{
					double fv = oldFvBegin[n] + oneMinusActs[k];
					if (n)
					{
						fv = logAdd(fv, oldFvBegin[n-1] + acts[k]);
					}
					fvBegin[n] = fv;
				}
			}
			kLogProb = fvs.back();//[((T-1) * numLabels) + labelCounts[k]];
			bvs.resize(T * numLabels);
			fill(bvs.begin(), bvs.end(), -doubleMax);
			bvs.back() = 0;
			for (int t = T - 2; t >= 0; --t)
			{
				int minN = max(0, numLabels - T + t);
				int maxN = min(t+2, numLabels);
				double* bvBegin = &bvs[t * numLabels];
				double* oldBvBegin = bvBegin + numLabels;
				double* oldActs = &logActBuffer[(t + 1) * size];
				double* oldOneMinusActs = &logOneMinusActBuffer[(t + 1) * size];
				for (int n = minN; n < maxN; ++n)
				{
					double bv = oldBvBegin[n] + oldOneMinusActs[k];
					if (n < (numLabels - 1))
					{
						bv = logAdd(bv, oldBvBegin[n+1] + oldActs[k]);
					}
					bvBegin[n] = bv;
				}
			}
		}
		else
		{
			kLogProb = 0;
			for (int t = 0; t < T; ++t)
			{
				kLogProb += logOneMinusActBuffer[(t * size) + k];
			}
		}
		logProb += kLogProb;

		//INJECT THE ERRORS
		if (kLogProb <= 0)
		{
			for (int t = 0; t < T; ++t)
			{
				double act = actBuffer[(t * size) + k];
				double* errIt = &errorBuffer[(t * size) + k];
				if (numLabels > 1)
				{
					//calculate dE/dY terms and rescaling factor
					double plusDEdYTerm = -doubleMax;
					double minusDEdYTerm = -doubleMax;
					const double* bvBegin = &bvs[t * numLabels];
					const double* fvBegin = &fvs[t * numLabels];					
					if (t == 0)
					{	
						if (labelCounts[k])
						{
							plusDEdYTerm = bvs[1];
						}
						minusDEdYTerm = bvs[0];
					}
					else
					{
						const double* oldFvBegin = &fvs[(t-1) * numLabels];
						for (int n = 0; n < numLabels; ++n)
						{
							double backVar = bvBegin[n];
							if (n)
							{
								plusDEdYTerm = logAdd(plusDEdYTerm, oldFvBegin[n-1] + backVar);
							}
							minusDEdYTerm = logAdd(minusDEdYTerm, oldFvBegin[n] + backVar);
						}
					}
					double normTerm = -doubleMax;
					for (int n = 0; n < numLabels; ++n)
					{
						normTerm = logAdd(normTerm, fvBegin[n] + bvBegin[n]);
					}

					//inject errors through logistic activation
					*errIt = act * (1 - act) * (safeExp(minusDEdYTerm - normTerm) - safeExp(plusDEdYTerm - normTerm));
#ifdef BLANK_UNIT
					double blankAct = actBuffer[(t * size) + blankUnit];
					*errIt *= blankAct;
					errorBuffer[(t * size) + blankUnit] += act * blankAct * (1 - blankAct) * (safeExp(minusDEdYTerm - normTerm) - safeExp(plusDEdYTerm - normTerm));
#endif
				}
				else
				{
					//inject errors through logistic activation
#ifdef BLANK_UNIT
					double blankAct = actBuffer[(t * size) + blankUnit];
					double prob = 1 - (blankAct * act);
					*errIt = (blankAct * act * (1 - act)) / prob;
					errorBuffer[(t * size) + blankUnit] += (act * blankAct * (1 - blankAct)) / prob;
#else
					*errIt = act;
#endif
				}
			}
		}
	}
	double labMlError = -logProb;
	if (labMlError >= 0)
	{
		//get most probable label seq
		vector<int> outputLabelCounts(size, 0);
		VDCI actIt = actBuffer.begin();
		for (int t = 0; t < T; ++t)
		{
#ifdef BLANK_UNIT
			for (int k = 0; k < blankUnit; ++k, ++actIt)
			{
				if ((*actIt * actBuffer[(t * size) + blankUnit]) > 0.5)
				{
					++outputLabelCounts[k];
				}
			}
#else
			for (int k = 0; k < size; ++k, ++actIt)
			{
				if (*actIt > 0.5)
				{
					++outputLabelCounts[k];
				}
			}
#endif
		}

#ifndef _WIN32
		if (sequenceDebugOutput)
		{
			//debug code
			cout << "output label counts:" << endl;
#ifdef BLANK_UNIT
			for (int k = 0; k < blankUnit; ++k)
#else
			for (int k = 0; k < size; ++k)
#endif
			{
				cout << outputLabelCounts[k] << " ";
			}
			cout << endl;
		}
#endif

		//calculate labelling errors
		int deletions = 0;
		int insertions = 0;
#ifdef BLANK_UNIT
		for (int k = 0; k < blankUnit; ++k)
#else
		for (int k = 0; k < size; ++k)
#endif
		{
			int err = outputLabelCounts[k] - labelCounts[k];
			if (err > 0)
			{
				insertions += err;
			}
			else if (err < 0)
			{
				deletions -= err;
			}
		}
		int labelErr = deletions + insertions;

		//store errors in map
		int totalLabels = accumulate(labelCounts.begin(), labelCounts.end(), 0);
        /* Freshly commented out.
		errorMap["insertions"] += make_pair<int,double>(totalLabels, insertions);
		errorMap["deletions"] += make_pair<int,double>(totalLabels, deletions);
		errorMap["labelErrorRate"] += make_pair<int,double>(totalLabels, labelErr);
		errorMap["seqErrorRate"] += make_pair<int,double>(1, (labelErr > 0));
		errorMap["labMlError"] += make_pair<int,double>(1, labMlError);
        */

		//TODO substitution confusion matrix, insertion and deletion lists
#ifndef _WIN32
		if (sequenceDebugOutput)
		{
			cout << "labMlError " <<  labMlError << endl;
			cout << "labelErrorRate " <<  labelErr << endl;
			cout << "insertions " <<  insertions << endl;
			cout << "deletions " <<  deletions << endl;
		}
#endif
		return labMlError;
	}
	else
	{
		return doubleMax;
	}
}
