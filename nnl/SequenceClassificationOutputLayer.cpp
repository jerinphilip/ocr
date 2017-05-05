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
#include "SequenceClassificationOutputLayer.h"
#include "NnMath.h"
#include "Typedefs.h"
#include "Helpers.h"

//global debug switch
//bool sequenceDebugOutput=false;;

SequenceClassificationOutputLayer::SequenceClassificationOutputLayer(const string& nam, int numClasses, const vector<string>& lab, vector<string>& criteria):
		labels(lab),
		name(nam),
		size(numClasses)
{
	targetVects.resize(size);
	acts.resize(size);
	for (VVDI it = targetVects.begin(); it != targetVects.end(); ++it)
	{
		it->resize(size, 0);
		it->at(distance(targetVects.begin(), it)) = 1;
	}
	criteria.push_back("classificationErrorRate");
	criteria.push_back("crossEntropyError");
}

void SequenceClassificationOutputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(inActBuffer, path + name + "_input_activations", size, 0, &labels);
	printBufferToFile(acts, path + name + "_activations", size, 0, &labels);
	printBufferToFile(errorBuffer, path + name + "_errors", size, 0, &labels);
}

void SequenceClassificationOutputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" SequenceClassificationOutputLayer size " << size << endl;
}

const string& SequenceClassificationOutputLayer::getName() const
{
	return name;
}

void SequenceClassificationOutputLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{
	*errBegin = &errorBuffer[offset * size];
	*errEnd = *errBegin + size;
}

//only one act per sequence
void SequenceClassificationOutputLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &acts.front();
	*actEnd = &acts.back();
}

int SequenceClassificationOutputLayer::inputSize() const
{
	return size;
}

int SequenceClassificationOutputLayer::outputSize() const
{
	return size;
}

void SequenceClassificationOutputLayer::updateDerivs(int offset) 
{
	const double* errBegin = 0;
	const double* errEnd = 0;
	getErrors(&errBegin, &errEnd, offset);
	InputConnsImpl::updateDerivs(errBegin, errEnd, offset);
}

//NB assumes acts already filled in
void SequenceClassificationOutputLayer::setErrToOutputDeriv(int timestep, int outputNum)
{
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	double outputAct = acts[outputNum];
	for (VDI errIt = errorBuffer.begin(); errIt != errorBuffer.end();)
	{
		for (VDI actIt = acts.begin(); actIt != acts.end(); ++actIt, ++errIt)
		if (distance(acts.begin(), actIt) == outputNum)
		{
			*errIt = *actIt * (1 - *actIt);
		}
		else
		{
			*errIt = -(outputAct * *actIt);
		}
	}
}

double SequenceClassificationOutputLayer::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	if (seq.labelCounts.size() != acts.size())
	{
		cerr << "ERROR: SequenceClassificationOutputLayer::injectSequenceErrors seq.labelCounts.size() ";
		cerr << (int)seq.labelCounts.size() << " != acts.size() " << (int)acts.size() << ", exiting" << endl;
		exit(0);
	}
	int targetClass = (int)distance(seq.labelCounts.begin(), find_if(seq.labelCounts.begin(), seq.labelCounts.end(), bind2nd(greater<int>(), 0)));
	int outputClass = (int)distance(acts.begin(), max_element(acts.begin(), acts.begin() + size));
	bool correct;
	double crossEntropyErr = 0;
	if (targetClass >= 0 && targetClass < (int)seq.labelCounts.size())	
	{
		//inject training error		
		const vector<double>& targetVect = targetVects[targetClass];
		VDCI actIt = acts.begin();
		VDI errBegin = errorBuffer.begin();
		for (VDCI targIt = targetVect.begin(); targIt != targetVect.end(); ++actIt, ++errBegin, ++targIt)
		{
			double targ = *targIt;
			double act = *actIt;
			double injErr = act - targ;
			for (VDI errIt = errBegin; errIt < errorBuffer.end(); errIt += size)
			{
				*errIt = injErr;
			}
			if (targ)
			{
				crossEntropyErr -= targ * safeLog(act / targ);
			}
		}

		//TODO classification confusion matrix
		correct = (outputClass == targetClass);
	}
	else
	{
		targetClass = -1;
		fill(errorBuffer.begin(), errorBuffer.end(), 0);
		correct = true;
	}

#ifndef _WIN32
    /*
	if (sequenceDebugOutput)
	{
		cout << "crossEntropyError " <<  crossEntropyErr << endl;
		cout << "target class " <<  targetClass << endl;
		cout << "output class " <<  outputClass << endl;
		cout << "correct " <<  correct << endl;
	}
    */
#endif

    /*  Freshly commented out
	errorMap["crossEntropyError"] += make_pair<int,double>(1, crossEntropyErr);
	errorMap["classificationErrorRate"] += make_pair<int,double>(1, !correct);
	if (targetClass >= 0)
	{
		errorMap["\t" + labels[targetClass] + "-errors"] += make_pair<int,double>(1, !correct);
	}
    */
	return crossEntropyErr;
}

void SequenceClassificationOutputLayer::feedForward(const vector<int>& dims, int seqLength, const DataSequence& seq)
{
	seqDims = dims;
	errorBuffer.resize(seqLength * size);
	inActBuffer.resize(errorBuffer.size());
	fill(inActBuffer.begin(), inActBuffer.end(), 0);
	fill(acts.begin(), acts.end(), 0);
	double* inActBegin = &inActBuffer.front();
	for (int i = 0; i < seqLength; ++i, inActBegin += size)
	{
		//gather acts over n-1 dimensional slice
		double* inActEnd = inActBegin + size;
		for (VPBCCI it = inputs.begin(); it != inputs.end(); ++it)
		{
			(*it)->feedForward(inActBegin, inActEnd, i);
		}
		transform(inActBegin, inActEnd, acts.begin(), acts.begin(), plus<double>());
	}

	//center acts on 0 for safer exponentiation
	double maxAct = *max_element(acts.begin(), acts.end());
	double minAct = *min_element(acts.begin(), acts.end());
	transform(acts.begin(), acts.end(), acts.begin(), bind2nd(minus<double>(), maxAct + minAct / (double) 2));

	//apply softmax activation
	transform(acts.begin(), acts.end(), acts.begin(), bdedExp);
	double expSum = accumulate(acts.begin(), acts.end(), 0.0);
	transform(acts.begin(), acts.end(), acts.begin(), bind2nd(divides<double>(), expSum));
}
