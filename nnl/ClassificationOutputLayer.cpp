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
#include "ClassificationOutputLayer.h"
#include "NnMath.h"
#include "Typedefs.h"
#include "Helpers.h"

//global debug switch
extern bool sequenceDebugOutput;

ClassificationOutputLayer::ClassificationOutputLayer(const string& nam, int numClasses, const vector<string>& lab, vector<string>& criteria):
		labels(lab),
		name(nam),
		size(numClasses)
{
	exponentials.resize(size);
	targetVects.resize(size);
	for (VVDI it = targetVects.begin(); it != targetVects.end(); ++it)
	{
		it->resize(size, 0);
		it->at(distance(targetVects.begin(), it)) = 1;
	}
	criteria.push_back("classificationErrorRate");
	criteria.push_back("crossEntropyError");
	classErrors.resize(size);
	classTargets.resize(size);
}

void ClassificationOutputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(actBuffer, path + name + "_activations", size, &seqDims, &labels);
	printBufferToFile(errorBuffer, path + name + "_errors", size, &seqDims, &labels);
	printBufferToFile(outputClasses, path + name + "_classes", 1, &seqDims, &labels);
}

void ClassificationOutputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" ClassificationOutputLayer size " << size << endl;
}

const string& ClassificationOutputLayer::getName() const
{
	return name;
}

void ClassificationOutputLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{
	*errBegin = &errorBuffer[offset * size];
	*errEnd = *errBegin + size;
}

void ClassificationOutputLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &actBuffer[offset * size];
	*actEnd = *actBegin + size;
}

int ClassificationOutputLayer::inputSize() const
{
	return size;
}

int ClassificationOutputLayer::outputSize() const
{
	return size;
}

void ClassificationOutputLayer::updateDerivs(int offset) 
{
	const double* errBegin = 0;
	const double* errEnd = 0;
	getErrors(&errBegin, &errEnd, offset);
	InputConnsImpl::updateDerivs(errBegin, errEnd, offset);
}


//NB assumes actBuffer already filled in
void ClassificationOutputLayer::setErrToOutputDeriv(int timestep, int outputNum)
{
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	int initOffset = timestep * size;
	int endOffset = initOffset + size;
	int outputOffset = initOffset + outputNum;
	double outputAct = actBuffer[outputOffset];
	for (int offset = initOffset; offset < endOffset; ++offset)
	{
		if (offset - initOffset == outputNum)
		{
			errorBuffer[offset] = outputAct * (1 - outputAct);
		}
		else
		{
			errorBuffer[offset] = - (outputAct * actBuffer[offset]);
		}
	}
}

//TODO: check that seq.targetClasses.size() == actBuffer.size() / size
double ClassificationOutputLayer::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	if (seq.size != actBuffer.size() / size)
	{
		cerr <<  "ERROR ClassificationOutputLayer::injectSequenceErrors seq.size ";
		cerr << seq.size << " !=  actBuffer.size() / size = " << actBuffer.size() / size << ", exiting" << endl;
		exit(0);
	}
	fill (classErrors.begin(), classErrors.end(), 0);
	fill (classTargets.begin(), classTargets.end(), 0);
	double seqErr = 0;
	int frameErrors = 0;
	int frames = 0;
	VDI errBegin = errorBuffer.begin();
	VICI targClassIt = seq.targetClasses.begin();
	outputClasses.clear();
	for (VDCI actBegin = actBuffer.begin(); actBegin != actBuffer.end(); 
			++targClassIt, actBegin += size, errBegin += size)
	{
		int outputClass = (int)distance(actBegin, max_element(actBegin, actBegin + size));
		outputClasses.push_back(outputClass);
		int classNum = *targClassIt;
		if (classNum >= 0 && classNum < size)
		{
			//inject training error		
			const vector<double>& targetVect = targetVects[classNum];
			double crossEntropyErr = 0;
			VDI errIt = errBegin;
			VDCI actIt = actBegin;
			for (VDCI targIt = targetVect.begin(); targIt != targetVect.end(); ++actIt, ++errIt, ++targIt)
			{
				double targ = *targIt;
				double act = *actIt;
				double err = act - targ;
				*errIt = err;
				if (targ)
				{
					crossEntropyErr -= targ * safeLog(act / targ);
				}
			}
			//TODO classification confusion matrix
			bool frameCorrect = (outputClass == classNum);
			seqErr += crossEntropyErr;
			frameErrors += !frameCorrect;
			++frames;
			++classTargets[classNum];
			classErrors[classNum] += !frameCorrect;
		}
		else
		{
			fill(errBegin, errBegin + size, 0);
		}
	}

//#define MNIST_SEG_HACK
#ifdef MNIST_SEG_HACK
	//HACK for mnist seg experiments
	vector<double> cumActs(size - 1, 0);
	VDCI actBuffIt = actBuffer.begin();
	int timesteps = actBuffer.size() / size;
	for (int i = 0; i < timesteps; ++i, actBuffIt += size)
	{
		for (int j = 0; j < size - 1; ++j)
		{
			cumActs[j] += actBuffIt[j];
		}
	}
	int mnistSegClass = (int)distance(cumActs.begin(), max_element(cumActs.begin(), cumActs.end()));
	int mnistTargClass = atoi(seq.targetString.c_str());
	errorMap["mnistSegErr"] += make_pair<int,double>(1, (mnistSegClass != mnistTargClass));
#endif

#ifndef _WIN32
	if (sequenceDebugOutput)
	{
		cout << "crossEntropyError " <<  seqErr << endl;
		cout << "num classifications " << frames << endl;
		cout << "classification errors " <<  frameErrors << endl;
		cout << "class error rates:" << endl;
		for (int i = 0; i < size; ++i)
		{
			if (classTargets[i])
			{
				cout << labels[i] << ' ' << classErrors[i] << '/' << classTargets[i];
				cout << " = " << (double)classErrors[i] / classTargets[i] << endl;
			}
		}

		//HACK for mnist seg experiments
#ifdef MNIST_SEG_HACK
		cout << "MNIST CUMULATIVE ACTS:" << endl;
		copy (cumActs.begin(), cumActs.end(), ostream_iterator<double>(cout, " "));
		cout << endl;
		cout << "MNIST SEG CLASS " << mnistSegClass << endl;
		cout << "MNIST TARG CLASS " << mnistTargClass << endl;
#endif
	}
#endif

    /* Freshly commented out.
	errorMap["crossEntropyError"] += make_pair<int,double>(1, seqErr);
	errorMap["classificationErrorRate"] += make_pair<int,double>(frames, frameErrors);
	for (int i = 0; i < size; ++i)
	{
		errorMap["\t" + labels[i] + "-errors"] += make_pair<int,double>(classTargets[i], classErrors[i]);
	}
    */
	return seqErr;
}

void ClassificationOutputLayer::feedForward(const vector<int>& dims, int seqLength, const DataSequence& seq)
{
	seqDims = dims;
	actBuffer.resize(seqLength * size);
	errorBuffer.resize(seqLength * size);
	fill(actBuffer.begin(), actBuffer.end(), 0);
	double* actBegin = &actBuffer.front();
	for (int i = 0; i < seqLength; ++i)
	{
		double* actEnd = actBegin + size;
		for (VPBCCI it = inputs.begin(); it != inputs.end(); ++it)
		{
			(*it)->feedForward(actBegin, actEnd, i);
		}

		//center acts on 0 for safer exponentiation
		double maxAct = *max_element(actBegin, actEnd);
		double minAct = *min_element(actBegin, actEnd);
		transform(actBegin, actEnd, actBegin, bind2nd(minus<double>(), maxAct + minAct / (double) 2));

		//apply softmax activation
		transform(actBegin, actEnd, actBegin, bdedExp);
		double expSum = accumulate(actBegin, actEnd, 0.0);
		transform(actBegin, actEnd, actBegin, bind2nd(divides<double>(), expSum));
		actBegin = actEnd;
	}
}
