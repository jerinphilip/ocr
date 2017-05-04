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
#include "RegressionOutputLayer.h"
#include "Typedefs.h"
#include "Helpers.h"


//global debug switch
extern bool sequenceDebugOutput;

RegressionOutputLayer::RegressionOutputLayer(const string& nam, int siz, vector<string>& criteria):
		name(nam),
		size(siz)
{
	criteria.push_back("rmsError");
}

void RegressionOutputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(actBuffer, path + name + "_activations", size, &seqDims);
	printBufferToFile(errorBuffer, path + name + "_errors", size, &seqDims);
}

void RegressionOutputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" RegressionOutputLayer size " << size << endl;
}

const string& RegressionOutputLayer::getName() const
{
	return name;
}

void RegressionOutputLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{
	*errBegin = &errorBuffer[offset * size];
	*errEnd = *errBegin + size;
}

void RegressionOutputLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &actBuffer[offset * size];
	*actEnd = *actBegin + size;
}

int RegressionOutputLayer::inputSize() const
{
	return size;
}

int RegressionOutputLayer::outputSize() const
{
	return size;
}

void RegressionOutputLayer::updateDerivs(int offset) 
{
	const double* errBegin = 0;
	const double* errEnd = 0;
	getErrors(&errBegin, &errEnd, offset);
	InputConnsImpl::updateDerivs(errBegin, errEnd, offset);
}

//NB assumes actBuffer already filled in
void RegressionOutputLayer::setErrToOutputDeriv(int timestep, int outputNum)
{
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	int offset = (timestep * size) + outputNum;
	errorBuffer[offset] = actBuffer[offset];
}

//TODO: check that seq.targetPatterns.size() == actBuffer.size() / size
double RegressionOutputLayer::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	VDCI actBegin = actBuffer.begin();
	VDI errBegin = errorBuffer.begin();
	const float* targBegin = seq.targetPatterns;
	double rmsErr = 0;
	for (int i = 0; i < seq.size; ++i, actBegin += size, errBegin += size, targBegin += size)
	{
#ifdef MIN_TARG_VAL
		if (targBegin && *targBegin > MIN_TARG_VAL)
#else
		if (targBegin)
#endif
		{
			//inject training error
			double sumErr = 0;
			VDCI actEnd = actBegin + size;
			VDI errIt = errBegin;
			const float* targIt = targBegin;
			for (VDCI actIt = actBegin; actIt != actEnd; ++actIt, ++errIt, ++targIt)
			{
				double err = *actIt - *targIt;
				sumErr += err * err;
				*errIt = err;
			}
			rmsErr += sumErr;
		}
		else
		{
			fill(errBegin, errBegin + size, 0);
		}
	}
	double sumSquaresErr = 0.5 * rmsErr;

#ifndef _WIN32
		if (sequenceDebugOutput)
		{
			cout << "sumSquaresErr " <<  sumSquaresErr << endl;
		}
#endif

	//store errors in map
    /* Freshly commented out.
	errorMap["sumSquaresError"] += make_pair<int, double>(1, sumSquaresErr);
	errorMap["rmsError"] += make_pair<int, double>(0, rmsErr);
    */
	return sumSquaresErr;
}

void RegressionOutputLayer::feedForward(const vector<int>& dims, int seqLength, const DataSequence& seq)
{
	seqDims = dims;
	actBuffer.resize(seqLength * size);
	errorBuffer.resize(seqLength * size);
	fill(actBuffer.begin(), actBuffer.end(), 0);
	double* actBegin = &actBuffer.front();
	for (int i = 0; i < seqLength; ++i, actBegin += size)
	{
		for (VPBCCI it = inputs.begin(); it != inputs.end(); ++it)
		{
			(*it)->feedForward(actBegin, actBegin + size, i);
		}
	}
}

