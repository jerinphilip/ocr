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

#include <math.h>
#include "PredictionOutputLayer.h"
#include "Typedefs.h"

#ifdef DELAY_IN_CONN

//global debug switch
//bool sequenceDebugOutput=false;;


PredictionOutputLayer::PredictionOutputLayer(const string& nam, vector<string>& criteria, Layer* inputLay, bool delCumActs, PredictionOutputLayer* firstPredLayer):
		RegressionOutputLayer(nam, inputLay->outputSize(), criteria),
		inputLayer(inputLay),
		firstPredictionLayer(firstPredLayer),
		delCumulativeActs(delCumActs)
{
	targetMean.resize(size);
	criteria.push_back("totalError");
	cout << delCumulativeActs << endl;
}

PredictionOutputLayer::~PredictionOutputLayer()
{
}

void PredictionOutputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" PredictionOutputLayer size " << size << endl;
}

void PredictionOutputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(cumulativeActs, path + name + "_cumulative_activations", size, &seqDims);
	RegressionOutputLayer::outputInternalVariables(path);
}

void PredictionOutputLayer::updateDerivs(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts) 
{
	int offset = inner_product(coords.begin(), coords.end(), dimProducts.begin(), 0);
	const double* errBegin = &errorBuffer[offset * size];
	const double* errEnd = errBegin + size;
	for (VPBCCI inputIt = inputs.begin(); inputIt != inputs.end(); ++inputIt)
	{
		(*inputIt)->updateDerivs(errBegin, errEnd, coords, seqDims, dimProducts);
	}
}

void PredictionOutputLayer::resizeBuffers()
{
	int size = inputLayer->getActBuffer().size();
	actBuffer.resize(size);
	errorBuffer.resize(size);
}

void PredictionOutputLayer::feedForward(const vector<int>& coords, const vector<int>& dims, const vector<int>& dimProducts)
{
	seqDims = dims;
	int offset = inner_product(coords.begin(), coords.end(), dimProducts.begin(), 0);
	double* actBegin = &actBuffer[offset * size];
	double* actEnd = actBegin + size;
	fill(actBegin, actEnd, 0);

	//feed forward inputs from the net
	for (VPBCCI inputIt = inputs.begin(); inputIt != inputs.end(); ++inputIt)
	{
		(*inputIt)->feedForward(actBegin, actEnd, coords, seqDims, dimProducts);
	}
}

void PredictionOutputLayer::feedBack(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts)
{
	int offset = inner_product(coords.begin(), coords.end(), dimProducts.begin(), 0);
	double* errBegin = &errorBuffer[offset * size];
	double* errEnd = errBegin + size;

	//feed back errors from the Net
	for (VPBCCI outputIt = outputs.begin(); outputIt != outputs.end(); ++outputIt)
	{
		(*outputIt)->feedBack(errBegin, errEnd, coords, seqDims, dimProducts);
	}
}

//TODO check that seq.targetPatterns.size() == actBuffer.size() / size
double PredictionOutputLayer::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	//for error prediction layers, subtract from prev cumulative acts to get cumulative error
	VDCI targBeginIt;
	VDCI cumTargBeginIt;
	cumulativeActs.resize(actBuffer.size());
	if (firstPredictionLayer)
	{
		const vector<double>& cumulativeInputs = inputLayer->getCumulativePredictionActs();
		if (delCumulativeActs)
		{
			transform(cumulativeInputs.begin(), cumulativeInputs.end(), actBuffer.begin(), cumulativeActs.begin(), minus<double>());
		}
		else
		{
			transform(cumulativeInputs.begin(), cumulativeInputs.end(), actBuffer.begin(), cumulativeActs.begin(), plus<double>());
		}
		targBeginIt = inputLayer->getErrorBuffer().begin();
		cumTargBeginIt = firstPredictionLayer->inputLayer->getActBuffer().begin();
	}
	else
	{
		copy(actBuffer.begin(), actBuffer.end(), cumulativeActs.begin());
		targBeginIt = inputLayer->getActBuffer().begin();
		cumTargBeginIt = targBeginIt;
	}
	const vector<double>& cumTargetMean = firstPredictionLayer ? firstPredictionLayer->targetMean : targetMean;
	VDCI cumTargIt = cumTargBeginIt;
	VDCI targIt = targBeginIt;
	VDCI actIt = actBuffer.begin();
	VDI errIt = errorBuffer.begin();
	VDCI cumActIt = cumulativeActs.begin();
	double rmsErr = 0;
	double cumRmsErr = 0;
	int seqLength = actBuffer.size() / size;
	fill (targetMean.begin(), targetMean.end(), 0);
	for (int i = 0; i < seqLength; ++i)
	{
		//inject training error
		for (int j = 0; j < size; ++j, ++targIt, ++actIt, ++errIt, ++cumActIt, ++cumTargIt)
		{
			targetMean[j] += *targIt;
			double err = *actIt - *targIt;
			double cumErr = *cumActIt - *cumTargIt;
			cumRmsErr += cumErr * cumErr;
			rmsErr += err * err;
			*errIt = err;
		}
	}
	double sumSquaresErr = 0.5 * rmsErr;
	double cumSumSquaresErr = 0.5 * cumRmsErr;
	
	//TODO proper rms normalization over whole data set
	transform(targetMean.begin(), targetMean.end(), targetMean.begin(), bind2nd(divides<double>(), seqLength));
	double rmsDivisor = 0;
	double cumRmsDivisor = 0;
	targIt = targBeginIt;
	cumTargIt = cumTargBeginIt;
	for (int i = 0; i < seqLength; ++i)
	{
		for (int j = 0; j < size; ++j, ++targIt, ++cumTargIt)
		{
			rmsDivisor += pow(*targIt - targetMean[j], (int)2);
			cumRmsDivisor += pow(*cumTargIt - cumTargetMean[j], (int)2);
		}
	}
	rmsErr /= rmsDivisor;
	cumRmsErr /= cumRmsDivisor;
	
	//add term to input errors to account for providing targets to this layers
	if (firstPredictionLayer)
	{
		transform(inputLayer->getErrorBuffer().begin(), inputLayer->getErrorBuffer().end(), 
			  errorBuffer.begin(), inputLayer->getErrorBuffer().begin(), minus<double>());
	}
	else
	{
		transform(errorBuffer.begin(), errorBuffer.end(), inputLayer->getErrorBuffer().begin(), inputLayer->getErrorBuffer().begin(), plus<double>());
	}
	
#ifndef _WIN32
    /*
	if (sequenceDebugOutput)
	{
		cout << "sumSquaresErr " <<  sumSquaresErr << endl;
		cout << "rmsErr " <<  rmsErr << endl;
		if (firstPredictionLayer)
		{
			cout << "cumulativeSumSquaresErr " <<  cumSumSquaresErr << endl;
			cout << "cumulativeRmsErr " <<  cumRmsErr << endl;
		}
	}
    */
#endif

	//store errors in map
	errorMap[name + "_sumSquaresError"] += make_pair<int, double>(1, sumSquaresErr);
	errorMap[name + "_rmsError"] += make_pair<int, double>(1, rmsErr);
	if (firstPredictionLayer)
	{
		errorMap[name + "_cumulativeSumSquaresErr"] += make_pair<int, double>(1, cumSumSquaresErr);
		errorMap[name + "_cumulativeRmsErr"] += make_pair<int, double>(1, cumRmsErr);
	}
	return sumSquaresErr;
}

#endif DELAY_IN_CONN
