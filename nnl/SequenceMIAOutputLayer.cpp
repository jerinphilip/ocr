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
#include "SequenceMIAOutputLayer.h"
#include "NnMath.h"
#include "ActivationFunctions.h"
#include "Typedefs.h"
#include <iomanip>


//global debug switch
extern bool sequenceDebugOutput;

SequenceMIAOutputLayer::SequenceMIAOutputLayer(const string& nam, const vector<string>& lab, vector<string>& criteria):
		name(nam),
		labels(lab),
		size((int)lab.size())
{
	criteria.push_back("miaError");
	criteria.push_back("featureErrorRate");
	featureInsertions.resize(size);
	featureDeletions.resize(size);	
	acts.resize(size);
}

SequenceMIAOutputLayer::~SequenceMIAOutputLayer(void)
{}

void SequenceMIAOutputLayer::feedForward(const vector<int>& dims, int seqLength, const DataSequence& seq)
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
	transform(acts.begin(), acts.end(), acts.begin(), Logistic::fn);
}

void SequenceMIAOutputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" sequence mia output layer size " << size << endl;
}

void SequenceMIAOutputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(acts, path + name + "_activations", size, 0, &labels);
	printBufferToFile(errorBuffer, path + name + "_errors", size, 0, &labels);
	printBufferToFile(inActBuffer, path + name + "_input_activations", size, 0, &labels);
}

const string& SequenceMIAOutputLayer::getName() const
{
	return name;
}

void SequenceMIAOutputLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{
	*errBegin = &errorBuffer[offset * size];
	*errEnd = *errBegin + size;
}

//only one act per sequence
void SequenceMIAOutputLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &acts.front();
	*actEnd = &acts.back();
}

int SequenceMIAOutputLayer::inputSize() const
{
	return size;
}

int SequenceMIAOutputLayer::outputSize() const
{
	return size;
}

void SequenceMIAOutputLayer::updateDerivs(int offset)
{
	const double* errBegin = 0;
	const double* errEnd = 0;
	getErrors(&errBegin, &errEnd, offset);
	InputConnsImpl::updateDerivs(errBegin, errEnd, offset);
}

//NB assumes acts already filled in
void SequenceMIAOutputLayer::setErrToOutputDeriv(int timestep, int outputNum)
{
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	double outputAct = acts[outputNum];
	double deriv = outputAct * (1 - outputAct);
	for (VDI errIt = errorBuffer.begin(); errIt != errorBuffer.end(); errIt += size)
	{
		errIt[outputNum] = deriv;
	}
}

double SequenceMIAOutputLayer::injectSequenceErrors(map<const string, pair<int, double> > & errorMap, const DataSequence& seq)
{
	if (seq.labelCounts.size() != acts.size())
	{
		cerr << "ERROR: SequenceMIAOutputLayer::injectSequenceErrors seq.labelCounts.size() ";
		cerr << (int)seq.labelCounts.size() << " != acts.size() " << (int)acts.size() << ", exiting" << endl;
		exit(0);
	}

	//inject training error
	double seqErr = 0;
	int targFeaturesOn = 0;
	VDCI actIt = acts.begin();
	VDI errBegin = errorBuffer.begin();
	fill(featureInsertions.begin(), featureInsertions.end(), 0);
	fill(featureDeletions.begin(), featureDeletions.end(), 0);
	VII featInsIt = featureInsertions.begin();
	VII featDelIt = featureDeletions.begin();
	for (VICI labCountIt = seq.labelCounts.begin(); labCountIt != seq.labelCounts.end(); ++actIt, ++labCountIt, ++errBegin, ++featInsIt, ++featDelIt)
	{
		int targ = *labCountIt > 0 ? 1 : 0;
		double act = *actIt;
		double injErr =  act - targ;
		seqErr -= targ ? safeLog(act) : safeLog(1 - act);
		for (VDI errIt = errBegin; errIt < errorBuffer.end(); errIt += size)
		{
			*errIt = injErr;
		}
		if (targ)
		{
			++targFeaturesOn;
			if (act < 0.5)
			{
				++(*featDelIt);
			}
		}
		else if (act > 0.5)
		{
			++(*featInsIt);
		}
	}

	//store errors in map
	int insertions = accumulate(featureInsertions.begin(), featureInsertions.end(), 0);
	int deletions = accumulate(featureDeletions.begin(), featureDeletions.end(), 0);
    /* Freshly commented out
	errorMap["miaError"] += make_pair<int, double>(1, seqErr);
	errorMap["featureErrorRate"] += make_pair<int, double>(1, insertions + deletions);
	errorMap["insertions"] += make_pair<int, double>(size - targFeaturesOn, insertions);
	errorMap["deletions"] += make_pair<int, double>(targFeaturesOn, deletions);
    */
	
	for (int i = 0; i < size; ++i)
	{
		if (seq.labelCounts[i] > 0)
		{
			errorMap["\t" + labels[i] + "-deletions"] += make_pair<int,double>(1, featureDeletions[i]);
		}
		else
		{
			errorMap["\t" + labels[i] + "-insertions"] += make_pair<int,double>(1, featureInsertions[i]);
		}
	}
	
#ifndef _WIN32
	if (sequenceDebugOutput)
	{
		//debug code
		cout << "label       target  output  output act" << endl;
		cout << "-----       ------  ------  ----------" << endl;
		for (int i = 0; i < (int)labels.size(); ++i)
		{
			if (labels[i].size() > 10)
			{
				for (int j = 0; j < 9; ++j)
				{
					cout << labels[i].c_str()[j];
				}
				cout << "~  ";
			}
			else
			{
				cout << setw(10) << labels[i] << "  ";
			}
			if (seq.labelCounts[i])
			{
				cout << setw(6) << seq.labelCounts[i] << "  ";
			}
			else
			{
				cout << "        ";
			}
			if (acts[i] > 0.5)
			{
				cout << "*       ";
			}
			else
			{
				cout << "        ";
			}
			cout << acts[i] << endl;
		}
		cout << "miaError " << seqErr << endl;
		cout << "featureErrors " << insertions + deletions << endl;
	}
#endif
	return seqErr;
}
