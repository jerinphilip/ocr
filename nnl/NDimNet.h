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

#ifndef _INCLUDED_NDimNet_h  
#define _INCLUDED_NDimNet_h  

#include <map>
#include <vector>
#include "Net.h"
#include "InputLayer.h"
#include "BiasLayer.h"
#include "NDimLayer.h"
#include "OutputLayer.h"
#include "NDimLevel.h"
#include "DataSequence.h"


class NDimNet:
		public Net
{


public:
	//data
	DOMElement* netNode;
	string inputBlockDimString;
	bool multidirectional;
	int numDims;
	int inputOverlap;
	vector<NDimLevel*> levels;
	vector<int> dimProducts;
	InputLayer* inputLayer;
	OutputLayer* outputLayer;
	string task;
	BiasLayer bias;

	//functions
	void feedBack(const DataSequence& seq);
	void feedForward(const DataSequence& seq);
	double injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq);
	
#ifdef DELAY_IN_CONN
	void updateOutputDerivsR(int d, vector<int> outputCoords, const vector<int>& seqDims, const vector<int>& dimProducts);
	void feedForwardOutputR(int d, vector<int> outputCoords, const vector<int>& seqDims, const vector<int>& dimProducts);
#endif
	
public:

	//construction / destruction
	NET_API NDimNet(const DOMElement* element, const DataDimensions& dims, vector<string>& criteria);
	NET_API virtual ~NDimNet();

	//net operation functions
	double calculateGradient(map<const string, pair<int,double> >& errorMap, const DataSequence& seq);
	double calculateError(map<const string, pair<int,double> >& errorMap, const DataSequence& seq);

	//io functions
	void save(const string& indent, ostream&out=cout) const;
	void print(ostream& out = cout) const;
	void build();
	void outputInternalVariables(const string& path) const;
	void outputJacobian(const DataSequence& seq, const string& path, int timestep, int output);
	void printOutputs(ostream& out = cout);
};

typedef vector<NDimLevel*>::iterator VPNDLI;
typedef vector<NDimLevel*>::reverse_iterator VPNDLRI;
typedef vector<NDimLevel*>::const_reverse_iterator VPNDLCRI;
typedef vector<NDimLevel*>::const_iterator VPNDLCI;

#endif
