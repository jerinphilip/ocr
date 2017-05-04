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

#ifndef _INCLUDED_OutputLayer_h  
#define _INCLUDED_OutputLayer_h  

#include <iostream>
#include <fstream>
#include <map>
#include "Layer.h"
#include "DataSequence.h"

class OutputLayer:
		public Layer
{

public:

	virtual double injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq) = 0;
	virtual void feedForward(const vector<int>& seqDims, int seqLength, const DataSequence& seq) = 0;
	virtual	void build() = 0;
	virtual void printInputConns(ostream& out) const = 0;
	virtual void updateDerivs(int offset) = 0;
	virtual void setErrToOutputDeriv(int timestep, int output) = 0;
	virtual void printOutputs(ostream& out = cout) = 0;

};

#endif
