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

#ifndef _INCLUDED_InputLayer_h  
#define _INCLUDED_InputLayer_h  

#include "Layer.h"
#include "OutputConnsImpl.h"
#include "DataSequence.h"
#include "Typedefs.h"
#include "BlockTransformer.h"

class InputLayer:
		public Layer,
		protected OutputConnsImpl
{

private:

	//data
	int inSize;
	int outSize;
	int seqLength;
	string name;
	vector<double> inActBuffer;
	vector<double> actBuffer;
	vector<double> errorBuffer;
	vector<double> inErrBuffer;
	vector<int> dataSeqDims;
#ifdef ZERO_INPUT_CHECK
	vector<bool> activeOffsets;
#endif

public:

	//data
	BlockTransformer* blockTransformer;
	
	//defined functions
	InputLayer(const string& name, int size, const string& blockString);
	~InputLayer(void);
#ifdef ZERO_INPUT_CHECK
	bool active(int offset){return activeOffsets[offset];}
#endif
	void copyInputs (const DataSequence& seq);
	int outputSize() const;
	void getActs(const double** actBegin, const double** actEnd, int offset) const;
	void getErrors(const double** errBegin, const double** errEnd, int offset) const;
	int inputSize() const;
	void feedBack();
	const string& getName() const;
	void print(ostream& out) const;
	void outputInternalVariables(const string& path) const;
	const vector<int>& getSeqDims() const;
	int getSeqLength() const;
	vector<double>& getActBuffer() {return actBuffer;}
	vector<double>& getErrorBuffer() {return errorBuffer;}
	
	//forwarded functions
	void addOutputConn(Connection* conn) {OutputConnsImpl::addOutputConn(conn);}

	//null functions
	void addInputConn(Connection* conn){};

};

#endif
