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

#ifndef _INCLUDED_BlockLayer_h  
#define _INCLUDED_BlockLayer_h  

#include "Layer.h"
#include "OutputConnsImpl.h"
#include "BlockTransformer.h"


class BlockLayer:
		public Layer,
		protected OutputConnsImpl
{

private:

	string name;
	int size;
	vector<double> actBuffer;
	vector<double> errorBuffer;
	BlockTransformer* blockTransformer;

public:
	
	//data
	//HACK to give access to buffers to block transformer
	//TODO get rid of these


	//defined functions
// 	BlockLayer(const string& name, int size);
	BlockLayer(const string& name, BlockTransformer* bt);
	void outputInternalVariables(const string& path) const;
	int outputSize() const;
	void getActs(const double** actBegin, const double** actEnd, int offset) const;
	int inputSize() const;
	void getErrors(const double** errBegin, const double** errEnd, int offset) const;
	const string& getName() const;
// 	void feedBack(int offset);
	void feedBack(vector<double>& layerErrors);
	void feedForward(vector<double>& layerActs, const vector<int>& seqDims);
	void resizeErrorBuffer(int buffSize);
	vector<double>& getActBuffer() {return actBuffer;}
	vector<double>& getErrorBuffer() {return errorBuffer;}
	
	//forwarded functions
	void addOutputConn(Connection* conn) {OutputConnsImpl::addOutputConn(conn);}

	//null functions
	void addInputConn(Connection* conn) {}
	void print(ostream& out) const {}
	
};

#endif
