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

#ifndef _INCLUDED_NDimSubnet_h  
#define _INCLUDED_NDimSubnet_h  

#include "NDimLayer.h"
#include "OutputLayer.h"
#include "BlockTransformer.h"
#include "BlockLayer.h"
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/util/XMLString.hpp>

XERCES_CPP_NAMESPACE_USE


class NDimSubnet
{

private:

	//data
	vector<int> flatSeqDims;
	vector<int> dimProducts;
	int inputOffset;
	vector<int> coords;
#ifndef DELAY_IN_CONN
	vector<int> actSteps;
#endif
	int flatSeqLength;

	//functions
	void feedBackR(int d, const vector<int>& seqDims, const vector<int>& dimProducts, int inputOverlap);
	void feedForwardR(int d, const vector<int>& seqDims, const vector<int>& dimProducts, int inputOverlap);
	void initCoords(bool backwards, const vector<int>& seqDims);
	int calcOffset(const vector<int>& dimProducts) const;


public:

	//data
	NDimLayer* layer; 
	BlockLayer* blockLayer; 
	
	//TODO make this private again
	vector<int> actDirections;
	
	//functions
	NDimSubnet(const string& name, int numDims, int configNumber, const DOMNode* layerNode, BlockTransformer* blockTransformer);
	~NDimSubnet();
	void feedForward(const vector<int>& seqDims, const vector<int>& dimProducts, BlockTransformer* blockTransformer, int inputOverlap);
	void feedBack(const vector<int>& seqDims, const vector<int>& dimProducts, BlockTransformer* blockTransformer, int inputOverlap);

};

#endif
