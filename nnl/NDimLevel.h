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

#ifndef _INCLUDED_NDimLevel_h
#define _INCLUDED_NDimLevel_h

#include "NDimSubnet.h"
#include "NDimBackpropLayer.h"
#include "PredictionOutputLayer.h"
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/util/XMLString.hpp>

XERCES_CPP_NAMESPACE_USE


class NDimLevel
{
	
protected:
		
	//data
	vector<int> dimProducts;
	int inputOverlap;
	
	//TODO get rid of these: in blockTransformer already
	int inputSeqLength;
	vector<int> inputSeqDims;

public:

	//data
	vector<NDimSubnet*> subnets;
	NDimBackpropLayer<Tanh>* subsampleLayer;
	BlockTransformer* blockTransformer; 

#ifdef DELAY_IN_CONN	
	//prediction layer stuff
	PredictionOutputLayer* predLayer;
	void loopPredR(int d, vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts, bool updateDerivs);
#endif
	
	//functions
	//TODO get rid of all inputOverlap stuff
	NDimLevel(const string& name, int numDims, bool multidirectional, const DOMNode* levelNode, int inputOverlap);
	~NDimLevel();
	void feedForward(const vector<int>& seqDims);
	void feedBack();
	const vector<int>& getOutputSeqDims() const;
	const vector<int>& getOutputDimProducts() const;
	int getOutputSeqLength() const;
	int getOutputDepth() const;
	int size() const {return (int)subnets.size();}
	void connectTo(Layer* outLayer);
	void connectFrom(Layer* inLayer);
	void connectFrom(NDimLevel* inLevel);
	void save(const string& indent, ostream& out) const;
	void print(ostream& out) const;
	void printInputConns(ostream& out) const;
	void build();
	void outputInternalVariables(const string& path) const;
	double injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq);

};

typedef vector<NDimSubnet*>::iterator VPNDSNI;
typedef vector<NDimSubnet*>::const_iterator VPNDSNCI;

#endif
