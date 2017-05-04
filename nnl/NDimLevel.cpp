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

#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <math.h>
#include "NDimLevel.h"

NDimLevel::NDimLevel(const string& name, int numDims, bool multidirectional, const DOMNode* levelNode, int inputOvlap):
		blockTransformer(0),
		subsampleLayer(0),
		inputOverlap(inputOvlap)
#ifdef DELAY_IN_CONN
		,predLayer(0)
#endif
{
	dimProducts.resize(numDims);
	string blockDimString = "";
	DOMNamedNodeMap* attributes = levelNode->getAttributes();
	DOMNode* blockNode = attributes->getNamedItem(XMLString::transcode("blockDims"));
	int inputSize = atoi(XMLString::transcode(attributes->getNamedItem(XMLString::transcode("size"))->getNodeValue()));
	if (blockNode)
	{
		blockDimString = XMLString::transcode(blockNode->getNodeValue());
		blockTransformer = new BlockTransformer(inputSize, blockDimString);
	}

	//create the subnets, using the config number to initialise the directions
	//TODO allow for arbitrary combinations of subnet directions: subnets="0 1 0 0 1 1 0 0" etc
	int numConfigs = multidirectional ? (int)pow(2.0,numDims) : 1;
#ifdef DELAY_IN_CONN
	vector<int> delayCoords (numDims);
#endif
	for (int i = 0; i < numConfigs; ++i)
	{
		NDimSubnet* subnet = new NDimSubnet(name, numDims, i, levelNode, blockTransformer);
// 		cout << i << endl;
		for (int d = 0; d < numDims; ++d)
		{
#ifdef DELAY_IN_CONN
// 			cout << subnet->actDirections[d] << endl;
			fill(delayCoords.begin(), delayCoords.end(), 0);
			delayCoords[d] = subnet->actDirections[d];
			new Connection(subnet->layer, subnet->layer, d, &delayCoords);
#else
			new Connection(subnet->layer, subnet->layer, d);
#endif
		}
		subnets.push_back(subnet);
	}
}

NDimLevel::~NDimLevel()
{
	for_each(subnets.begin(), subnets.end(), deleteT<NDimSubnet>);
	delete blockTransformer;
	delete subsampleLayer;
}

void NDimLevel::print(ostream& out) const
{
	if (blockTransformer)
	{
		const vector<int>& blockDims = blockTransformer->getBlockDims();
		out << "block dims ";
		copy(blockDims.begin(), blockDims.end(), ostream_iterator<int>(out, " "));
		out << endl;
		out << "flat depth " << blockTransformer->getFlatDepth();
		out << " block depth " << blockTransformer->getBlockDepth();
		out << endl;
	}
	out << "subnets:" << endl;
	for (VPNDSNCI it = subnets.begin(); it != subnets.end(); ++it)
	{
		(*it)->layer->print(out);
	}
	if (subsampleLayer)
	{
		out << "subsample layer:" << endl;
		subsampleLayer->print(out);
	}
#ifdef DELAY_IN_CONN
	if (predLayer)
	{
		out << "subsample layer:" << endl;
		predLayer->print(out);
	}
#endif
}

void NDimLevel::printInputConns(ostream& out) const
{
	for (VPNDSNCI it = subnets.begin(); it != subnets.end(); ++it)
	{
		(*it)->layer->printInputConns(out);
	}
	if (subsampleLayer)
	{	
		subsampleLayer->printInputConns(out);
	}
#ifdef DELAY_IN_CONN
	if (predLayer)
	{	
		predLayer->printInputConns(out);
	}
#endif
}

void NDimLevel::build()
{
	for (VPNDSNI it = subnets.begin(); it != subnets.end(); ++it)
	{
		(*it)->layer->build();
	}
	if (subsampleLayer)
	{	
		subsampleLayer->build();
	}
#ifdef DELAY_IN_CONN
	if (predLayer)
	{	
		predLayer->build();
	}
#endif
}

void NDimLevel::outputInternalVariables(const string& path) const
{
	for (VPNDSNCI it = subnets.begin(); it != subnets.end(); ++it)
	{
		(*it)->layer->outputInternalVariables(path);
		if ((*it)->blockLayer)
		{
			(*it)->blockLayer->outputInternalVariables(path);
		}
	}
	if (subsampleLayer)
	{	
		subsampleLayer->outputInternalVariables(path);
	}
#ifdef DELAY_IN_CONN
	if (predLayer)
	{	
		predLayer->outputInternalVariables(path);
	}
#endif
}

void NDimLevel::connectTo(Layer* outLayer)
{
	for (VPNDSNI it = subnets.begin(); it != subnets.end(); ++it)
	{
		if (blockTransformer)
		{
			new Connection((*it)->blockLayer, outLayer);
		}
		else
		{
			new Connection((*it)->layer, outLayer);
		}
	}
}

void NDimLevel::connectFrom(Layer* inLayer)
{
	for (VPNDSNI it = subnets.begin(); it != subnets.end(); ++it)
	{
		new Connection(inLayer, (*it)->layer);
	}
}

void NDimLevel::connectFrom(NDimLevel* inLevel)
{
	for (VPNDSNI it = subnets.begin(); it != subnets.end(); ++it)
	{
		inLevel->connectTo((*it)->layer);
	}
}

double NDimLevel::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
#ifdef DELAY_IN_CONN
	if (predLayer)
	{
		return predLayer->injectSequenceErrors(errorMap, seq);
	}
	else
#endif
	{
		return 0;
	}
	
}

void NDimLevel::feedForward(const vector<int>& seqDims)
{
	//calculate dimProducts
	inputSeqDims = seqDims;
	int dimProduct = 1;
	for (int d = 0; d < (int)inputSeqDims.size(); ++d)
	{
		dimProducts[d] = dimProduct;
		dimProduct *= inputSeqDims[d];
	}
	inputSeqLength = dimProduct;

	for (VPNDSNI it = subnets.begin(); it != subnets.end(); ++it)
	{
		(*it)->feedForward(inputSeqDims, dimProducts, blockTransformer, inputOverlap);
	}
	if (subsampleLayer)
	{
		int outputSeqLength = blockTransformer ? blockTransformer->getBlockSeqLength() : inputSeqLength;
		subsampleLayer->resizeBuffers(outputSeqLength);
		const vector<int>& sd = blockTransformer ? blockTransformer->getBlockSeqDims() : inputSeqDims;
// 		cout << outputSeqLength << endl;
// 		copy (sd.begin(), sd.end(), ostream_iterator<int>(cout, " "));
// 		cout << endl;
// 		copy (inputSeqDims.begin(), inputSeqDims.end(), ostream_iterator<int>(cout, " "));
// 		cout << endl;
		for (int offset = 0; offset < outputSeqLength; ++offset)
		{
			//HACK dummy values for actSteps, coords etc 
			//WILL FAIL if subsample layer is recurrent
			subsampleLayer->feedForward(offset, sd, sd, sd, 0);
		}	
	}
#ifdef DELAY_IN_CONN
	if (predLayer)
	{
		predLayer->resizeBuffers();
		vector<int> coords(seqDims.size(), 0);
		loopPredR((int)seqDims.size() - 1, coords, seqDims, dimProducts, false);
	}
#endif
}

void NDimLevel::feedBack()
{
#ifdef DELAY_IN_CONN
	if (predLayer)
	{
		//NOTE will fail if block transformer exists
		vector<int> coords(inputSeqDims.size(), 0);
		loopPredR((int)inputSeqDims.size() - 1, coords, inputSeqDims, dimProducts, true);
	}
#endif
	if (subsampleLayer)
	{
		int outputSeqLength = blockTransformer ? blockTransformer->getBlockSeqLength() : inputSeqLength;
		for (int offset = 0; offset < outputSeqLength; ++offset)
		{
			//HACK dummy values for actSteps, coords etc 
			//WILL FAIL is subsample layer is recurrent
			subsampleLayer->feedBack(offset, inputSeqDims, inputSeqDims, inputSeqDims, inputOverlap);
			subsampleLayer->updateDerivs(offset, inputSeqDims, inputSeqDims, inputSeqDims, inputOverlap);
		}	
	}
	for (VPNDSNI it = subnets.begin(); it != subnets.end(); ++it)
	{
		(*it)->feedBack(inputSeqDims, dimProducts, blockTransformer, inputOverlap);
	}
}
#ifdef DELAY_IN_CONN
void NDimLevel::loopPredR(int d, vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts, bool updateDerivs)
{
	for (int i = 0; i < seqDims[d]; ++i)
	{
		if (d > 0)
		{
			loopPredR(d - 1, coords, seqDims, dimProducts, updateDerivs);
		}
		else
		{
			if (updateDerivs)
			{
				predLayer->feedBack(coords, seqDims, dimProducts);
				predLayer->updateDerivs(coords, seqDims, dimProducts);
			}
			else
			{
				predLayer->feedForward(coords, seqDims, dimProducts);
			}
		}
		++coords[d];
	}
	coords[d] = 0;
}
#endif
const vector<int>& NDimLevel::getOutputSeqDims () const
{
	if (blockTransformer)
	{
		return blockTransformer->getBlockSeqDims();
	}
	else
	{
		return inputSeqDims;
	}
}

const vector<int>& NDimLevel::getOutputDimProducts() const
{
	if (blockTransformer)
	{
		return blockTransformer->getBlockSeqDimProds();
	}
	else
	{
		return dimProducts;
	}
}

int NDimLevel::getOutputSeqLength() const
{
	if (blockTransformer)
	{
		return blockTransformer->getBlockSeqLength();
	}
	else
	{
		return inputSeqLength;
	}
}

int NDimLevel::getOutputDepth() const
{
	if (blockTransformer)
	{
		return blockTransformer->getBlockDepth();
	}
	else
	{
		return subnets.front()->layer->outputSize();
	}
}

