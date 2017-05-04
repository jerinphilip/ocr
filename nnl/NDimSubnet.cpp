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

#include "NDimSubnet.h"
#include "NDimBackpropLayer.h"
#include "NDimLstmLayer.h"


NDimSubnet::NDimSubnet(const string& name, int numDims, int configNumber, const DOMNode* layerNode, BlockTransformer* blockTransformer):
		blockLayer(0)
{
	DOMNamedNodeMap* attributes = layerNode->getAttributes();

	//TODO output error message if any of these not found
	string actFn = XMLString::transcode(attributes->getNamedItem(XMLString::transcode("actFn"))->getNodeValue());
	int size = atoi(XMLString::transcode(attributes->getNamedItem(XMLString::transcode("size"))->getNodeValue()));
	stringstream ss;
	ss << name << "_subnet_" << configNumber;

	//build layer according to layer type
	string layerType = XMLString::transcode(layerNode->getNodeName());
	if (layerType == "Backprop")
	{
		if (actFn == "tanh")
		{
			layer = new NDimBackpropLayer<Tanh> (ss.str(), actFn, size);
		}
		else if (actFn == "logistic")
		{
			layer = new NDimBackpropLayer<Logistic> (ss.str(), actFn, size);
		}
		else if (actFn == "identity")
		{
			layer = new NDimBackpropLayer<Identity> (ss.str(), actFn, size);
		}
		else
		{
			cerr << "error: activation function " << actFn << " not supported for layer type " << layerType << ", exiting" << endl;
			exit (0);
		}
	}
	else if (layerType == "Lstm")
	{
		if (actFn == "tanh tanh logistic")
		{
			layer = new NDimLstmLayer<Tanh,Tanh,Logistic> (ss.str(), actFn, size, numDims);
		}
		else if (actFn == "maxmin2 identity logistic")
		{
			layer = new NDimLstmLayer<Maxmin2,Identity,Logistic> (ss.str(), actFn, size, numDims);
		}
		else
		{
			cerr << "error: activation function " << actFn << " not supported for layer type " << layerType << ", exiting" << endl;
			exit (0);
		}
	}
	else
	{
		cerr << "error: layerType " << layerType << " not supported by NDimSubnet" << endl;
		exit (0);
	}

	//create a block layer, if doing block transormations
	if (blockTransformer)
	{
		blockLayer = new BlockLayer(ss.str() + "_block", blockTransformer);
// 		blockLayer = new BlockLayer(ss.str() + "_block", blockTransformer->getBlockDepth());
	}

	//resize the arrays
	actDirections.resize(numDims);
#ifndef DELAY_IN_CONN
	actSteps.resize(numDims);
#endif
	coords.resize(numDims);

	//initialse the act directions
	for (int d = 0; d < numDims; ++d)
	{
		int testBit = 1 << d;
		if (configNumber & testBit)
		{
			actDirections[d] = 1;
		}
		else
		{
			actDirections[d] = -1;
		}
	}
}

NDimSubnet::~NDimSubnet()
{
	delete layer; 
	delete blockLayer; 
}

int NDimSubnet::calcOffset(const vector<int>& dimProducts) const
{
	return inner_product(coords.begin(), coords.end(), dimProducts.begin(), 0);
}

void NDimSubnet::initCoords(bool backwards, const vector<int>& seqDims)
{
	for (int d = 0; d < (int)coords.size(); ++d)
	{
		if (backwards == (actDirections[d] == 1))
		{
			coords[d] = 0;
		}
		else
		{
			coords[d] = seqDims[d] - 1;
		}
	}
}

void NDimSubnet::feedForward(const vector<int>& seqDims, const vector<int>& dimProducts, 
							 BlockTransformer* blockTransformer, int inputOverlap)
{
#ifndef DELAY_IN_CONN
	//calculate inputSteps and actSteps from seq dimensions
	transform(actDirections.begin(), actDirections.end(), dimProducts.begin(), actSteps.begin(), multiplies<double>());

	if (inputOverlap)
	{
		for (int dim = 1; dim < (int)actSteps.size(); ++dim)
		{
			int offset = 0;
			for (int d = 0; d < dim; ++d)
			{
				offset -= inputOverlap * dimProducts[d] * actDirections[d];
			}
			actSteps[dim] += offset;
		}
	}
#endif
	//resize the buffers and init the coords
	layer->resizeBuffers(accumulate(seqDims.begin(), seqDims.end(), 1, multiplies<int>()));
	initCoords(false, seqDims);
	
	//feed forward, looping recursively over dimensions
	feedForwardR((int)seqDims.size() - 1, seqDims, dimProducts, inputOverlap);

	//collapse to blocked sequence, if using blocks
	if (blockLayer)
	{
		blockLayer->feedForward(layer->getActBuffer(), seqDims);
// 		blockTransformer->flatToBlocked(seqDims, layer->getActBuffer(), blockLayer->actBuffer);
	}
}

//TODO more elegant way of looping over multidim matrices (BOOST library?)
void NDimSubnet::feedForwardR(int d, const vector<int>& seqDims, const vector<int>& dimProducts, int inputOverlap)
{
	for (int i = 0; i < seqDims[d]; ++i)
	{
		if (d > 0)
		{
			feedForwardR(d - 1, seqDims, dimProducts, inputOverlap);
		}
		else
		{
			int offset = calcOffset(dimProducts);
#ifdef DELAY_IN_CONN
			layer->feedForward(coords, seqDims, dimProducts);
#else
			layer->feedForward(offset, coords, seqDims, actSteps, inputOverlap);
#endif
		}
		coords[d] -= actDirections[d];
	}
	coords[d] = ((actDirections[d] == -1) ? 0 : seqDims[d] - 1);
}

void NDimSubnet::feedBack(const vector<int>& seqDims, const vector<int>& dimProducts, BlockTransformer* blockTransformer, int inputOverlap)
{
	if (blockLayer)
	{
		//if using blocking, feed back to blocked layer	
 		blockLayer->feedBack(layer->getErrorBuffer());
// 		int blockSeqLength = blockTransformer->getBlockSeqLength();
// 		blockLayer->resizeErrorBuffer(blockSeqLength);
// 		for (int i = 0; i < blockSeqLength; ++i)
// 		{
// 			blockLayer->feedBack(i);
// 		}
// 
// 		//copy block errors to layer errors
// 		blockTransformer->blockedToFlat(blockLayer->errorBuffer, layer->getErrorBuffer());
	}

	//feed back errors to layer
	initCoords(true, seqDims);
	feedBackR((int)seqDims.size() - 1, seqDims, dimProducts, inputOverlap);
}	

void NDimSubnet::feedBackR(int d, const vector<int>& seqDims, const vector<int>& dimProducts, int inputOverlap)
{
	for (int i = 0; i < seqDims[d]; ++i)
	{
		if (d > 0)
		{
			feedBackR(d - 1, seqDims, dimProducts, inputOverlap);
		}
		else
		{
			int offset = calcOffset(dimProducts);
#ifdef DELAY_IN_CONN
			layer->feedBack(coords, seqDims, dimProducts);
			layer->updateDerivs(coords, seqDims, dimProducts);
#else
			layer->feedBack(offset, coords, seqDims, actSteps, inputOverlap);
			layer->updateDerivs(offset, coords, seqDims, actSteps, inputOverlap);
#endif
		}
		coords[d] += actDirections[d];
	}
	coords[d] = ((actDirections[d] == 1) ? 0 : seqDims[d] - 1);
}
