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

#include "BlockLayer.h"
#include "Helpers.h"
#include "Typedefs.h"


BlockLayer::BlockLayer(const string& nam, BlockTransformer* bt):
		name(nam),
		blockTransformer(bt),
		size(bt->getBlockDepth())
{
}

// BlockLayer::BlockLayer(const string& nam, int siz):
// 		name(nam),
// 		size(siz)
// {
// }

//BlockLayer::~BlockLayer()
//{
//}

void BlockLayer::feedForward(vector<double>& layerActs, const vector<int>& seqDims)
{
	blockTransformer->flatToBlocked(seqDims, layerActs, actBuffer);

}

void BlockLayer::feedBack(vector<double>& layerErrors)
{
 	int blockSeqLength = blockTransformer->getBlockSeqLength();
	resizeErrorBuffer(blockSeqLength);
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	VDI errBegin = errorBuffer.begin();
	int offset = 0;
	for (VDI errEnd = errBegin + size; errEnd != errorBuffer.end(); errBegin += size, errEnd += size, ++offset)
	{
		for (VPBCCI outputIt = outputs.begin(); outputIt != outputs.end(); ++outputIt)
		{
			(*outputIt)->feedBack(&*errBegin, &*errEnd, offset);
		}
	}
	blockTransformer->blockedToFlat(errorBuffer, layerErrors);
// 		for (int i = 0; i < blockSeqLength; ++i)
// 		{
// 			blockLayer->feedBack(i);
// 		}
	// 
// 		//copy block errors to layer errors
// 		blockTransformer->blockedToFlat(blockLayer->errorBuffer, layer->getErrorBuffer());
	
	
// 	double* errBegin = &errorBuffer[offset * size];
// 	double* errEnd = errBegin + size;
// 	fill(errBegin, errEnd, 0);
// 
// 	//feed back errors from the Net
// 	for (VPBCCI outputIt = outputs.begin(); outputIt != outputs.end(); ++outputIt)
// 	{
// 		(*outputIt)->feedBack(errBegin, errEnd, offset);
// 	}
}

void BlockLayer::resizeErrorBuffer(int buffSize)
{
	errorBuffer.resize(buffSize * size);
}

void BlockLayer::outputInternalVariables(const string& path) const
{
	const vector<int>& seqDims = blockTransformer->getBlockSeqDims();
	printBufferToFile(actBuffer, path + getName() + "_activations", size, &seqDims);
	printBufferToFile(errorBuffer, path + getName() + "_errors", size, &seqDims);
}

int BlockLayer::outputSize() const
{
	return size;
}

int BlockLayer::inputSize() const
{
	return size;
}

void BlockLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	int totalOffset = offset * size;
	if (offset < 0 || totalOffset >= (int)actBuffer.size())
	{
		cerr << "BlockLayer::getActs offset " << offset << " out of range, exiting" <<endl;
		exit(0);
	}
	*actBegin = &actBuffer[totalOffset];
	*actEnd = *actBegin + size;
}

void BlockLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{		
	int totalOffset = offset * size;
	if (offset < 0 || totalOffset >= (int)errorBuffer.size())
	{
		cerr << "BlockLayer::getErrors offset " << offset << " out of range, exiting" <<endl;
		exit(0);
	}
	*errBegin = &errorBuffer[totalOffset];
	*errEnd = *errBegin + size;
}

const string& BlockLayer::getName() const
{
	return name;
}
