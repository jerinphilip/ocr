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
#include <sstream>
#include <iterator>
#include <math.h>
#include "BlockTransformer.h"


BlockTransformer::BlockTransformer(int flatDep, const string& blockDimString):
		flatDepth(flatDep),
		flatSeqDims(0)
{
	stringstream ss(blockDimString);
	int i;
	while (ss >> i)
	{
		blockDims.push_back(i);
	}

	//resize the vectors
	blockDimProds.resize(blockDims.size());
	flatCoords.resize(blockDims.size());
	flatSeqDimProds.resize(blockDims.size());
	blockSeqDims.resize(blockDims.size());
	blockSeqDimProds.resize(blockDims.size());
	
	//calculate the block dim products
	int dimProd = 1;
	for (int d = 0; d < (int)blockDims.size(); ++d)
	{
		blockDimProds[d] = dimProd;
		dimProd *= blockDims[d];
	}
	blockDepth = dimProd * flatDepth;
}

const vector<int>& BlockTransformer::getBlockSeqDimProds() const
{
	return blockSeqDimProds;
}

const vector<int>& BlockTransformer::getFlatSeqDims() const 
{
	return *flatSeqDims;
}

const vector<int>& BlockTransformer::getBlockSeqDims() const
{
	return blockSeqDims;
}

int BlockTransformer::getBlockSeqLength() const
{
	return blockSeqLength;
}

int BlockTransformer::getFlatSeqLength() const
{
	return flatSeqLength;
}

int BlockTransformer::getBlockDepth() const
{
	return blockDepth;
}

int BlockTransformer::getFlatDepth() const
{
	return flatDepth;
}

const vector<int>& BlockTransformer::getBlockDims() const
{
	return blockDims;
}

#define PAD_INPUTS
#ifdef PAD_INPUTS
VDI BlockTransformer::transform(int d, VDI flatIt, vector<double>& blockedBuffer, bool toBlocked)
{
	for (flatCoords[d] = 0; flatCoords[d] < flatSeqDims->at(d);)
	{
		if (d > 0)
		{
			flatIt = transform(d - 1, flatIt, blockedBuffer, toBlocked);
			++flatCoords[d];	
		}
		else
		{
			int blockOffset = 0;
			for (int d2 = 0; d2 < (int)blockDims.size(); ++d2)
			{
				int blockCoord = flatCoords[d2] / blockDims[d2];
				int remainder = flatCoords[d2] % blockDims[d2];
				blockOffset += (blockCoord * blockSeqDimProds[d2] * blockDepth) + (remainder * blockDimProds[d2] * flatDepth);
			}
			int chunkLength;
			int overshoot = flatCoords[0] +  blockDims[0] - flatSeqDims->front();
			if (overshoot > 0)
			{
				chunkLength = blockDims[0] - overshoot;
			}
			else
			{
				chunkLength = blockDims[0];
			}
			int chunkSize = chunkLength * flatDepth;
			VDI blockIt = blockedBuffer.begin() + blockOffset;
			if (toBlocked)
			{
				copy(flatIt, flatIt + chunkSize, blockIt);
			}
			else
			{
				copy(blockIt, blockIt + chunkSize, flatIt);
			}
			flatIt += chunkSize;
			flatCoords[0] += blockDims[0];
		}
	}
	return flatIt;
}

#else

VDI BlockTransformer::transform(int d, VDI flatIt, vector<double>& blockedBuffer, bool toBlocked)
{
	for (flatCoords[d] = 0; flatCoords[d] + blockDims[d] <= flatSeqDims->at(d);)
	{
		if (d > 0)
		{
			flatIt = transform(d - 1, flatIt, blockedBuffer, toBlocked);
			++flatCoords[d];	
		}
		else
		{
			int blockOffset = 0;
			for (int d2 = 0; d2 < (int)blockDims.size(); ++d2)
			{
				int blockCoord = flatCoords[d2] / blockDims[d2];
				int remainder = flatCoords[d2] % blockDims[d2];
				blockOffset += (blockCoord * blockSeqDimProds[d2] * blockDepth) + (remainder * blockDimProds[d2] * flatDepth);
			}
			int chunkSize = blockDims[0] * flatDepth;
			VDI blockIt = blockedBuffer.begin() + blockOffset;
			if (toBlocked)
			{
				copy(flatIt, flatIt + chunkSize, blockIt);
			}
			else
			{
				copy(blockIt, blockIt + chunkSize, flatIt);
			}
			flatIt += chunkSize;
			flatCoords[0] += blockDims[0];
		}
	}

	//account for trimmed inputs
	int remainder = flatSeqDims->at(d) % blockDims[d];
	if (remainder)
	{
		flatIt += remainder * flatSeqDimProds[d];	
	}
	return flatIt;
}
#endif

void BlockTransformer::setSeqDims(const vector<int>& seqDims)
{
	if (seqDims.size() != blockDims.size())
	{
		cerr << "ERROR: BlockTransformer::setSeqDims seqDims.size()=" << (int)seqDims.size();
		cerr << " blockDims.size()=" << (int)blockDims.size() << ", exiting" << endl;
		exit(0);
	}

	flatSeqDims = &seqDims;
	
	//calculate dimProducts
	int blockSeqDimProd = 1;
	int flatSeqDimProd = 1;
	for (int d = 0; d < (int)flatSeqDims->size(); ++d)
	{
		int flatSeqDim = flatSeqDims->at(d);
		int blockSeqDim = ceil((double)flatSeqDim / (double)blockDims[d]);
		blockSeqDims[d] = blockSeqDim;
		blockSeqDimProds[d] = blockSeqDimProd;
		flatSeqDimProds[d] = flatSeqDimProd;
		flatSeqDimProd *= flatSeqDim;
		blockSeqDimProd *= blockSeqDim;
	}
	flatSeqLength = flatSeqDimProd;
	blockSeqLength = blockSeqDimProd;
}

void BlockTransformer::blockedToFlat(vector<double>& blockedBuffer, vector<double>& flatBuffer)
{
	transform((int)blockDims.size() - 1, flatBuffer.begin(), blockedBuffer, false);
	//copy(blockedBuffer.begin(), blockedBuffer.end(), flatBuffer.begin());
}
	
void BlockTransformer::flatToBlocked(const vector<int>& seqDims, vector<double>& flatBuffer, vector<double>& blockedBuffer)
{
	setSeqDims(seqDims);
	blockedBuffer.resize(blockSeqLength * blockDepth);
	fill(blockedBuffer.begin(), blockedBuffer.end(), 0);
	//copy(flatBuffer.begin(), flatBuffer.end(), blockedBuffer.begin());
	transform((int)blockDims.size() - 1, flatBuffer.begin(), blockedBuffer, true);
}

void BlockTransformer::print(ostream& out) const
{
	out << "block dimensions: ";
	copy(blockDims.begin(), blockDims.end(), ostream_iterator<int>(out, " "));
	out << endl;
}
