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

#ifndef _INCLUDED_BlockTransforms_h  
#define _INCLUDED_BlockTransforms_h  

#include <vector>
#include <algorithm>
#include "Typedefs.h"

using namespace std;

//TODO separate step sizes from block dims to allow e.g. block overlaps

//functions objects for transforming to and from block coords
class BlockTransformer
{

private:

	//data
	int blockSeqLength;
	int flatSeqLength;
	int flatDepth;
	int blockDepth;
	const vector<int>* flatSeqDims;
	vector<int> flatCoords;
	vector<int> blockSeqDims;
	vector<int> blockSeqDimProds;
	vector<int> blockDims;
	vector<int> flatSeqDimProds;
	vector<int> blockDimProds;
	
	//functions
	void setSeqDims(const vector<int>& seqDims);
	VDI transform(int d, VDI flatIt, vector<double>& blockedBuffer, bool flatToBlocked);

public:

	BlockTransformer(int flatDepth, const string& blockDimString);
	void blockedToFlat(vector<double>& blockedBuffer, vector<double>& flatBuffer);
	void flatToBlocked(const vector<int>& seqDims, vector<double>& flatBuffer, vector<double>& blockedBuffer);
	const vector<int>& getBlockSeqDims() const;
	const vector<int>& getBlockSeqDimProds() const;
	const vector<int>& getBlockDims() const;
	const vector<int>& getFlatSeqDims() const;
	int getBlockSeqLength() const;
	int getFlatSeqLength() const;
	int getBlockDepth() const;
	int getFlatDepth() const;
	void print(ostream& out) const;

};


#endif
