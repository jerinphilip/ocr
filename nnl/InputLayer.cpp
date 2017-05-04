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
#include <fstream>
#include "InputLayer.h"
#include "Typedefs.h"
#include "Helpers.h"

InputLayer::InputLayer(const string& n, int size, const string& blockString):
		name(n),
		inSize(size),
		blockTransformer(0)
{
	if (blockString == "")
	{
		outSize = inSize;
	}
	else
	{
		blockTransformer = new BlockTransformer(inSize, blockString);
		outSize = blockTransformer->getBlockDepth();
	}
}

InputLayer::~InputLayer(void)
{
}

void InputLayer::copyInputs (const DataSequence& seq)
{
	dataSeqDims = seq.dimensions;
	inActBuffer.resize((int)seq.size * inSize);
	inErrBuffer.resize(inActBuffer.size());
	copy(seq.inputs, seq.inputs + inActBuffer.size(), inActBuffer.begin());
	if (blockTransformer)
	{
		blockTransformer->flatToBlocked(seq.dimensions, inActBuffer, actBuffer);
		seqLength = blockTransformer->getBlockSeqLength();
	}
	else
	{
		actBuffer.swap(inActBuffer);
		seqLength = (int)seq.size;
	}
#ifdef ZERO_INPUT_CHECK
	activeOffsets.clear();
	for (VDCI actIt = actBuffer.begin(); actIt != actBuffer.end(); actIt += outSize)
	{
		activeOffsets.push_back(find_if(actIt, actIt + outSize, bind2nd(not_equal_to<double>(), 0)) != actIt + outSize);
	}
//	copy(activeOffsets.begin(), activeOffsets.end(), ostream_iterator<double>(cout, " "));
//	cout << endl;
#endif
	errorBuffer.resize(actBuffer.size());
}

void InputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(actBuffer, path + name + "_activations", outSize, &getSeqDims());
	printBufferToFile(errorBuffer, path + name + "_errors", outSize, &getSeqDims());
	if (blockTransformer)
	{
		printBufferToFile(inActBuffer, path + name + "_input_activations", inSize, &dataSeqDims);
		printBufferToFile(inErrBuffer, path + name + "_input_errors", inSize, &dataSeqDims);
	}
}

int InputLayer::outputSize() const 
{
	return outSize;
}

//TODO should be inErrors
void InputLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{
	*errBegin = &errorBuffer[offset * inSize];
	*errEnd = *errBegin + inSize;
}

void InputLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &actBuffer[offset * outSize];
	*actEnd = *actBegin + outSize;
}

int InputLayer::inputSize() const
{
	return inSize;
}

// void InputLayer::feedBack(int offset)
// {
// 	double* errBegin = &errorBuffer[offset * outSize];
// 	double* errEnd = errBegin + outSize;
// 	fill(errBegin, errEnd, 0);
// 
// 	//feed back errors from the Net
// 	for (VPBCI it = outputs.begin(); it != outputs.end(); ++it)
// 	{
// 		(*it)->feedBack(errBegin, errEnd, offset);
// 	}
// }

void InputLayer::feedBack()
{
	const vector<int>& sd = getSeqDims();
	int seqLength = accumulate(sd.begin(), sd.end(), 1, multiplies<int>());

	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	for (int offset = 0; offset < seqLength; ++offset)
	{
		double* errBegin = &errorBuffer[offset * outSize];
		double* errEnd = errBegin + outSize;
	
		//feed back errors from the Net
		for (VPBCI it = outputs.begin(); it != outputs.end(); ++it)
		{
			(*it)->feedBack(errBegin, errEnd, offset);
		}
	}
	if (blockTransformer)
	{
		blockTransformer->blockedToFlat(errorBuffer, inErrBuffer);
	}
}

const string& InputLayer::getName() const
{
	return name;
}

void InputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" input layer" << endl;
	out << "input size " << inSize << " output size " << outSize << endl;
	if (blockTransformer)
	{
		blockTransformer->print(out);
	}
}

const vector<int>& InputLayer::getSeqDims() const
{
	return blockTransformer ? blockTransformer->getBlockSeqDims() : dataSeqDims;
// 	return *seqDims;
}

int InputLayer::getSeqLength() const
{
	return seqLength;
}

