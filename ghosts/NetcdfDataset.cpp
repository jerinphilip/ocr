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

#define NETCDF_LIB_EXPORTS

#include "NetcdfDataset.h"
#include "NetcdfFile.h"
#include "Helpers.h"
#include "Typedefs.h"

//TODO combine the two constructors
NetcdfDataset::NetcdfDataset(const string& filename, double fraction):
		targets(0),
		targetClasses(0),
		seqLengths(0),
		inputs(0)
{
	NetcdfFile* netcdfFile = new NetcdfFile (filename);
	
	//constrain fraction
	if (fraction <= 0 || fraction > 1)
	{
		fraction = 1;
	}

	//work out how many dataSequences we need
	int numSeqs = (int)(netcdfFile->getInt("numSeqs") * fraction);

	//load sequence lengths
	seqLengths = netcdfFile->loadIntArr("seqLengths", numSeqs);

	//load inputs
	totalTimesteps = accumulate(seqLengths, seqLengths + numSeqs, 0);
	inputs = netcdfFile->loadFloatArr("inputs", totalTimesteps);

	//load targets according to type
	if (netcdfFile->contains("targetPatterns"))
	{
		targets = netcdfFile->loadFloatArr("targetPatterns", totalTimesteps);
		dataDimensions.targetPattSize = netcdfFile->getInt("targetPattSize");
	}
	else 
	{
		dataDimensions.targetPattSize = 0;
	}
        
	if (netcdfFile->contains("targetStrings"))
	{
		netcdfFile->loadStrings(targetStrings, "targetStrings", numSeqs);
	}

	if (netcdfFile->contains("labels"))
	{
		netcdfFile->loadStrings(dataDimensions.labels, "labels");
		for (int i=0; i<(int)dataDimensions.labels.size(); ++i)
		{
			dataDimensions.labelNumbers[dataDimensions.labels[i]] = i;
		}
		if (netcdfFile->contains("targetClasses"))
		{
			targetClasses = netcdfFile->loadIntArr("targetClasses", totalTimesteps);
		}
	}

	if (netcdfFile->contains("wordDelimiter"))
	{
		netcdfFile->loadString("wordDelimiter", dataDimensions.wordDelimiter);
	}
	else if (netcdfFile->contains("wordTargetStrings"))
	{
		netcdfFile->loadStrings(wordTargetStrings, "wordTargetStrings", numSeqs);
	}

	//load sequence tags, if present
	if (netcdfFile->contains("seqTags"))
	{
		netcdfFile->loadStrings(seqTags, "seqTags", numSeqs);
	}

	dataDimensions.inputPattSize = netcdfFile->getInt("inputPattSize");
	if (netcdfFile->contains("seqDims"))
	{
		dataDimensions.numDims = netcdfFile->getInt("numDims");
		seqDims = netcdfFile->loadIntArr("seqDims", numSeqs);
	}
	else
	{
		seqDims = 0;
		dataDimensions.numDims = 1;
	}
	buildData (numSeqs);
	delete netcdfFile;
}

//loads only a single sequence
NetcdfDataset::NetcdfDataset(const string& filename, int seqNum):
		targets(0),
		targetClasses(0),
		seqLengths(0),
		inputs(0)
{
	//load in data file
	NetcdfFile* netcdfFile = new NetcdfFile (filename);

	//constrain seqNum
	int totalSeqs = (int)netcdfFile->getInt("numSeqs");
	if (seqNum <= 0)
	{
		seqNum = 0;
	}
	else if (seqNum > totalSeqs)
	{
		seqNum = totalSeqs-1;
	}

	//load sequence lengths up to seqNum
	int* cumulativeSeqLengths = netcdfFile->loadIntArr("seqLengths", seqNum + 1);
	int itOffset = accumulate(cumulativeSeqLengths, cumulativeSeqLengths + seqNum, 0);

	seqLengths = new int;
	*seqLengths = *(cumulativeSeqLengths + seqNum);
	delete cumulativeSeqLengths;

	//load inputs
	totalTimesteps = *seqLengths;
	inputs = netcdfFile->loadFloatArr("inputs", totalTimesteps, itOffset);

	//load targets according to type
	if (netcdfFile->contains("targetPatterns"))
	{
		targets = netcdfFile->loadFloatArr("targetPatterns", totalTimesteps, itOffset);
		dataDimensions.targetPattSize = netcdfFile->getInt("targetPattSize");
	}
	
	if (netcdfFile->contains("targetStrings"))
	{
			netcdfFile->loadStrings(targetStrings, "targetStrings", 1, seqNum);
	}

	if (netcdfFile->contains("labels"))
	{
		netcdfFile->loadStrings(dataDimensions.labels, "labels");
		for (int i=0; i<(int)dataDimensions.labels.size(); ++i)
		{
			dataDimensions.labelNumbers[dataDimensions.labels[i]] = i;
		}
		if (netcdfFile->contains("targetClasses"))
		{
			targetClasses = netcdfFile->loadIntArr("targetClasses", totalTimesteps, itOffset);
		}
	}

	if (netcdfFile->contains("wordDelimiter"))
	{
		netcdfFile->loadString("wordDelimiter", dataDimensions.wordDelimiter);
	}
	else if (netcdfFile->contains("wordTargetStrings"))
	{
		netcdfFile->loadStrings(wordTargetStrings, "wordTargetStrings", 1, seqNum);
	}

	//load sequence tags, if present
	if (netcdfFile->contains("seqTags"))
	{
		netcdfFile->loadStrings(seqTags, "seqTags", 1, seqNum);
	}

	dataDimensions.inputPattSize = netcdfFile->getInt("inputPattSize");
	if (netcdfFile->contains("seqDims"))
	{
		dataDimensions.numDims = netcdfFile->getInt("numDims");
		seqDims = netcdfFile->loadIntArr("seqDims", 1, seqNum);
	}
	else
	{
		seqDims = 0;
		dataDimensions.numDims = 1;
	}
	buildData (1);
	delete netcdfFile;
}

void NetcdfDataset::buildData (int numSeqs)
{
	//build the data sequences
	int seqOffset = 0;
	totalTargetStringLength = 0;
	vector<double> meanTarg(dataDimensions.targetPattSize);
	int numTargs = 0;
	for (int s=0; s<numSeqs; ++s)
	{
		DataSequence* seq = new DataSequence;
		seq->inputs = inputs + (seqOffset * dataDimensions.inputPattSize);
		seq->size = seqLengths[s];
		if (targets)
		{
			seq->targetPatterns = targets + (seqOffset * dataDimensions.targetPattSize);
		}
		for (int n = 0; n < seq->size; ++n)
		{
			int frameOffset = seqOffset + n;
			if (targetClasses)
			{
				int targetClass = *(targetClasses + frameOffset);
				seq->targetClasses.push_back(targetClass);
			}
			if (targets)
			{
				const float* targ = seq->targetPatterns + (n * meanTarg.size());
#ifdef MIN_TARG_VAL
				if (targ && *targ > MIN_TARG_VAL)
#else
				if (targ)
#endif
				{
					++numTargs;
					transform(targ, targ + meanTarg.size(), meanTarg.begin(), meanTarg.begin(), plus<double>());
				}
			}
		}
		if (seqTags.size())
		{
			seq->tag = seqTags[s];
		}
		else
		{
			stringstream ss;
			ss << s;
			seq->tag = ss.str();
		}
		if (!wordTargetStrings.empty())
		{
			seq->wordTargetString = wordTargetStrings[s];
		}
		if (seqDims)
		{
			int dimOffset = s * dataDimensions.numDims;
			int dimProduct = 1;
			for (int n = dimOffset; n < dimOffset + dataDimensions.numDims; ++n)
			{
				int dim = seqDims[n];
				seq->dimensions.push_back(dim);
				dimProduct *= dim;
			}
			if (dimProduct != seq->size)
			{
				cerr << "ERROR: NetcdfDataset::buildData seq " << s << " tag " << seq->tag << endl;
				cerr << "dimProduct " << dimProduct;
				cerr << " seq->size " << seq->size;
				cerr << ",exiting" << endl;
				exit(0);
			}
		}
		else
		{
			seq->dimensions.push_back(seq->size);
		}
		if (!targetStrings.empty())
		{
			stringstream ss;
			ss << targetStrings[s];
			string str;
			seq->targetString = ss.str();
			seq->labelCounts.resize(dataDimensions.labels.size());
			while (ss >> str)
			{
				MCSII it = dataDimensions.labelNumbers.find(str);
				if (it == dataDimensions.labelNumbers.end())
				{
					cerr <<"ERROR! NetcdfDataset label " << str << " found in targetStrings but not labels" << endl;
					exit(0);
				}
				else
				{
					seq->targetSeq.push_back(it->second);
					++seq->labelCounts[it->second];
				}
			}
			if (seq->targetSeq.size() > seq->dimensions.back())
			{
				cerr << "NetcdfDataset::buildData WARNING! seq " << seq->tag << endl;
				cerr << " target sequence length " << (int)seq->targetSeq.size();
				cerr << " > than input sequence length " << seq->dimensions.back() << endl;
				//exit(0);
			}
			totalTargetStringLength += (int)seq->targetSeq.size();
		}
		dataSequences.push_back(seq);
		seqOffset += seq->size;
	}
	if (targets)
	{
		double rmsNormFactor = 0;
		transform(meanTarg.begin(), meanTarg.end(), meanTarg.begin(), bind2nd(divides<double>(), numTargs));
		for (vector<DataSequence*>::iterator it = dataSequences.begin(); it != dataSequences.end(); ++it)
		{
			const float* pattEnd = (*it)->targetPatterns + ((*it)->size * dataDimensions.targetPattSize);
			for (const float* pattIt = (*it)->targetPatterns; pattIt != pattEnd; pattIt += dataDimensions.targetPattSize)
			{
				if (pattIt)
				{
#ifdef MIN_TARG_VAL
					if (*pattIt > MIN_TARG_VAL)
#endif
					rmsNormFactor += squaredEuclideanDistance(meanTarg.begin(), meanTarg.end(), pattIt);
				}
			}
		}
		dataDimensions.rmsNormFactor = rmsNormFactor;
	}
}

NetcdfDataset::~NetcdfDataset(void)
{
	for_each(dataSequences.begin(), dataSequences.end(), deleteT<DataSequence>);
	delete inputs;
	delete targets;  
	delete seqLengths;
	delete targetClasses;
}

int NetcdfDataset::getTotalTargetStringLength() const
{
	return totalTargetStringLength;
}

void NetcdfDataset::print(ostream& out) const
{
	out << (int)dataSequences.size() << " sequences" << endl;
	out << totalTimesteps << " timesteps" << endl;
	if (!targetStrings.empty())
	{
		out << (int)dataDimensions.labels.size() << " labels" << endl;
		copy(dataDimensions.labels.begin(), dataDimensions.labels.end(), ostream_iterator<string>(cout, " "));
		out << endl;
	}
	if (!dataDimensions.labels.empty())
	{
		out << "classification targets on " << dataDimensions.labels.size() << " classes" << endl;
	}
	if (dataDimensions.wordDelimiter != "")
	{
		out << "word delimiter '" << dataDimensions.wordDelimiter  << "'" << endl;
	}
	if (!hTargetStrings.empty())
	{
		out << "hierarchical targets:" << endl;
		for (int i = 0; i < (int)hTargetStrings.size(); ++i)
		{
			out << "ctc on " << (int)hLabels[i].size() << " labels:" << endl;
			copy(hLabels[i].begin(), hLabels[i].end(), ostream_iterator<string>(cout, " "));
			out << endl;
		}
	}
}

const DataSequence& NetcdfDataset::getDataSequence(int seqNum) const
{
	if (seqNum < 0)
	{
		return *dataSequences.front();
	}
	else if (seqNum >= (int)dataSequences.size())
	{
		return *dataSequences.back();
	}
	else
	{
		return *dataSequences[seqNum];
	}
}

const DataDimensions& NetcdfDataset::getDimensions() const
{
	return dataDimensions;
}

int NetcdfDataset::getNumSeqs() const
{
	return (int)dataSequences.size();
}

int NetcdfDataset::getNumTimesteps() const
{
	return totalTimesteps;
}
