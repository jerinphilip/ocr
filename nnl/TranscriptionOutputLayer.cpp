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
#include <limits>
#include "TranscriptionOutputLayer.h"
#include "NnMath.h"
#include "StringAlignment.h"
#include "Typedefs.h"
#include "Helpers.h"


//global debug switch
//bool sequenceDebugOutput=false;;

TranscriptionOutputLayer::TranscriptionOutputLayer(const string& nam, const vector<string>& lab, int numDims, 
					vector<string>& criteria, const map<const string, int>& labelNumbers, 
					const string& wordDelimiter, const DOMElement* element):
		name(nam),
		labels(lab),
		size((int)lab.size()+1),
		blankUnit((int)lab.size()),
		wordDecoder(0),
		dataExporter(nam),
		contiguous(true)
{

	string dictFilename = "";
	const DOMElement* dictElement = (DOMElement*)element->getElementsByTagName(XMLString::transcode("Dictionary"))->item(0);
	if (dictElement && dictElement->hasChildNodes())
	{
		dictFilename = XMLString::transcode(dictElement->getFirstChild()->getNodeValue());
	}	
	string bigramFilename = "";
	const DOMElement* bigramElement = (DOMElement*)element->getElementsByTagName(XMLString::transcode("Bigrams"))->item(0);
	if (bigramElement && bigramElement->hasChildNodes())
	{
		bigramFilename = XMLString::transcode(bigramElement->getFirstChild()->getNodeValue());
	}
	int nBest = 1;
	const DOMElement* nbElement = (DOMElement*)element->getElementsByTagName(XMLString::transcode("NBest"))->item(0);
	if (nbElement && nbElement->hasChildNodes())
	{
		nBest = atoi(XMLString::transcode(nbElement->getFirstChild()->getNodeValue()));
	}	
	int fixedLength = -1;
	const DOMElement* fixedElement = (DOMElement*)element->getElementsByTagName(XMLString::transcode("FixedLength"))->item(0);
	if (fixedElement && fixedElement->hasChildNodes())
	{
		fixedLength = atoi(XMLString::transcode(fixedElement->getFirstChild()->getNodeValue()));
	}
	if (dictFilename != "")
	{
		ifstream in (dictFilename.c_str());
		if (in.is_open())
		{
			cout << "reading in dictionary file " << dictFilename << endl;
			wordDecoder = new WordDecoder(in, labelNumbers, blankUnit, labels, wordDelimiter, size, bigramFilename, nBest, fixedLength);
		}
	}

	dEdYTerms.resize(size);
	criteria.push_back("labelErrorRate") ;
	criteria.push_back("ctcMlError") ;
	criteria.push_back("seqErrorRate");
	dataExporter.exportVal("contiguous", contiguous);
}

TranscriptionOutputLayer::~TranscriptionOutputLayer()
{
	delete wordDecoder;
}

void TranscriptionOutputLayer::feedForward(const vector<int>& dims, int seqLength, const DataSequence& seq)
{
	int actBuffSize = dims.back();
	int sliceSize = seqLength / actBuffSize;
	actBuffer.resize(actBuffSize * size);
	logActBuffer.resize(actBuffer.size());
	errorBuffer.resize(seqLength * size);
	inActBuffer.resize(errorBuffer.size());
	fill(inActBuffer.begin(), inActBuffer.end(), 0);
	fill(logActBuffer.begin(), logActBuffer.end(), 0);
	fill(actBuffer.begin(), actBuffer.end(), 0);
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	if (!sliceSize)
	{
		cout << "warning, seq " << seq.tag << " has sliceSize 0:  actBuffSize" << actBuffSize << " > seqLength " << seqLength << endl;
		cout << "exiting" << endl;
		exit(0);
	}
	double* actBegin = &actBuffer.front();
	double* logActBegin = &logActBuffer.front();
	double* inActBegin = &inActBuffer.front();
	int offset = 0;
	for (int i = 0; i < actBuffSize; ++i)
	{
		//gather acts over n-1 dimensional slice
		double* logActEnd = logActBegin + size;
		for (int j = 0; j < sliceSize; ++j, ++offset)
		{
			double* inActEnd = inActBegin + size;
			for (VPBCCI it = inputs.begin(); it != inputs.end(); ++it)
			{
				//(*it)->feedForward(logActBegin, logActEnd, offset);
				(*it)->feedForward(inActBegin, inActEnd, offset);
			}
			transform(inActBegin, inActEnd, logActBegin, logActBegin, plus<double>());
			inActBegin = inActEnd;
		}

		//center acts on 0 for safer exponentiation
		double maxAct = *max_element(logActBegin, logActEnd);
		double minAct = *min_element(logActBegin, logActEnd);
		transform(logActBegin, logActEnd, logActBegin, bind2nd(minus<double>(), maxAct + minAct / (double) 2));

		//apply softmax activation
		double* actEnd = actBegin + size;
		transform(logActBegin, logActEnd, actBegin, bdedExp);
		double expSum = accumulate(actBegin, actEnd, 0.0);
		transform(actBegin, actEnd, actBegin, bind2nd(divides<double>(), expSum));

		//divide log acts
		double logScaleFactor = -doubleMax;
		for (double* logActIt = logActBegin; logActIt != logActEnd; ++logActIt)
		{
			logScaleFactor = logAdd(logScaleFactor, *logActIt);
		}
		transform(logActBegin, logActEnd, logActBegin, bind2nd(minus<double>(), logScaleFactor));
		actBegin = actEnd;
		logActBegin = logActEnd;
	}
}

void TranscriptionOutputLayer::getStringFromPath(vector<int>& str, const vector<int>& path)
{
	str.clear();
	int prevLabel = -1;
	for (VICI pathIt=path.begin();pathIt!=path.end();++pathIt)
	{
		int label = *pathIt;
		if (label != blankUnit && (!contiguous || (str.empty() || (label != str[str.size()-1]) || (prevLabel==blankUnit))))
		{
			str.push_back(label);
		}
		if (contiguous)
		{
			prevLabel=label;
		}
	}
}

void TranscriptionOutputLayer::getMostProbableString(vector<int>& str)
{
	static vector<int> path;
	path.clear();
	for (VDCI actIt = actBuffer.begin(); actIt != actBuffer.end(); actIt += size)
	{
		VDCI maxElement = max_element(actIt, actIt + size);
		int maxLabel = (int)distance(actIt, maxElement);
		path.push_back(maxLabel);
	}
	getStringFromPath(str, path);
}

void TranscriptionOutputLayer::outputInternalVariables(const string& path) const
{
	printBufferToFile(actBuffer, path + name + "_activations", size, 0, &labels);
	printBufferToFile(errorBuffer, path + name + "_errors", size, 0, &labels);
	printBufferToFile(inActBuffer, path + name + "_input_activations", size, 0, &labels);
}

void TranscriptionOutputLayer::print(ostream& out) const
{
	out << "\"" << getName() << "\" TranscriptionOutputLayer size " << size << endl;
	dataExporter.save("", out);
	out << endl;
}

const string& TranscriptionOutputLayer::getName() const
{
	return name;
}

void TranscriptionOutputLayer::getErrors(const double** errBegin, const double** errEnd, int offset) const
{
	*errBegin = &errorBuffer[offset * size];
	*errEnd = *errBegin + size;
}

//NB: acts are collapsed, so offsets are different from error offsets
void TranscriptionOutputLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &actBuffer[offset * size];
	*actEnd = *actBegin + size;
}

vector<string> TranscriptionOutputLayer::getLabels(){
    return labels;
}

int TranscriptionOutputLayer::inputSize() const
{
	return size;
}

int TranscriptionOutputLayer::outputSize() const
{
	return size;
}

void TranscriptionOutputLayer::updateDerivs(int offset) 
{
	const double* errBegin = 0;
	const double* errEnd = 0;
	getErrors(&errBegin, &errEnd, offset);
	InputConnsImpl::updateDerivs(errBegin, errEnd, offset);
}

//NB assumes actBuffer already filled in
void TranscriptionOutputLayer::setErrToOutputDeriv(int timestep, int outputNum)
{
	int sliceSize = (int)(errorBuffer.size() / actBuffer.size());
	fill(errorBuffer.begin(), errorBuffer.end(), 0);
	int initActOffset = timestep * size;
	int endActOffset = initActOffset + size;
	int outputOffset = initActOffset + outputNum;
	double outputAct = actBuffer[outputOffset];
	VDI errIt = errorBuffer.begin() + (initActOffset * sliceSize);
	for (int j = 0; j < sliceSize; ++j)
	{
		for (int offset = initActOffset; offset < endActOffset; ++offset, ++errIt)
		{
			if (offset - initActOffset == outputNum)
			{
				*errIt = outputAct * (1 - outputAct);
			}
			else
			{
				*errIt = - (outputAct * actBuffer[offset]);
			}
		}
	}
}

double TranscriptionOutputLayer::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	//RUN THE FORWARD ALGORITHM
	int totalTime = (int)actBuffer.size() / size;
	const vector<int>& labelString = seq.targetSeq;
	int requiredTime = (int)labelString.size();
	if (contiguous)
	{
		int oldLabel = -1;
		for (VICI it = labelString.begin(); it != labelString.end(); ++it)
		{
			if (*it == oldLabel)
			{
				++requiredTime;
			}
			oldLabel = *it;
		}
	}
	if (totalTime < requiredTime)
	{
		cout << "warning, seq " << seq.tag << " has requiredTime " << requiredTime << " > totalTime " << totalTime << endl;
		return 0;
	}

	int totalSegments = ((int)labelString.size() * 2) + 1;

	//calculate (scaled) forward variables
	forwardVariables.resize(totalTime);
	for (VVDI it=forwardVariables.begin(); it!=forwardVariables.end(); ++it)
	{
		it->resize(totalSegments);
		fill(it->begin(), it->end(), -doubleMax);
	}
	double firstLabelAct = (labelString.empty() ? 0 : logActBuffer[labelString[0]]);
	double firstBlankAct = logActBuffer[blankUnit];
	forwardVariables.front()[0] = firstBlankAct;
	if (totalSegments > 1)
	{
		forwardVariables.front()[1] = firstLabelAct;
	}
	for (int t = 1; t < totalTime; ++t)
	{
		const double* acts = &logActBuffer[t * size];
		int startSegment = max(0, totalSegments - (2 *(totalTime-t)));
		int endSegment = min(totalSegments, 2 * (t + 1));
		for (int s = startSegment; s < endSegment; ++s)
		{
			double fv;
			const vector<double>& fVars = forwardVariables[t-1];

			//s odd (label output)
			if (s&1)
			{
				int labelIndex = s/2;
				int labelNum = labelString[labelIndex];
				if(contiguous)
				{
					int lastLabelNum = labelString[labelIndex-1];
					fv = logAdd(fVars[s], fVars[s-1]);
					if ((s-1) > 0 && (labelNum != lastLabelNum))
					{
						fv = logAdd(fv, fVars[s-2]);
					}
				}
				else
				{
					fv = fVars[s-1];
					if ((s-1))
					{
						fv = logAdd(fv, fVars[s-2]);
					}
				}
				fv += acts[labelNum];
			}
			//s even (blank output)
			else
			{
				fv = fVars[s];
				if (s)
				{
					fv = logAdd(fv, fVars[s-1]);
				}
				fv += acts[blankUnit];
			}
			forwardVariables[t][s] = fv;
		}
	}
	double logProb = forwardVariables.back().back();
	if (totalSegments > 1)
	{
		logProb = logAdd(logProb, forwardVariables.back()[totalSegments - 2]);
	}
	double ctcMlError = -logProb;
	if (ctcMlError >= 0)
	{
		//RUN THE BACKWARD ALGORITHM
		//init arrays
		backwardVariables.resize(totalTime);
		for (VVDI it = backwardVariables.begin(); it != backwardVariables.end(); ++it)
		{
			it->resize(totalSegments);
			fill(it->begin(), it->end(), -doubleMax);
		}

		//initialise (scaled) backward variables
		backwardVariables.back().back() = 0;
		if (totalSegments > 1)
		{
			backwardVariables.back()[totalSegments-2] = 0;
		}
		//loop over time, calculating backward variables recursively
		for (int t = totalTime - 2; t >= 0; --t)
		{
			int startSegment = max(0, totalSegments - (2*(totalTime-t)));
			int endSegment = min(totalSegments, 2*(t+1));
			const double *oldActs = &logActBuffer[(t+1) * size];
			for (int s = startSegment; s < endSegment; ++s)
			{
				double bv;
				const vector<double>& bVals = backwardVariables[t+1];
				

				//s odd (label output)
				if (s&1)
				{
					int labelNum = labelString[s/2];
					int nextLabelNum = labelString[(s+1)/2];
					if (contiguous)
					{
						bv = logAdd(bVals[s] + oldActs[labelNum], bVals[s+1] + oldActs[blankUnit]);
						if ((s < (totalSegments-2)) && (labelNum != nextLabelNum))
						{
							bv = logAdd(bv, bVals[s+2] + oldActs[nextLabelNum]);
						}
					}
					else
					{
						bv = bVals[s+1] + oldActs[blankUnit];
						if ((s < (totalSegments-2)))
						{
							bv = logAdd(bv, bVals[s+2] + oldActs[nextLabelNum]);
						}
					}
				}

				//s even (blank output)
				else
				{
					bv = bVals[s] + oldActs[blankUnit];
					if (s < (totalSegments-1))
					{
						bv = logAdd(bv, bVals[s+1] + oldActs[labelString[s/2]]);
					}
				}
				backwardVariables[t][s] = bv;
			}
		}

		//INJECT THE ERRORS
		VDI errIt = errorBuffer.begin();
		VDCI actBegin = actBuffer.begin();
		int sliceSize = (int)(errorBuffer.size() / actBuffer.size());
		for (int t = 0; t < totalTime; ++t, actBegin += size)
		{
			//calculate dE/dY terms and rescaling factor
			fill(dEdYTerms.begin(), dEdYTerms.end(), -doubleMax);
			double normTerm = -doubleMax;
			for (int s = 0; s < totalSegments; ++s)
			{
				//k = no decision node for even s, node in target string for odd s
				int k = (s&1) ? labelString[s/2] : blankUnit;
				double forVar = forwardVariables[t][s];
				double backVar = backwardVariables[t][s];
				double fbProduct = forVar + backVar;
				dEdYTerms[k] = logAdd(dEdYTerms[k], fbProduct);
				normTerm = logAdd(normTerm, fbProduct);
			}

			//inject errors thru softmax layer for all acts in slice
			for (int j = 0; j < sliceSize; ++j)
			{
				for (int k = 0; k < size; ++k, ++errIt)
				{
					*errIt = actBegin[k] - safeExp(dEdYTerms[k] - normTerm);
				}
			}
		}

		//get most probable label seq
		vector<int> outputLabelSeq;
		getMostProbableString(outputLabelSeq);

#ifndef _WIN32
        /*
		if (sequenceDebugOutput)
		{
			//debug code
			cout << "output label string:" << endl;
			for (VII it = outputLabelSeq.begin(); it != outputLabelSeq.end(); ++it)
			{
				cout << labels[*it] << " ";
			}
			cout << endl;
		}
        */
#endif

		if (wordDecoder)
		{
			//vector<int> outputWordLabelSeq;
			wordDecoder->decode(/*outputWordLabelSeq,*/ errorMap, seq, logActBuffer);
		}
		
		StringAlignment alignment(seq.targetSeq, outputLabelSeq);
		int targetLength = seq.targetSeq.size();
		double labelErr = alignment.getDistance();
		double substitutions = alignment.getSubstitutions();
		double insertions = alignment.getInsertions();
		double deletions = alignment.getDeletions();



		//store errors in map
		double normLabelErrorRate = (double)labelErr / (double)targetLength;
        /* Freshly commented out.
		errorMap["normLabelErrorRate"] += make_pair<int,double>(1, normLabelErrorRate);
		errorMap["substitutions"] += make_pair<int,double>(targetLength, substitutions);
		errorMap["insertions"] += make_pair<int,double>(targetLength, insertions);
		errorMap["deletions"] += make_pair<int,double>(targetLength, deletions);
		errorMap["labelErrorRate"] += make_pair<int,double>(targetLength, labelErr);
		errorMap["seqErrorRate"] += make_pair<int,double>(1, (labelErr > 0));
		errorMap["ctcMlError"] += make_pair<int,double>(1, ctcMlError);
        */

		//TODO substitution confusion matrix, insertion and deletion lists
#ifndef _WIN32
        /*
		if (sequenceDebugOutput)
		{
			cout << "ctcMlError " <<  ctcMlError << endl;
			cout << "labelErrorRate " <<  labelErr << endl;
			cout << "insertions " <<  insertions << endl;
			cout << "deletions " <<  deletions << endl;
			cout << "substitutions " <<  substitutions << endl;
		}
        */
#endif
		return ctcMlError;
	}
	else
	{
		return doubleMax;
	}
}



