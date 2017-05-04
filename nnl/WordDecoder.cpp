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

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <time.h>
#include "WordDecoder.h"
#include "NnMath.h"
#include "StringAlignment.h"
#include "Typedefs.h"

extern bool sequenceDebugOutput;

WordDecoder::WordDecoder(ifstream& dictFile, const map<const string, int>& labelNumbers, int blankUnit, 
						 const vector<string>& lab, const string& wd, int os, const string& bigramFilename,
						 int nb, int fl):
		wordDelimiter(wd),
		labels(lab),
		outputSize(os),
		meanBigramProb(0),
		nBest(nb),
		fixedLength(fl)
{
	string line;
	while (getline(dictFile, line))
	{
		if (line != "")
		{
			Word* word = new Word(line, labelNumbers, blankUnit);
			if (word->error)
			{
				delete word;
			}
			else
			{
				wordIndices.insert(make_pair(word->word, (int)words.size()));//[word->word] = (int)words.size();
				if (wordDelimiter != "")
				{
					labelStrIndices[&line[line.rfind("  ") + 2]] = (int)words.size();
				}
				words.push_back(word);
			}
		}
	}
	cout << (int)words.size() << " words read" << endl;
	nullTheToken = new TheToken(0,0,-doubleMax);

	if(bigramFilename != "")
	{
		langModelWeight = 1;
		//string bigramFilename = "bigrams_small.txt";
		ifstream bigramFile (bigramFilename.c_str());
		if (bigramFile.is_open())
		{
			cout << "reading transition probabilities from " << bigramFilename << endl;
			string line;
			while (line.find("\\1-grams:") == string::npos && getline(bigramFile, line));
			while (line.find("\\2-grams:") == string::npos && getline(bigramFile, line))
			{
				if (line != "")
				{
					stringstream ss;
					ss << line;
					double inProb;
					double outProb;
					string word;
					ss >> inProb >> word >> outProb;
					pair<MMCSICI,MMCSICI> indices = wordIndices.equal_range(word);
					if (indices.first != indices.second)
					{
						for (MMCSICI it = indices.first; it != indices.second; ++it)
						{
							Word* w = words[it->second];
							oneGramProbs[w] = make_pair(inProb, outProb);
						}
					}
					else
					{
						cout << "WARNING word " << word << " found in oneGrams but not in dict" << endl;
					}
					/*MCSII wordIt = wordIndices.find(word);
					if (wordIt != wordIndices.end())
					{
						
						Word* w = words[wordIt->second];
						oneGramProbs[w] = make_pair<double, double>(inProb, outProb);
					}
					else
					{
						cout << "WARNING word " << word << " found in oneGrams but not in dict" << endl;
					}*/
				}
			}
			cout << (int)oneGramProbs.size() << " one-grams read" << endl;
			int numMissingOneGrams = (int)(words.size() - oneGramProbs.size());
			if (numMissingOneGrams)
			{
				cout << "WARNING: " << numMissingOneGrams << " words with missing 1-grams" << endl;
				for (VPWI it = words.begin(); it != words.end(); ++it)
				{
					if (oneGramProbs.find(*it) == oneGramProbs.end())
					{
						cout << (*it)->word << endl;
					}
				}
			}
			while (line.find("\\end\\") == string::npos && getline(bigramFile, line))
			{
				if (line != "")
				{
					stringstream ss;
					ss << line;
					double prob;
					string word1;
					string word2;
					ss >> prob >> word1 >> word2;
					
					int numBigramProbs = 0;
					pair<MMCSICI,MMCSICI> fromIndices = wordIndices.equal_range(word1);
					pair<MMCSICI,MMCSICI> toIndices = wordIndices.equal_range(word2);
					if (fromIndices.first != fromIndices.second && toIndices.first != toIndices.second)
					{
						for (MMCSICI fromIt = fromIndices.first; fromIt != fromIndices.second; ++fromIt)
						{
							for (MMCSICI toIt = toIndices.first; toIt != toIndices.second; ++toIt)
							{
								biGramProbs[make_pair<const Word*, const Word*>(words[fromIt->second], words[toIt->second])] = prob;
							}
							
						}
						meanBigramProb += prob;
						++numBigramProbs;
					}
				
					/*MCSII fromIt = wordIndices.find(word1);
					MCSII toIt = wordIndices.find(word2);
					if (fromIt != wordIndices.end() && toIt != wordIndices.end())
					{
						biGramProbs[make_pair<const Word*, const Word*>(words[fromIt->second], words[toIt->second])] = prob;
						meanBigramProb += prob;
					}*/
					meanBigramProb /= numBigramProbs;
				}
			}
			//meanBigramProb /= biGramProbs.size();
			cout << (int)biGramProbs.size() << " bi-grams read" << endl;
			cout << "meanBigramProb " << meanBigramProb << endl;
		}
		else
		{
			cout << "unable to find bigram file " << bigramFilename << endl;
		}
	}
}

WordDecoder::~WordDecoder(void)
{
	for_each(words.begin(), words.end(), deleteT<Word>);
	delete nullTheToken;
}

void WordDecoder::stringToWordSequence(const string& str, vector<int>& wordSequence)
{
	wordSequence.clear();
	stringstream ss;
	ss << str;
	string word;
	while (ss >> word)
	{
		pair<MMCSICI,MMCSICI> indices = wordIndices.equal_range(word);
		if (indices.first != indices.second)
		{
			wordSequence.push_back(indices.first->second);
		}
		/*MCSII wordIt = wordIndices.find(word);
		if (wordIt != wordIndices.end())
		{
			wordSequence.push_back(wordIt->second);
		}*/
		else
		{
			wordSequence.push_back(-1);
		}
	}
}

void WordDecoder::labelToWordSequence(const vector<int>& labelSequence, vector<int>& wordSequence)
{
	if (wordDelimiter == "")
	{
		cerr << "ERROR: WordDecoder::labelToWordSequence called with empty wordDelimiter, exiting" << endl;
		exit(0);
	}
	wordSequence.clear();
	for (VICI it = labelSequence.begin(); it != labelSequence.end(); ++it)
	{
		stringstream ss;
		for (; labels[*it] != wordDelimiter; ++it)
		{
			if (it == labelSequence.end())
			{
				cerr << "ERROR: WordDecoder::labelToWordSequence it == labelSequence.end(), exiting" << endl;
				exit(0);
			}
			ss << labels[*it] << ' ';
		}
		ss << wordDelimiter;
		MCSII labelStrIt = labelStrIndices.find(ss.str());
		if (labelStrIt != labelStrIndices.end())
		{
			wordSequence.push_back(labelStrIt->second);
		}
		else
		{
			wordSequence.push_back(-1);
		}
	}
}

//void WordDecoder::wordToLabelSequence(const vector<int>& wordSequence, vector<int>& labelSequence)
//{
//	labelSequence.clear();
//	for (VICI it = wordSequence.begin(); it != wordSequence.end(); ++it)
//	{
//		copy (words[*it]->labelSeq.begin(), words[*it]->labelSeq.end(), back_insert_iterator<vector<int> >(labelSequence));
//	}
//}

double WordDecoder::getTransitionProb(const Word* from, const Word* to)
{
	MPWWDCI it = biGramProbs.find(make_pair(from, to));
	if (it == biGramProbs.end())
	{
		MPWDDCI fromIt = oneGramProbs.find(from);
		MPWDDCI toIt = oneGramProbs.find(to);
		if (fromIt != oneGramProbs.end() && toIt != oneGramProbs.end())
		{
			return fromIt->second.second + toIt->second.first;
		}
		else
		{
			//cerr << "WARNING: WordDecoder::getTransitionProb either \'" << from->word << "\' or \'" << to->word << "\' not in 1-grams ";
			//cerr << "prob = " << -doubleMax << endl;
			return 2 * meanBigramProb;
		}
	}
	else
	{
		return it->second;
	}
}

void WordDecoder::decode(/*vector<int>& outputSeq,*/ map<const string, pair<int,double> >& errorMap, 
						 const DataSequence& seq, const vector<double>& logActs)
{
	int totalTime = (int)logActs.size() / outputSize;
	vector<int> targetWordSequence;
	if (wordDelimiter == "")
	{
		stringToWordSequence(seq.wordTargetString, targetWordSequence);
	}
	else
	{
		labelToWordSequence(seq.targetSeq, targetWordSequence);
	}

#ifndef _WIN32
	if (sequenceDebugOutput)
	{
		cout << "target word string:" << endl;
		for (VICI it = targetWordSequence.begin(); it != targetWordSequence.end(); ++it)
		{
			if (*it >= 0)
			{
				cout << words[*it]->word << ' ';
			}
			else
			{
				cout << "OUT_OF_DICT ";
			}
		}
		cout << endl;
	}
#endif

	/*cout << "seq length " << totalTime << endl;
	cout << "target label string:" << endl;
	cout << seq.targetString << endl;
	cout << "target label seq:" << endl;
	copy(seq.targetSeq.begin(), seq.targetSeq.end(), ostream_iterator<int>(cout, " "));
	cout << endl;
	cout << "target word string:" << endl;
	cout << seq.wordTargetString << endl;
	cout << "target word sequence:" << endl;
	copy(targetWordSequence.begin(), targetWordSequence.end(), ostream_iterator<int>(cout, " "));
	cout << endl;*/

	time_t seconds = time (NULL);

	//clean up the old tokens
	for (VPTI it = tokens.begin(); it != tokens.end(); ++it)
	{
		Word::tokenStore.deleteObject(*it);
	}
	tokens.clear();
	stringstream wordStream;
	TheToken* bestTheToken = 0;
// 	if (oneGramProbs.empty())
// 	{
// 		//initialise the words and get the best initial token (0 unless some word has length 1)
// 		TheToken* testTheToken = nullTheToken;
// 		for (VPWI it = words.begin(); it != words.end(); ++it)
// 		{
// 			testTheToken = TheToken::maxTheTokenProb(testTheToken, (*it)->initTheTokens(logActs));
// 		}
// 		bestTheToken = Word::tokenStore.copy(testTheToken);
// 		tokens.push_back(bestTheToken);
// 
// 		//pass the tokens
// 		for (int t = 1; t < totalTime; ++t)
// 		{
// 			const double* logActBegin = &logActs[t * outputSize];
// 			testTheToken = nullTheToken;
// 			for (VPWI it = words.begin(); it != words.end(); ++it)
// 			{
// 				testTheToken = TheToken::maxTheTokenProb(testTheToken, (*it)->passTheTokens(bestTheToken, 0, logActBegin, t, totalTime));
// 			}
// 			bestTheToken = Word::tokenStore.copy(testTheToken);
// 			tokens.push_back(bestTheToken);
// 		}
// 	}
// 	else
	{
		//clean up old tokens
		for (SPTTPGI it = newBestTheTokens.begin(); it != newBestTheTokens.end(); ++it)
		{
			Word::tokenStore.deleteObject(*it);
		}
		newBestTheTokens.clear();
		oldBestTheTokens.clear();

		//initialise the words and get the best initial token (0 unless some word has length 1)
		for (VPWI it = words.begin(); it != words.end(); ++it)
		{
			TheToken* t = Word::tokenStore.copy((*it)->initTheTokens(logActs));
			newBestTheTokens.insert(t);
			if (oneGramProbs.empty() && newBestTheTokens.size() > nBest)
			{
				SPTTPGI endTokIt = --newBestTheTokens.end();
				newBestTheTokens.erase(endTokIt);
				Word::tokenStore.deleteObject(*endTokIt);
			}
		}

		//pass the tokens
		for (int t = 1; t < totalTime; ++t)
		{
			newBestTheTokens.swap(oldBestTheTokens);
			for (SPTTPGI it = oldBestTheTokens.begin(); it != oldBestTheTokens.end(); ++it)
			{
				(*it)->chosen = false;
			}
			newBestTheTokens.clear();
			const double* logActBegin = &logActs[t * outputSize];
			for (VPWI wordIt = words.begin(); wordIt != words.end(); ++wordIt)
			{
				Word* w = *wordIt;
				double bestProb = -doubleMax;
				TheToken* inputTheToken = nullTheToken;
				double inputTransProb = oneGramProbs.empty() ? 0 : -doubleMax;
				for (SPTTPGI tokenIt = oldBestTheTokens.begin(); tokenIt != oldBestTheTokens.end() && (*tokenIt)->prob > bestProb; ++tokenIt)
				{
					TheToken* tok = *tokenIt;
					double transProb = oneGramProbs.empty() ? 0 : (langModelWeight * getTransitionProb(tok->word, w));
					double newProb = tok->prob + transProb;
					if (newProb > bestProb && (fixedLength < 0 || tok->length() < fixedLength))
					{
						bestProb = newProb;
						inputTheToken = tok;
						inputTransProb = transProb;
					}
				}
				inputTheToken->chosen = true;
				TheToken* outputTheToken = Word::tokenStore.copy(w->passTheTokens(inputTheToken, inputTransProb, logActBegin, t, totalTime));
				newBestTheTokens.insert(outputTheToken);
				if (oneGramProbs.empty() && newBestTheTokens.size() > nBest)
				{
					SPTTPGI endTokIt = --newBestTheTokens.end();
					newBestTheTokens.erase(endTokIt);
					Word::tokenStore.deleteObject(*endTokIt);
				}
			} 
			for (SPTTPGI it = oldBestTheTokens.begin(); it != oldBestTheTokens.end(); ++it)
			{
				if ((*it)->chosen)
				{
					tokens.push_back(*it);
				}
				else
				{
					Word::tokenStore.deleteObject(*it);		
				}
			}
			oldBestTheTokens.clear();
		}
		if (fixedLength >= 0)
		{
			for (SPTTPGI it = newBestTheTokens.begin(); it != newBestTheTokens.end(); ++it)
			{
				if ((*it)->length() != fixedLength)
				{
					newBestTheTokens.erase(it);
					Word::tokenStore.deleteObject(*it);
				}
			}
		}
		bestTheToken = *newBestTheTokens.begin();
	}
	bestTheToken->print(wordStream);

#ifndef _WIN32
	if (sequenceDebugOutput)
	{
		seconds = time (NULL) - seconds;
		cout << (int)seconds << " seconds" << endl;
	}
#endif

	vector<int> bestWordSequence;
	stringToWordSequence(wordStream.str(), bestWordSequence);
	StringAlignment alignment(targetWordSequence, bestWordSequence);
	double labelErr = alignment.getDistance();
	double substitutions = alignment.getSubstitutions();
	double insertions = alignment.getInsertions();
	double deletions = alignment.getDeletions();

#ifndef _WIN32
	if (sequenceDebugOutput)
	{
		vector<string> bestTranscriptions;
		for (SPTTPGI it = newBestTheTokens.begin(); (int)bestTranscriptions.size() < nBest && it != newBestTheTokens.end(); ++it)
		{
			stringstream out;
			(*it)->print(out);
			if (find(bestTranscriptions.begin(), bestTranscriptions.end(), out.str()) == bestTranscriptions.end())
			{
				bestTranscriptions.push_back(out.str());
				if (nBest > 1)
				{
					cout << (int)bestTranscriptions.size() << " best" << endl;
				}
				cout << "output word string:" << endl;
				cout << out.str();
				cout << "log probability " << (*it)->prob << endl;
				vector<int> wordSeq;
				stringToWordSequence(out.str(), wordSeq);
				StringAlignment align(targetWordSequence, wordSeq);
				cout << "edit distance: " << align.getDistance() << endl;
			}
		}
	}
#endif

	//store errors in map
	int targetLength = (int)targetWordSequence.size();
	double normWordErrorRate = (double)labelErr / (double)targetLength;
    /*
	errorMap["normWordErrorRate"] += make_pair<int,double>(1, normWordErrorRate);
	errorMap["wordSubstitutions"] += make_pair<int,double>(targetLength, substitutions);
	errorMap["wordInsertions"] += make_pair<int,double>(targetLength, insertions);
	errorMap["wordDeletions"] += make_pair<int,double>(targetLength, deletions);
	errorMap["wordErrorRate"] += make_pair<int,double>(targetLength, labelErr);
	errorMap["wordSeqErrorRate"] += make_pair<int,double>(1, (labelErr > 0));
    */
}
