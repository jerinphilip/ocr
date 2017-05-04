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

#ifndef _INCLUDED_WordDecoder_h  
#define _INCLUDED_WordDecoder_h  

#include <fstream>
#include "Word.h"
#include "DataExporter.h"
#include "Helpers.h"
#include "DataSequence.h"

#define N_BEST

struct token_prob_greater : public binary_function<const TheToken*, const TheToken*, const bool> 
{
	bool operator()(const TheToken* t1, const TheToken* t2) const
	{
		return t1->prob > t2->prob; 
	}
};

class WordDecoder
{

private:

	//data
	multimap <const string, int> wordIndices;
	map <const string, int> labelStrIndices;
	vector <Word*> words;
	TheToken* nullTheToken;
	vector<TheToken*> tokens;
	string wordDelimiter;
	const vector<string>& labels;
	int outputSize;
	double meanBigramProb;
	double langModelWeight;
	map <const Word*, pair<double, double> > oneGramProbs;
	map <pair<const Word*, const Word*>, double> biGramProbs;
	multiset<TheToken*, token_prob_greater> oldBestTheTokens;
	multiset<TheToken*, token_prob_greater> newBestTheTokens;
	int fixedLength;
	int nBest;

	//functions
	void stringToWordSequence(const string& str, vector<int>& wordSequence);
	double getTransitionProb(const Word* from, const Word* to);
	//void wordToLabelSequence(const vector<int>& wordSequence, vector<int>& labelSequence);
	void labelToWordSequence(const vector<int>& labelSequence, vector<int>& wordSequence);

public:

	void decode(/*vector<int>& outputSeq,*/ map<const string, pair<int,double> >& errorMap, 
					const DataSequence& seq, const vector<double>& logActs);
	WordDecoder(ifstream& dictFile, const map<const string, int>& labelNumbers, int blankUnit, 
					const vector<string>& lab, const string& wd, int os, const string& bigramFilename,
					int nb, int fl);
	~WordDecoder();
};

typedef vector<Word*>::iterator VPWI;
typedef vector<Word*>::const_iterator VPWCI;
typedef map <pair<const Word*, const Word*>, double>::const_iterator MPWWDCI;
typedef map <const Word*, pair<double, double> >::const_iterator MPWDDCI;
typedef multiset<TheToken*, token_prob_greater>::iterator SPTTPGI;
typedef multimap <const string, int>::const_iterator MMCSICI;

#endif
