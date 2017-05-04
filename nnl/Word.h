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

#ifndef _INCLUDED_Word_h  
#define _INCLUDED_Word_h  

#include <string>
#include <vector>
#include <string>
#include <map>
#include "Helpers.h"
#include "NnMath.h"

using namespace std;

class Word;

struct TheToken
{
	//data
	Word* word;
	const TheToken* prev;
	double prob;
	bool chosen;	

	//functions
	static TheToken* maxTheTokenProb (TheToken* t1, TheToken*t2);
	static bool equalWords (const TheToken* t1, const TheToken*t2);
	void reset();
	void print(ostream& out = cout);
	int length() const;
	TheToken(Word* wor = 0, TheToken* pre = 0, double pro = 0);
};

class Word
{

private:

	//data
	int totalSegments;
	int blankUnit;	
	vector<TheToken*> tokens;
	vector<TheToken*> oldTheTokens;
	TheToken* transitionTheToken;

public:
	
	//HACK workaround for gcc 3.2.3 throw/catch in constructor bug
	bool error;

	//functions
	TheToken* emitTheToken();

	//data
	static objectStore<TheToken> tokenStore;
	string word;
	vector<int> labelSeq;

	//functions
	Word(const string& definition, const map<const string, int>& labelNumbers, int bu);
	~Word();
	TheToken* initTheTokens(const vector<double>& logActs);
	TheToken* passTheTokens(const TheToken* inputTheToken, double transitionProb, const double* logActBegin, int time, int totalTime);

};

typedef vector<TheToken*>::iterator VPTI;

#endif
