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

#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "Word.h"
#include "NnMath.h"
#include "Typedefs.h"

objectStore<TheToken> Word::tokenStore;

Word::Word(const string& definition, const map<const string, int>& labelNumbers, int bu):
		blankUnit(bu)
{
	stringstream ss;
	ss << definition;
	ss >> word;
	string label;
	while (ss >> label)
	{
		const map<const string, int>::const_iterator it = labelNumbers.find(label);
		if (it == labelNumbers.end())
		{
			stringstream s;
			s <<"WARNING label '" << label << "' found in definition " << definition << " but not in labelNumbers";
			cerr << s.str() << endl;
			error = true;
			return;
			//throw runtime_error (s.str());
		}
		else
		{
			error = false;
			labelSeq.push_back(it->second);
		}
	}
	totalSegments = (2*(int)labelSeq.size()) + 1;
	for (int i = 0; i < totalSegments; ++i)
	{
		tokens.push_back(new TheToken(this, 0, -doubleMax));
		oldTheTokens.push_back(new TheToken(this, 0, -doubleMax));
	}
	transitionTheToken = new TheToken(this);
}

Word::~Word()
{
	for_each(tokens.begin(), tokens.end(), deleteT<TheToken>);
	for_each(oldTheTokens.begin(), oldTheTokens.end(), deleteT<TheToken>);
}

TheToken* Word::emitTheToken()
{
	return TheToken::maxTheTokenProb (tokens[totalSegments-1], tokens[totalSegments-2]);
}

TheToken* Word::initTheTokens(const vector<double> & logActs)
{
	//initialise start tokens (blank and first segment)
	oldTheTokens[0]->prob = logActs[blankUnit];
	if (oldTheTokens.size() > 1)
	{
		oldTheTokens[1]->prob = logActs[labelSeq[0]];
	}
	for (int i = 0; i < (int)tokens.size(); ++i)
	{
		tokens[i]->prev = 0;
		tokens[i]->prob = -doubleMax;
		oldTheTokens[i]->prev = 0;
		if (i > 1)
		{
			oldTheTokens[i]->prob = -doubleMax;
		}
	}

	//return output segment
	return emitTheToken();
}

//TODO put in CONTIGUOUS switch
TheToken* Word::passTheTokens(const TheToken* inputTheToken, double transitionProb, const double* logActBegin, int time, int totalTime)
{
	//copy the external token to an updated transition token
	transitionTheToken->prob = inputTheToken->prob + transitionProb;
	transitionTheToken->prev = inputTheToken;

	//loop over the allowed segments, passing the tokens forwards
	int startSegment = max(0, totalSegments - (2 *(totalTime-time)));
	int endSegment = min(totalSegments, 2 * (time + 1));
	for (int s = startSegment; s < endSegment; ++s)
	{
		TheToken* testTheToken;
		if (s == 0 || s == 1)
		{
			testTheToken = TheToken::maxTheTokenProb (transitionTheToken, oldTheTokens[s]);
		}
		else
		{
			testTheToken = oldTheTokens[s];
		}
		if (s)
		{
			testTheToken = TheToken::maxTheTokenProb (testTheToken, oldTheTokens[s-1]);
		}
		if (s&1)
		{
			if ((s > 1) && (labelSeq[s/2] != labelSeq[(s/2)-1]))
			{
				testTheToken = TheToken::maxTheTokenProb (testTheToken, oldTheTokens[s-2]);
			}
		}
		TheToken* token = tokens[s];
		token->prob = testTheToken->prob;
		token->prev = testTheToken->prev;
		if (s&1)
		{
			token->prob += logActBegin[labelSeq[s/2]];
		}
		else
		{			
			token->prob += logActBegin[blankUnit];
		}
	}
	for (int i = 0; i < (int)tokens.size(); ++i)
	{
		oldTheTokens[i]->prev = tokens[i]->prev;
		oldTheTokens[i]->prob = tokens[i]->prob;
	}
	return emitTheToken();
}

TheToken::TheToken(Word* wor, TheToken* pre, double pro):
		word(wor),
		prev(pre),
		prob(pro),
		chosen(false)
{
}

TheToken* TheToken::maxTheTokenProb (TheToken* t1, TheToken*t2)
{
	return t1->prob > t2->prob ? t1 : t2;
}

void TheToken::print(ostream& out)
{
	list<const Word*> words;
	const TheToken* t = this;
	while (t && t->word)
	{
		words.push_front(t->word);
		t = t->prev;
	}
	for (list<const Word*>::iterator it = words.begin(); it != words.end(); ++it)
	{
		out << (*it)->word << " ";
	}
	out << endl;
}

int TheToken::length() const
{
	int l = 0;
	for (const TheToken* t = this; t; t=t->prev, ++l);
	return l;
}

bool TheToken::equalWords (const TheToken* t1, const TheToken*t2)
{
	for (;t1 && t2; t1 = t1->prev, t2 = t2->prev)
	{
		if (t1->word->word != t2->word->word)
		{
			return false;
		}
	}
	return !(t1 || t2);
}

