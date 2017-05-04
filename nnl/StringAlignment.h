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

#ifndef _INCLUDED_StringAlignment_h  
#define _INCLUDED_StringAlignment_h 

#include <vector>
#include <iostream>

using namespace std;

class StringAlignment {

private:

	vector< vector<int> > matrix;
	int n;
	int m;
	int subPenalty;
	int insPenalty;
	int delPenalty;
	int substitutions;
	int deletions;
	int insertions;
	int distance;
	void allocateMemory (int refSeqSize, int testSeqSize);

public:

	StringAlignment (const vector<int>& reference_sequence,const vector<int>& test_sequence,bool backtrace=true,int sp=1,int dp=1,int ip=1);
	StringAlignment (int sp = 1,int dp = 1,int ip = 1);
	int alignStrings (const vector<int>& reference_sequence,const vector<int>& test_sequence, bool backtrace);
	int getSubstitutions () const;
	int getInsertions () const;
	int getDeletions () const;
	int getDistance () const;
};

#endif
