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

#ifndef _INCLUDED_DataSequence_h  
#define _INCLUDED_DataSequence_h  

#include <vector>
#include <iostream>
#include <iterator>
#include <string>

using namespace std;

#define MIN_TARG_VAL -1e5

struct DataDimensions
{
	int numDims;
	double rmsNormFactor;
	int inputPattSize;
	int targetPattSize;
	map<const string, int> labelNumbers;	//TODO bidirectional map would be better for these 2
	vector<string> labels;
	string wordDelimiter;
};

//TODO write useful print functions
struct DataSequence
{
	//data seequence
	int size;
	vector<int> dimensions;
	const float* inputs;
	const float* targetPatterns;
	vector<int> targetClasses;
	vector<int> targetSeq;	
	vector<int> labelCounts;
	string tag;
	string targetString;
	string wordTargetString;
	DataSequence():
			size(0),
			inputs(0),
			targetPatterns(0)
	{
	}
	friend bool operator == (const DataSequence& ds1, const DataSequence& ds2)
	{
		return (ds1.size == ds2.size 
			&& ds1.dimensions == ds2.dimensions
			&& ds1.targetClasses == ds2.targetClasses 
			&& ds1.targetSeq == ds2.targetSeq
			&& equal(ds1.inputs, ds1.inputs + ds1.size, ds2.inputs)
			&& (((ds1.targetPatterns == ds2.targetPatterns) == 0) 
				||  equal(ds1.targetPatterns, ds1.targetPatterns + ds1.size, ds2.targetPatterns)));
	}

    template <class Type>
    void aux_vec_print(string name, vector<Type> &v){
        cout << name << ": [";
        for(int i=0; i<v.size(); i++){
            cout << v[i] << ((i==v.size()-1)?"":" ");
        }
        cout << "]"<<endl;
    }

    void print(){
        aux_vec_print("dimensions", dimensions);
        aux_vec_print("targetClasses", targetClasses);
        aux_vec_print("targetSeq", targetSeq);
        aux_vec_print("labelCounts", labelCounts);

        cout << "size: " << size << endl;
        cout << "tag: " << tag <<endl;
        cout << "targetString: "<< targetString << endl;
        cout << "wordTargetString: "<<wordTargetString << endl;
    }
};	

#endif
