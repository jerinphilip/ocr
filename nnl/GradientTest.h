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

#ifndef _INCLUDED_GradientTest_h  
#define _INCLUDED_GradientTest_h  

#include <algorithm>
#include <numeric>
#include <math.h>
#include "Net.h"

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef GRADIENT_TEST_EXPORTS
#define GRADIENT_TEST_API DLLEXPORT
#else
#define GRADIENT_TEST_API DLLIMPORT
#endif

class GRADIENT_TEST_API GradientTest
{

private:

	//data
	Net* net;
	const DataSequence& seq;
	double perturbation;
	ostream& out;

	//functions
	void differentiateErrorWrtWeights(vector<double>& pds, vector<double>& weights);
	void printGradientDiffs(const vector<double>& numPds, const vector<double>& algPds, vector<double>& differences);

public:

	GradientTest(Net* net, const DataSequence& seq, double perturbation, ostream& out = cout);
};

#endif
