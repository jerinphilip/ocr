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

#ifndef _INCLUDED_NetcdfDataset_h  
#define _INCLUDED_NetcdfDataset_h  

#include <functional>
#include <algorithm>
#include <string>
#include <sstream>
#include <numeric>
#include <map>
#include "DataSequence.h"


using namespace std;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef NETCDF_LIB_EXPORTS
#define NETCDF_LIB_API DLLEXPORT
#else
#define NETCDF_LIB_API DLLIMPORT
#endif

//TODO: compatibility test for netcdf datasets
class NetcdfDataset
{

private:

	//data
	DataDimensions dataDimensions;
	vector<DataSequence*> dataSequences;
	int numTargets;
	int numInputs;
	int* seqLengths;
	float* inputs;
	float* targets;
	int* targetClasses;

	//TODO replace seqLengths with seqDims
	int* seqDims;
	int totalTimesteps;
	int totalTargetStringLength;
	vector<string> seqTags;
	vector<string> targetStrings;
	vector<string> wordTargetStrings;

	//hierarchical ctc stuff
	vector<map<const string, int> > hLabelNumbers;
	vector<vector<string> > hTargetStrings;
	vector<vector<string> > hLabels;

	//functions
	void buildData (int numSeqs);

public:

	NETCDF_LIB_API NetcdfDataset(const string& filename, int seqNum);
	NETCDF_LIB_API NetcdfDataset(const string& filename, double fraction = 1);
	NETCDF_LIB_API ~NetcdfDataset();
	NETCDF_LIB_API const DataSequence& getDataSequence(int seqNum) const;
	NETCDF_LIB_API const DataDimensions& getDimensions() const;
	NETCDF_LIB_API void print(ostream& out = cout) const;
	NETCDF_LIB_API int getTotalTargetStringLength() const;
	NETCDF_LIB_API int getNumSeqs() const;
	NETCDF_LIB_API int getNumTimesteps() const;

};

#endif
