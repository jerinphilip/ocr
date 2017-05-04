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

#ifndef _INCLUDED_NetcdfFile_h  
#define _INCLUDED_NetcdfFile_h  

#include<vector>
#include<iostream>
#include<string>
#include<netcdf.h>

using namespace std;


class NetcdfFile
{

private:
	
	//data
	int fileId;

	//functions
	int* getDims(int variableId) const;
	int* getDims(const string& name) const;
	int getRows(const string& name) const;
	int getCols(const string& name) const;
	int getNDims(const string& name) const;

public:

	bool contains(const string& name) const;
	int getInt(const string& name) const;
	int* loadIntArr(const string& name, int size = 0, int offset = 0) const;
	float* loadFloatArr(const string& name, int size = 0, int offset = 0) const;
	void loadStrings(vector<string>& strings, const string& name, int size = 0, int offset = 0) const;
	void loadString(const string& name, string& dest) const;
	NetcdfFile(const string& filename);
	~NetcdfFile(void);
};

#endif
