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

#include "NetcdfFile.h"
#include <stdlib.h>



//TODO get rid of the redundancy in these functions
NetcdfFile::NetcdfFile(const string& filename)
{
	//try to open the file
	int status;
	status = nc_open(filename.c_str(), NC_NOWRITE, &fileId);
	if (status != NC_NOERR) 
	{
		cerr << "GroupNC file " << filename << " cannot be opened" << endl;
		cerr << "status = " << nc_strerror(status)<< endl;
		exit(0);
	}
}

NetcdfFile::~NetcdfFile()
{
	nc_close(fileId);
}

int NetcdfFile::getNDims(const string& name) const
{
	//try to get the variable 
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	if (status != NC_NOERR)  
	{
		cerr << "NetcdfFile::getDims " << name << " not found in File " << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		return 0;
	}

	//try to get the dimensions
	nc_type variableType;                 
	int numDims;                      
	int* dims = new int[NC_MAX_VAR_DIMS];     
	int numAttributes;                      
	status = nc_inq_var (fileId, variableId, 0, &variableType, &numDims, dims, &numAttributes);
	delete dims;
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile::getNDims Could not get attributes of " << variableId << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		return 0;
	}
	else
	{
		return numDims;
	}
}

int* NetcdfFile::getDims(int variableId) const
{
	//try to get the dimensions
	nc_type variableType;                 
	int numDims;                      
	int* dims = new int[NC_MAX_VAR_DIMS];     
	int numAttributes;                      
	int status = nc_inq_var (fileId, variableId, 0, &variableType, &numDims, dims, &numAttributes);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile::getDims Could not get attributes of " << variableId << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		return 0;
	}

	//replace dimids with dimension sizes (which we really want)
	size_t dimSize;
	for (int i=0; i<numDims; ++i) 
	{
		status = nc_inq_dimlen(fileId, dims[i], &dimSize);
		if (status != NC_NOERR) 
		{
			cerr << "NetcdfFile::getDims could not get length of dimension for variable " << variableId << endl;
			cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		}
		else
		{
			dims[i] = (int)dimSize;
		}
	}
	return dims;
}


int* NetcdfFile::getDims(const string& name) const
{
	//try to get the variable 
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	if (status != NC_NOERR)  
	{
		cerr << "NetcdfFile::getDims " << name << " not found in File " << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		return 0;
	}
	//get the dimensions
	else
	{
		return getDims(variableId);
	}
}

int NetcdfFile::getRows(const string& name) const
{
	int* dims = getDims(name);
	int rows = dims[0];
	delete dims;
	return rows;
}

int NetcdfFile::getCols(const string& name) const
{
	int ndims = getNDims(name);
	if (ndims < 2)
	{
		return 1;
	}
	else 
	{
		int* dims = getDims(name);
		int cols = dims[1];
		delete dims;
		return cols;
	}
}

bool NetcdfFile::contains(const string& name) const
{
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	return status == NC_NOERR;
}

int NetcdfFile::getInt(const string& name) const
{
	//try to get it as a variable 
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	
	//if the variable doesn't exist, try to get it as a dimension
	if (status != NC_NOERR)  
	{ 
		int dimensionId;
		status = nc_inq_dimid (fileId, name.c_str(), &dimensionId);
		if (status !=NC_NOERR) 
		{ 
			cerr << "NetcdfFile::getInt int " << name << " not found in file" << endl;
			cerr << "Status: " << status << " " << nc_strerror(status) << endl;
			exit(0);
		}
		else
		{
			size_t dimensionSize;
			status = nc_inq_dimlen(fileId, dimensionId, &dimensionSize);
			if (status != NC_NOERR) 
			{
				cerr << "NetcdfFile::getInt unable to get dimension length " << name << endl;
				cerr << "status: " << status << " " << nc_strerror(status) << endl;
				exit(0);
			} 
			else 
			{
				return (int)dimensionSize;
			}
		}
    }
	else
	{
		int val[1];
		status = nc_get_var_int(fileId, variableId, val);
		if (status != NC_NOERR) 
		{
			cerr << "NetcdfFile::getInt could not read variable " << name << endl;
			cerr << "Status: " << status << " " << nc_strerror(status) << endl;
			exit(0);
		}
		else
		{
			return val[0];
		}
	}
}

int* NetcdfFile::loadIntArr(const string& name, int size, int offset) const
{
	//try to get the variable 
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile: loadIntArr " << name << " not found" << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}

	//get the array size
	int numCols = getCols(name);
	int numRows = size ? size : getRows(name);
	size_t start[] = {offset, 0};
	size_t count[] = {numRows, numCols};

	//load in the array
	int* val = new int [numRows*numCols];
	status = nc_get_vara_int(fileId, variableId, start, count, val);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile::loadIntArr could not read variable " << name << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}
	return val;
}

float* NetcdfFile::loadFloatArr(const string& name, int size, int offset) const
{
	//try to get the variable 
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile: loadFloatArr " << name << " not found" << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}

	//get the array size
	int numCols = getCols(name);
	int numRows = size ? size : getRows(name);
	size_t start[] = {offset, 0};
	size_t count[] = {numRows, numCols};

	//load in the array
	float* val = new float [numRows*numCols];
	status = nc_get_vara_float(fileId, variableId, start, count, val);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile::loadFloatArr could not read variable " << name << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}
	return val;
}

void NetcdfFile::loadStrings(vector<string>& strings, const string& name, int size, int offset) const
{
	//try to get the variable 
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile: loadStrings " << name << " not found" << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}

	//get the array size
	int numCols = getCols(name);
	int numRows = size ? size : getRows(name);
	size_t start[] = {offset, 0};
	size_t count[] = {numRows, numCols};

	//load in the array
	char* val = new char [numRows*numCols];
	status = nc_get_vara_text(fileId, variableId, start, count, val);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile::loadStrings could not read variable " << name << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}
	strings.clear();
	for (int i=0; i<numRows; ++i)
	{
		strings.push_back(val + (numCols*i));
	}
	delete val;
}

void NetcdfFile::loadString(const string& name, string& dest) const
{
	//try to get the variable 
	int variableId;
	int status = nc_inq_varid (fileId, name.c_str(), &variableId);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile: loadString " << name << " not found" << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}

	//get the array size
	int size = getRows(name);

	//load in the array
	dest.resize(size);
	char *val = new char [size];
	nc_get_var_text(fileId, variableId, val);
	if (status != NC_NOERR) 
	{
		cerr << "NetcdfFile::loadString could not read variable " << name << endl;
		cerr << "Status: " << status << " " << nc_strerror(status) << endl;
		exit(0);
	}
	dest = val;
	delete val;
}
