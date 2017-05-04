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

#ifndef _INCLUDED_DataExporter_h  
#define _INCLUDED_DataExporter_h  

class DataExporter;

#include <map>
#include <string>
#include <iostream>
#include "ExportVect.h"
#include "ExportParam.h"
#include "DataExportHandler.h"

using namespace std;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef DATA_EXPORTER_EXPORTS
#define DATA_EXPORTER_API DLLEXPORT
#else
#define DATA_EXPORTER_API DLLIMPORT
#endif


class DataExporter
{

private:

	map<const string,ExportVal*> vals;
	DataExporter& operator = (const DataExporter&);//prevent assignments
	DataExporter(const DataExporter&);//prevent copies
	string name;

public:

	DataExporter(const string &n);
	virtual ~DataExporter();
	DATA_EXPORTER_API ExportVal* getVal (const string& valName);
	void save(const string& indent, ostream& out) const;
	const string& getName() const;

	template<class T> void exportVal(const string& paramName, T& p)
	{
		vals[paramName] = new ExportParam<T>(p);
	}	

	template<class T> void exportVal(const string& paramName, vector<T>& v)
	{
		vals[paramName] = new ExportVect<T>(v);
	}	
};

typedef map<const string,ExportVal*>::iterator MSPEV;
typedef map<const string,ExportVal*>::const_iterator MSPCEV;

#endif
