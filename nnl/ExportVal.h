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

#ifndef _INCLUDED_ExportVal_h  
#define _INCLUDED_ExportVal_h  

#include <vector>
#include <iostream>
#include <iterator>

using namespace std;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef EXPORT_VAL_EXPORTS
#define EXPORT_VAL_API DLLEXPORT
#else
#define EXPORT_VAL_API DLLIMPORT
#endif


class ExportVal
{

public:

	EXPORT_VAL_API virtual void print(ostream& out) const = 0;
	virtual void save(ostream&out, const string& name, const string& indent) const = 0;
	EXPORT_VAL_API virtual double get(int index) const = 0;
	EXPORT_VAL_API virtual void set(double val, int index=0) = 0;
	EXPORT_VAL_API virtual void set(istream& in) = 0;
	EXPORT_VAL_API virtual int size() const = 0;
	virtual ~ExportVal(){}

};

static ostream& operator << (ostream& out, const ExportVal& v)
{	
	v.print(out);
	return out;
}

#endif
