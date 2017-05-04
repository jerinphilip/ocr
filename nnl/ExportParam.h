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

#ifndef _INCLUDED_ExportParam_h  
#define _INCLUDED_ExportParam_h  

#include "ExportVal.h"

using namespace std;


//conversion must exist from double to T
template <class T> class ExportParam: 
		public ExportVal
{

private:

	T& param;

public:

	void print (ostream&out) const
	{
		out << param;
	}

	void save(ostream&out, const string& name, const string& indent) const
	{
		out << indent << "<Param name=\"" << name << "\" val=\"";
		out << *this << "\"/>" << endl;
	}

	double get(int index) const
	{
		return (double)param;
	}

	void set(double val, int index=0)
	{
		param = (T)val;
	}

	void set(istream& in)
	{
		in >> param;
	}

	int size() const
	{
		return 1;
	}

	ExportParam(T& p):
			param(p)
	{
	}
};

#endif
