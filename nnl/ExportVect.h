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

#ifndef _INCLUDED_ExportVect_h  
#define _INCLUDED_ExportVect_h  

#include "ExportVal.h"


template <class T> class ExportVect: 
		public ExportVal
{

private:

	vector<T>& vect;

public:

	void print(ostream&out) const
	{
		copy(vect.begin(), vect.end(), ostream_iterator<T>(out, " "));
	}

	double get(int index) const
	{
		return (double)vect.at(index);
	}

	void set(double val, int index=0)
	{
		vect.at(index) = (T)val;
	}

	void save(ostream&out, const string& name, const string& indent) const
	{
		out << indent << "<Vect name=\"" << name << "\" size=\"" << size() << "\">";
		out << *this;
		out << "</Vect>"<<endl;
	}

	void set(istream& in)
	{
		vector<T> temp;
		copy(istream_iterator<T>(in), istream_iterator<T>(), back_inserter(temp));
		if (temp.size() == vect.size())
		{
			vect.swap(temp);
		}
	}

	int size() const
	{
		return (int)vect.size();
	}

	ExportVect(vector<T>& v):
			vect(v)
	{
	}
};

#endif
