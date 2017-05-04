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

#ifndef _INCLUDED_BuffInputConnsImpl_h  
#define _INCLUDED_BuffInputConnsImpl_h  

#include <vector>
#include <iostream>
#include <algorithm>
#include "Connection.h"

using namespace std;


class InputConnsImpl
{

protected:

	vector <Connection*> inputs;

public:
#ifdef DELAY_IN_CONN
	void updateDerivs(const double* errBegin, const double* errEnd, const vector<int>& coords, 
			  const vector<int>& seqDims, const vector<int>& dimProducts) const;
#endif
	void updateDerivs(const double* errBegin, const double* errEnd, int offset) const;
	void build();
	void addInputConn(Connection* conn);
	void printInputConns(ostream& out) const;
	virtual ~InputConnsImpl();
};

#endif
