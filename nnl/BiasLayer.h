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

#ifndef _INCLUDED_BiasLayer_h  
#define _INCLUDED_BiasLayer_h  

#include "OutputConnsImpl.h"
#include "Layer.h"

class BiasLayer:
		public Layer,
		protected OutputConnsImpl
{

private:

	double act;
	string name;
	static vector<double> dummy;

public:

	//constructor / destructor
	BiasLayer();
	~BiasLayer();

	int outputSize() const;
	void getActs(const double** actBegin, const double** actEnd, int offset) const;
	const string& getName() const;

	//forwarded functions
	void addOutputConn(Connection* conn){OutputConnsImpl::addOutputConn(conn);}

	//fat interface funtions
	//TODO get rid of these
	int inputSize() const {return 1;}
	void print(ostream& out) const{}
	void getErrors(const double** errBegin, const double** errEnd, int offset) const{};
	void addInputConn(Connection* conn){};
	void outputInternalVariables(const string& path) const {}
	vector<double>& getActBuffer() {return dummy;}
	vector<double>& getErrorBuffer() {return dummy;}
};

#endif
