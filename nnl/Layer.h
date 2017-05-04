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

#ifndef _INCLUDED_Layer_h  
#define _INCLUDED_Layer_h  

class Layer;

#include<vector>
#include<iostream>
#include<string>
#include"Connection.h"


#define ZERO_INPUT_CHECK

using namespace std;

class Layer
{

public:

	virtual ~Layer() {}
	virtual void print(ostream& out) const = 0;
	
#ifdef ZERO_INPUT_CHECK
	virtual bool active(int offset) {return true;}
#endif

	//TODO replace this with dataDisplayHandler, allow outputting by layer/variable name etc
	//Should be possible to combine with export handler (register variable)
	virtual void outputInternalVariables(const string& path) const = 0;

	//from layer
	virtual int outputSize() const = 0;

	virtual void getActs(const double** actBegin, const double** actEnd, int offset) const = 0;
	virtual void addOutputConn(Connection* conn) = 0;

	//to layer
	virtual int inputSize() const = 0;
	virtual void getErrors(const double** errBegin, const double** errEnd, int offset) const = 0;
	virtual void addInputConn(Connection* conn) = 0;

	//named object
	virtual const string& getName() const = 0;
	
	//HACK to give block transformer access to layer buffers
	//TODO get rid of these
	virtual vector<double>& getActBuffer() = 0;
	virtual vector<double>& getErrorBuffer() = 0;
	
	//HACK for prediction output layers to access cumulative acts
	virtual const vector<double>& getCumulativePredictionActs() const {static vector<double> dummy; return dummy;}

};

#endif
