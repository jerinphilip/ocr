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

#ifndef _INCLUDED_Connection_h  
#define _INCLUDED_Connection_h  

class Connection;

#include "TrainableWeights.h"
#include "Layer.h"
#include "DataExporter.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <sstream>

// #define DELAY_IN_CONN
class Connection: 
		public TrainableWeights
{

private:

	DataExporter *dataExporter;
	vector<double> weights;
	vector<double> derivs;
	string name;
	
#ifdef DELAY_IN_CONN
	vector<int> delayCoords;
	bool withinBoundary(bool backwards, const vector<int>& coords, const vector<int>& seqDims) const;
	int getDelayOffset(bool backwards, const vector<int>& coords, const vector<int>& dimProducts) const;
	int getAlteredCoord(bool backwards, int coord, int dim) const;
#endif
	
public:

	Layer* from;
	Layer* to;
#ifdef DELAY_IN_CONN
	Connection(Layer* f, Layer* t, int suffixNumber = -1, const vector<int>* delay = 0);
#else
	Connection(Layer* f, Layer* t, int suffixNumber = -1);
#endif
	virtual ~Connection();
	const string& getName() const;
	vector<double>& getTrainableWeights();
	vector<double>& getDerivs();
	int getInputSize() const;
// 	const vector<double>& getWeights() const;
	void resetDerivs();
#ifdef DELAY_IN_CONN
	void updateDerivs(const double* errBegin, const double* errEnd, const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts);
	void feedForward (double* actBegin, double* actEnd, const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts) const;
	void feedBack (double* errBegin, double* errEnd, const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts) const;
#endif
	void updateDerivs(const double* errBegin, const double* errEnd, int offset);
	void feedForward (double* actBegin, double* actEnd, int offset) const;
	void feedBack (double* errBegin, double* errEnd, int offset) const;
	void build();
	void print(ostream& out) const;

};

typedef vector<Connection*>::iterator VPBCI;
typedef vector<Connection*>::const_iterator VPBCCI;

#endif
