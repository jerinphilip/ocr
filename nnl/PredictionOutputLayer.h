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

#ifndef _INCLUDED_PredictionOutputLayer_h
#define _INCLUDED_PredictionOutputLayer_h

#include "RegressionOutputLayer.h"
#include "OutputConnsImpl.h"

class PredictionOutputLayer:
		public RegressionOutputLayer,
  		protected OutputConnsImpl
{
	
private:	

	PredictionOutputLayer* firstPredictionLayer;
	Layer *inputLayer;
	bool delCumulativeActs;
	vector<double> cumulativeActs;
	vector<double> targetMean;
	
public:
	
	PredictionOutputLayer(const string& nam, vector<string>& criteria, Layer* inputLay, bool delCumActs, PredictionOutputLayer* firstPredictionLayer = 0);
	~PredictionOutputLayer();
	void updateDerivs(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts);
	void feedForward(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts);
	void feedBack(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts);
	double injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq);
	void print(ostream& out) const;
	void resizeBuffers();
	vector<double>& getActBuffer() {return actBuffer;}
	vector<double>& getErrorBuffer() {return errorBuffer;}
	const vector<double>& getCumulativePredictionActs() const {return cumulativeActs;}
	void outputInternalVariables(const string& path) const;
	
	//forwarded functions
	void addOutputConn(Connection* conn) {OutputConnsImpl::addOutputConn(conn);}
	
};

#endif
