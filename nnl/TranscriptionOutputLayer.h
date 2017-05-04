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

#ifndef _INCLUDED_TranscriptionOutputLayer_h  
#define _INCLUDED_TranscriptionOutputLayer_h  

#include "OutputLayer.h"
#include "InputConnsImpl.h"
#include "WordDecoder.h"
#include "Helpers.h"
#include "DataExporter.h"


class TranscriptionOutputLayer: 
		public OutputLayer,
		protected InputConnsImpl
{

private:

	//data
	int size;
	int blankUnit;
	int totalSegments;
	vector<double> dEdYTerms;
	vector<double> errorBuffer;
	vector<double> inActBuffer;
	vector<double> actBuffer;
	vector<double> logActBuffer;
	vector<string> labels;
	string name;
	vector<vector<double> > forwardVariables;
	vector<vector<double> > backwardVariables;
	WordDecoder* wordDecoder;
	bool contiguous;
	DataExporter dataExporter;
	
	//functions
	void getStringFromPath(vector<int> &str,const vector<int>& path);

public:			


	void getMostProbableString(vector<int>& str);

	TranscriptionOutputLayer(const string& nam, const vector<string>& lab, int numDims, 
					vector<string>& criteria, const map<const string, int>& labelNumbers, 
					const string& wordDelimiter, const DOMElement* element);
	~TranscriptionOutputLayer();
	void print(ostream& out) const;
	const string& getName() const;
	void getActs(const double** actBegin, const double** actEnd, int offset) const;
	void getErrors(const double** errBegin, const double** errEnd, int offset) const;
    vector<string> getLabels();
	int inputSize() const;
	int outputSize() const;
#ifdef DELAY_IN_CONN
	void updateDerivs(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts);
#endif
	void updateDerivs(int offset);
	double injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq);
	void feedForward(const vector<int>& seqDims, int seqLength, const DataSequence& seq);
	void outputInternalVariables(const string& path) const;
	void setErrToOutputDeriv(int timestep, int output);
	vector<double>& getActBuffer() {return actBuffer;}
	vector<double>& getErrorBuffer() {return errorBuffer;}
	void printOutputs(ostream& out) {printBufferToStreamT(actBuffer, out, size);}
	
	//forwarded functions
	void build() {InputConnsImpl::build();}
	void printInputConns(ostream& out) const {InputConnsImpl::printInputConns(out);}
	void addInputConn(Connection* conn) {InputConnsImpl::addInputConn(conn);}

	//null functions
	void addOutputConn(Connection* conn) {}

};

#endif
