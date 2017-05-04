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

#ifndef _INCLUDED_BpttBackpropLayer_h  
#define _INCLUDED_BpttBackpropLayer_h  

#include <fstream>
#include <string>
#include "NDimLayer.h"
#include "ActivationFunctions.h"
#include "OutputConnsImpl.h"
#include "InputConnsImpl.h"
#include "Typedefs.h"
#include "Helpers.h"

using namespace std;

//TODO rename Backprop to something more descriptive (RNN? NeuronLayer?)
template <class T> class NDimBackpropLayer:
		public NDimLayer,
		protected OutputConnsImpl,
		protected InputConnsImpl
{

private:

	int size;
	string name;
	string actFn;
	vector<double> actBuffer;
	vector<double> errorBuffer;
	vector<int> seqDims;
	
public:

	NDimBackpropLayer (const string& nam, const string& actf, int siz):
			size(siz),
			name(nam),
			actFn(actf)
	{
	}

	//HACK to give access to buffers to block transformer
	//TODO get rid of these
	vector<double>& getActBuffer()
	{
		return actBuffer;
	}

	vector<double>& getErrorBuffer()
	{
		return errorBuffer;
	}

	void outputInternalVariables(const string& path) const
	{
		printBufferToFile(actBuffer, path + getName() + "_activations", size, &seqDims);
		printBufferToFile(errorBuffer, path + getName() + "_errors", size, &seqDims);
	}

	void resizeBuffers(int buffSize)
	{
		int totalSize = buffSize * size;
		actBuffer.resize(totalSize);
		errorBuffer.resize(totalSize);

		//needed for block errors
		fill(errorBuffer.begin(), errorBuffer.end(), 0);
	}

	//TODO get rid of this
	void save(ostream& out) const
	{
		out << "<Backprop size=\"" << size << "\"";
		out << " actFn=\"" << actFn << "\"/>";
	}

	void print(ostream& out) const
	{
		save(out);
		out << endl;
	}
	
	int inputSize() const
	{
		return size;
	}

	int outputSize() const
	{
		return size;
	}

	void getErrors(const double** errBegin, const double** errEnd, int offset) const
	{
		int totalOffset = offset * size;
		if (offset < 0 || totalOffset >= (int)errorBuffer.size())
		{
			cerr << "NDimBackpropLayer::getErrors offset " << offset << " out of range, exiting" <<endl;
			exit(0);
		}
		else
		{
			*errBegin = &errorBuffer[totalOffset];
			*errEnd = *errBegin + size;
		}
	}

	void getActs(const double** actBegin, const double** actEnd, int offset) const
	{
		int totalOffset = offset * size;
		if (offset < 0 || totalOffset >= (int)actBuffer.size())
		{
			cerr << "NDimBackpropLayer::getActs layer name " << name << " offset " << offset << " out of range, exiting" <<endl;
			exit(0);
		}
		else
		{
			*actBegin = &actBuffer[totalOffset];
			*actEnd = *actBegin + size;
		}
	}

#ifdef DELAY_IN_CONN
	
	//TODO put range check on offset in next 3 functions
	void feedForward(const vector<int>& coords, const vector<int>& dims, const vector<int>& dimProducts)
	{
		seqDims = dims;
		int offset = inner_product(coords.begin(), coords.end(), dimProducts.begin(), 0);
		double* actBegin = &actBuffer[offset * size];
		double* actEnd = actBegin + size;
		fill(actBegin, actEnd, 0);

		//feed forward inputs from the net
		for (VPBCCI inputIt = inputs.begin(); inputIt != inputs.end(); ++inputIt)
		{
			(*inputIt)->feedForward(actBegin, actEnd, coords, seqDims, dimProducts);
		}

		//apply activation function
		transform(actBegin, actEnd, actBegin, T::fn);
	}

	void feedBack(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts)
	{
		int offset = inner_product(coords.begin(), coords.end(), dimProducts.begin(), 0);
		double* errBegin = &errorBuffer[offset * size];
		double* errEnd = errBegin + size;

		//feed back errors from the Net
		for (VPBCCI outputIt = outputs.begin(); outputIt != outputs.end(); ++outputIt)
		{
			(*outputIt)->feedBack(errBegin, errEnd, coords, seqDims, dimProducts);
		}

		//feed errors through activation function
		double* actIt = &actBuffer[offset * size];
		for (double* errIt = errBegin; errIt != errEnd; ++errIt, ++actIt)
		{
			*errIt = T::deriv(*actIt) * (*errIt);
		}
	}

	void updateDerivs(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts) 
	{
		int offset = inner_product(coords.begin(), coords.end(), dimProducts.begin(), 0);
		const double* errBegin = &errorBuffer[offset * size];
		const double* errEnd = errBegin + size;
		for (VPBCCI inputIt = inputs.begin(); inputIt != inputs.end(); ++inputIt)
		{
			(*inputIt)->updateDerivs(errBegin, errEnd, coords, seqDims, dimProducts);
		}
	}
#endif
	void feedForward(int offset, const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps, int overlap)
	{
		seqDims = dimensions;
		double* actBegin = &actBuffer[offset * size];
		double* actEnd = actBegin + size;
		fill(actBegin, actEnd, 0);

		//feed forward inputs from the net
		int d = 0;
		for (VPBCCI inputIt = inputs.begin(); inputIt != inputs.end(); ++inputIt)
		{
			if ((*inputIt)->from == this)
			{
				int actStep = actSteps[d];
				if (!onBoundary(false, actSteps, coords, dimensions, d, overlap))
				{
					(*inputIt)->feedForward(actBegin, actEnd, offset + actStep);
				}
				++d;
			}
			else
			{
				(*inputIt)->feedForward(actBegin, actEnd, offset);
			}
		}

		//apply activation function
		transform(actBegin, actEnd, actBegin, T::fn);
	}

	void feedBack(int offset, const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps, int overlap)
	{
		double* errBegin = &errorBuffer[offset * size];
		double* errEnd = errBegin + size;

		//feed back errors from the Net
		int d = 0;
		for (VPBCCI outputIt = outputs.begin(); outputIt != outputs.end(); ++outputIt)
		{
			if ((*outputIt)->to == this)
			{
				int actStep = -actSteps[d];
				if (!onBoundary(true, actSteps, coords, dimensions, d, overlap))
				{
					(*outputIt)->feedBack(errBegin, errEnd, offset + actStep);
				}
				++d;
			}
			else
			{
				(*outputIt)->feedBack(errBegin, errEnd, offset);
			}
		}

		//feed errors through activation function
		double* actIt = &actBuffer[offset * size];
		for (double* errIt = errBegin; errIt != errEnd; ++errIt, ++actIt)
		{
			*errIt = T::deriv(*actIt) * (*errIt);
		}
	}

	void updateDerivs(int offset, const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps, int overlap) 
	{
		const double* errBegin = &errorBuffer[offset * size];
		const double* errEnd = errBegin + size;
		int d = 0;
		for (VPBCCI inputIt = inputs.begin(); inputIt != inputs.end(); ++inputIt)
		{
			if ((*inputIt)->from == this)
			{
				int actStep = actSteps[d];
				if (!onBoundary(false, actSteps, coords, dimensions, d, overlap))
				{
					(*inputIt)->updateDerivs(errBegin, errEnd, offset + actStep);
				}
				++d;
			}
			else
			{
				(*inputIt)->updateDerivs(errBegin, errEnd, offset);
			}
		}
	}

	const string& getName() const 
	{
		return name;
	}

	//forwarded functions
	void build() {InputConnsImpl::build();}
	void printInputConns(ostream& out) const {InputConnsImpl::printInputConns(out);}
	void addInputConn(Connection* conn) {InputConnsImpl::addInputConn(conn);}
	void addOutputConn(Connection* conn) {OutputConnsImpl::addOutputConn(conn);}

};

#endif
