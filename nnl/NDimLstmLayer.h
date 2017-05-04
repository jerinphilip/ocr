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

#ifndef _INCLUDED_LstmLayer_h  
#define _INCLUDED_LstmLayer_h  

#include <fstream>
#include <map>
#include "NDimLayer.h"
#include "DataExporter.h"
#include "ActivationFunctions.h"
#include "OutputConnsImpl.h"
#include "InputConnsImpl.h"
#include "NnMath.h"
#include "Typedefs.h"
#include "Random.h"

using namespace std;

#define PEEPS

struct PeepholeWeights: 
		public TrainableWeights
{
	DataExporter dataExporter;
	string name;
	vector<double> weights;
	vector<double> derivs;
	vector<double>& getTrainableWeights() 
	{
		return weights;
	}
	vector<double>& getDerivs() 
	{
		return derivs;
	}
	void resetDerivs() 
	{
		fill(derivs.begin(), derivs.end(), 0);
	}
	const string& getName() const 
	{
		return name;
	}
	void print(ostream& out) const
	{
		out << "LSTM PeepholeWeights \"" << name << "\"";
		out << " (" << (int)weights.size() << " wts";
		out << ")";
		out << endl;
	}
	PeepholeWeights(const string& n, int size):
			name(n),
			dataExporter(n)
	{
        /* Weights are exported here? */
		dataExporter.exportVal("weights", weights);
		weights.resize(size);
		derivs.resize(size);
	}
};

template <class CI, class CO, class G> class NDimLstmLayer: 
		public NDimLayer,
		protected OutputConnsImpl,
		protected InputConnsImpl
{

private:

	DataExporter dataExporter;
	string name;
#ifdef PEEPS
	PeepholeWeights* peepholes;
#endif
	int peepWtsPerBlock;
	vector<double> inGateActBuffer;
	vector<double> forgetGateActBuffer;
	vector<double> outGateActBuffer;
	vector<double> preOutGateActBuffer;
	vector<double> stateBuffer;
	vector<double> preGateStateBuffer;
	vector<double> actBuffer;
	vector<double> errorBuffer;
	vector<double> cellErrorBuffer;
	vector<double> cellOutErrorBuffer;
	vector<double> inputActs;
	string actFn;
	int cellsPerBlock;
	int unitsPerBlock;
	int numBlocks;
	int peepsPerBlock;
	int numCells;
	int totalSize;
	int numDims;
	bool constrainErrors;
	vector<int> seqDims;
	
public:

	NDimLstmLayer(const string& nam, const string& actf, int size, int nDims):
			name(nam),
			actFn(actf),
			numBlocks(size),
			cellsPerBlock(1),
			peepsPerBlock(0),
			numDims(nDims),
			constrainErrors(true),
			dataExporter(nam)
	{
		dataExporter.exportVal("constrainErrors", constrainErrors);

		//init topology
		int gatesPerBlock = 2 + numDims;
		unitsPerBlock = gatesPerBlock + cellsPerBlock;
		numCells = numBlocks * cellsPerBlock;
		totalSize = unitsPerBlock * numBlocks;

		//resize arrays
		//cellOutErrors.resize(numCells);
		inputActs.resize(totalSize);

		//initialize peephole weights
		peepsPerBlock = cellsPerBlock * gatesPerBlock;
#ifdef PEEPS
		peepholes = new PeepholeWeights(name + "_peepholes", peepsPerBlock * numBlocks);
#endif
	}

	~NDimLstmLayer()
	{
#ifdef PEEPS
		delete peepholes;
#endif
	}

	void resizeBuffers(int buffSize)
	{
		inGateActBuffer.resize(buffSize * numBlocks);
		forgetGateActBuffer.resize(buffSize * numBlocks * numDims);
		outGateActBuffer.resize(buffSize * numBlocks);
		preOutGateActBuffer.resize(buffSize * numCells);
		stateBuffer.resize(buffSize * numCells);
		preGateStateBuffer.resize(buffSize * numCells);
		actBuffer.resize(buffSize * numCells);
		errorBuffer.resize(buffSize * totalSize);
		cellErrorBuffer.resize(buffSize * numCells);
		cellOutErrorBuffer.resize(buffSize * numCells);

		//needed for block errors
		fill(cellOutErrorBuffer.begin(), cellOutErrorBuffer.end(), 0);
	}

	//HACK to give access to buffers to block transformer
	//TODO get rid of these
	vector<double>& getActBuffer()
	{
		return actBuffer;
	}

	vector<double>& getErrorBuffer()
	{
		return cellOutErrorBuffer;
	}

	void outputInternalVariables(const string& path) const
	{
		printBufferToFile(inGateActBuffer, path + getName() + "_inGateActs", numBlocks, &seqDims);
		printBufferToFile(forgetGateActBuffer, path + getName() + "_forgetGateActs", numBlocks * numDims, &seqDims);
		printBufferToFile(outGateActBuffer, path + getName() + "_outGateActs", numBlocks, &seqDims);
		printBufferToFile(preOutGateActBuffer, path + getName() + "_preOutGateActs", numCells, &seqDims);
		printBufferToFile(stateBuffer, path + getName() + "_states", numCells, &seqDims);
		printBufferToFile(preGateStateBuffer, path + getName() + "_preGateStates", numCells, &seqDims);
		printBufferToFile(actBuffer, path + getName() + "_activations", numCells, &seqDims);
		printBufferToFile(errorBuffer, path + getName() + "_errors", totalSize, &seqDims);
		printBufferToFile(cellErrorBuffer, path + getName() + "_cellErrors", numCells, &seqDims);
	}
#ifdef DELAY_IN_CONN
	void feedForward(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts)
	{
	}
	
	void feedBack(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts)
	{
	}

	void updateDerivs(const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts) 
	{
	}
#endif
	//TODO rewrite this in a sensible order: [all input gates, all forget gates, etc]
	void feedForward(int offset, const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps, int overlap)
	{
		seqDims = dimensions;
		if (offset < 0 || offset * numCells >= (int)actBuffer.size())
		{
			cerr << "NDimLstmLayer::feedForward offset " << offset << " out of range, exiting" <<endl;
			exit(0);
		}
		double* actBegin = &actBuffer[offset * numCells];
		double* outGateActBegin = &outGateActBuffer[offset * numBlocks];
		double* inGateActBegin = &inGateActBuffer[offset * numBlocks];
		double* stateBegin = &stateBuffer[offset * numCells];
		double* preGateStateBegin = &preGateStateBuffer[offset * numCells];
		double* preOutGateActBegin = &preOutGateActBuffer[offset * numCells];
		double* forgetGateActBegin = &forgetGateActBuffer[offset * numBlocks * numDims];

		//collect inputs from the net
		double* inputActBegin = &*inputActs.begin();
		double* inputActEnd = &*inputActs.end();
		fill(inputActBegin, inputActEnd, 0);
		int d = 0;
		for (VPBCCI inputIt = inputs.begin(); inputIt != inputs.end(); ++inputIt)
		{
			if ((*inputIt)->from == this)
			{
				int actStep = actSteps[d];
				if (!onBoundary(false, actSteps, coords, dimensions, d, overlap))
				{
					(*inputIt)->feedForward(inputActBegin, inputActEnd, offset + actStep);
				}
				++d;
			}
			else
			{
				(*inputIt)->feedForward(inputActBegin, inputActEnd, offset);
			}
		}

		//feed them forward into the blocks
		double* inActIt = inputActBegin;
#ifdef PEEPS
		const double* peepWtIt = &peepholes->weights.front();
#endif
		for (int b = 0; b < numBlocks; ++b)	
		{
			int cellStart = b * cellsPerBlock;
			int cellEnd = cellStart + cellsPerBlock;
#ifdef PEEPS
			//input gate
			//extra inputs from peepholes (from old states)
			for (int d = 0; d < (int)dimensions.size(); ++d)
			{
				int actStep = actSteps[d];
				if (!onBoundary(false, actSteps, coords, dimensions, d, overlap))
				{
					const double* oldStateBegin = &stateBuffer[(offset + actStep) * numCells];
					matrixAddMult(inActIt, inActIt + 1, peepWtIt, oldStateBegin + cellStart, oldStateBegin + cellEnd);
				}
			}
			peepWtIt += cellsPerBlock;
#endif	
			double inGateAct = G::fn(*inActIt);
			inGateActBegin[b] = inGateAct;
			++inActIt;

			//forget gates
			//extra inputs from peepholes (from old states)	
			for (int d = 0; d < (int)dimensions.size(); ++d)
			{
#ifdef PEEPS
				int actStep = actSteps[d];
				if (!onBoundary(false, actSteps, coords, dimensions, d, overlap))
				{
					const double* oldStateBegin = &stateBuffer[(offset + actStep) * numCells];
					matrixAddMult(inActIt, inActIt + 1, peepWtIt, oldStateBegin + cellStart, oldStateBegin + cellEnd);
				}
				peepWtIt += cellsPerBlock;
#endif
				forgetGateActBegin[(b*numDims) + d] = G::fn(*inActIt);
				++inActIt;
			}

			//pre-gate cell states
			transform(inActIt, inActIt + cellsPerBlock, preGateStateBegin + cellStart, CI::fn);
			inActIt += cellsPerBlock;

			//cell states
			for (int c = cellStart; c < cellEnd; ++c)
			{
				double state = inGateAct * preGateStateBegin[c];
				for (int d = 0; d < (int)dimensions.size(); ++d)
				{
					int actStep = actSteps[d];
					if (!onBoundary(false, actSteps, coords, dimensions, d, overlap))
					{
						double forgetGateAct = forgetGateActBegin[(b * numDims) + d];
						double oldState = stateBuffer[((offset + actStep) * numCells) + c];
						state += forgetGateAct * oldState;
					}
				}
				stateBegin[c] = state;
				preOutGateActBegin[c] = CO::fn(state);
			}

			//output gate
			//extra input from peephole (from current state)
#ifdef PEEPS			
			matrixAddMult(inActIt, inActIt + 1, peepWtIt, stateBegin + cellStart, stateBegin + cellEnd);
			peepWtIt += cellsPerBlock;
#endif

			double outGateAct = G::fn(*inActIt);
			outGateActBegin[b] = outGateAct;
			++inActIt;

			//output activations
			transform(preOutGateActBegin + cellStart, preOutGateActBegin + cellEnd, actBegin + cellStart, 
				bind2nd(multiplies<double>(), outGateAct));
		}
	}

	void feedBack(int offset, const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps, int overlap)
	{
		if (offset < 0 || offset * totalSize >= (int)errorBuffer.size())
		{
			cerr << "NDimLstmLayer::feedBack offset " << offset << " out of range, exiting" <<endl;
			exit(0);
		}

		//activations
		const double* outGateActBegin = &outGateActBuffer[offset * numBlocks];
		const double* inGateActBegin = &inGateActBuffer[offset * numBlocks];
		const double* preGateStateBegin = &preGateStateBuffer[offset * numCells];
		const double* preOutGateActBegin = &preOutGateActBuffer[offset * numCells];
		const double* forgetGateActBegin = &forgetGateActBuffer[offset * numBlocks * numDims];
		
		//errors
		double* cellErrorBegin = &cellErrorBuffer[offset * numCells];
		double* errorBegin = &errorBuffer[offset * totalSize];
		double* errorEnd = errorBegin + totalSize;

		//collect errors from net
		//double* cellOutErrorBegin = &*cellOutErrors.begin();
		//double* cellOutErrorEnd = &*cellOutErrors.end();
		//fill(cellOutErrorBegin, cellOutErrorEnd, 0);
		double* cellOutErrorBegin = &cellOutErrorBuffer[offset * numCells];
		double* cellOutErrorEnd = cellOutErrorBegin + numCells;
		int d = 0;
		for (VPBCCI outputIt = outputs.begin(); outputIt != outputs.end(); ++outputIt)
		{
			if ((*outputIt)->to == this)
			{
				int actStep = -actSteps[d];
				if (!onBoundary(true, actSteps, coords, dimensions, d, overlap))
				{
					(*outputIt)->feedBack(cellOutErrorBegin, cellOutErrorEnd, offset + actStep);
				}
				++d;
			}
			else
			{
				(*outputIt)->feedBack(cellOutErrorBegin, cellOutErrorEnd, offset);
			}
		}

		//feed them back through the blocks
		double* errorIt = errorBegin;
#ifdef PEEPS
		const double* peepWtIt = &peepholes->weights.front();
#endif
		for (int b = 0; b < numBlocks; ++b)	
		{
			int cellStart = b * cellsPerBlock;
			int cellEnd = cellStart + cellsPerBlock;
			double inGateAct = inGateActBegin[b];
			double outGateAct = outGateActBegin[b];

			//output gate error
			double outGateError = G::deriv(outGateAct) * 
					inner_product(preOutGateActBegin + cellStart, preOutGateActBegin + cellEnd, 
						cellOutErrorBegin + cellStart, 0.0);

			//cell pds (dE/dState)
			for (int c = cellStart; c < cellEnd; ++c)
			{
				double deriv = (CO::deriv(preOutGateActBegin[c]) * outGateAct * cellOutErrorBegin[c]/*cellOutErrors[c]*/);
#ifdef PEEPS
				int cOffset = c - cellStart;
				double igPeepWt = peepWtIt[cOffset];
				double ogPeepWt = peepWtIt[peepsPerBlock - cellsPerBlock + cOffset];
				deriv += outGateError * ogPeepWt;
#endif
				for (int d = 0; d < (int)dimensions.size(); ++d)
				{
#ifdef PEEPS
					double fgPeepWt = peepWtIt[cOffset + (cellsPerBlock * (d + 1))];
#endif
					int actStep = -actSteps[d];
					if (!onBoundary(true, actSteps, coords, dimensions, d, overlap))
					{
#ifdef PEEPS
						double nextFGError = errorBuffer[((offset + actStep) * totalSize) + (b * unitsPerBlock) + 1 + d];
						double nextIgError = errorBuffer[((offset + actStep) * totalSize) + (b * unitsPerBlock)];
						deriv += (nextFGError * fgPeepWt) + (nextIgError * igPeepWt);
#endif
						double nextFGAct = forgetGateActBuffer[((offset + actStep) * numBlocks * numDims) + (b * numDims) + d];
						double nextCellError = cellErrorBuffer[((offset + actStep) * numCells) + c];
						deriv += (nextFGAct * nextCellError);

					}
				}
				cellErrorBegin[c] = deriv;
			}

			//input gate error
			*errorIt = G::deriv(inGateAct) * 
					inner_product(cellErrorBegin + cellStart, cellErrorBegin + cellEnd, preGateStateBegin + cellStart, 0.0);
			++errorIt;

			//forget gate error
			for (int d = 0; d < (int)dimensions.size(); ++d, ++errorIt)
			{
				int actStep = actSteps[d];
				if (onBoundary(false, actSteps, coords, dimensions, d, overlap))
				{
					*errorIt = 0;
				}
				else
				{
					double forgetGateAct = forgetGateActBegin[(b * numDims) + d];
					const double* oldStateBegin = &stateBuffer[(offset + actStep) * numCells];
					*errorIt = G::deriv(forgetGateAct) * 
							inner_product(cellErrorBegin + cellStart, cellErrorBegin + cellEnd, oldStateBegin + cellStart, 0.0);
				}
			}

			//cell errors
			for (int c = cellStart; c < cellEnd; ++c)
			{
				*errorIt = inGateAct * CI::deriv(preGateStateBegin[c]) * cellErrorBegin[c];
				++errorIt;
			}
			*errorIt = outGateError;	
			++errorIt;
#ifdef PEEPS
			peepWtIt += peepsPerBlock;
#endif
		}

		//constrain errors to be in [-1,1] for stability
		if (constrainErrors)
		{
			for (double* errIt = errorBegin; errIt != errorEnd; ++errIt)
			{
				if (*errIt > 1.0)
				{
					*errIt = 1.0;
				}
				else if (*errIt < -1.0)
				{
					*errIt = -1.0;
				}
			}
		}
	}

	void updateDerivs(int offset, const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps, int overlap)
	{
		if (offset < 0 || offset * totalSize >= (int)errorBuffer.size())
		{
			cerr << "NDimLstmLayer::updateDerivs offset " << offset << " out of range, exiting" <<endl;
			exit(0);
		}

#ifdef PEEPS
		const double* stateBegin = &stateBuffer[offset * numCells];
		const double* errorBegin = &errorBuffer[offset * totalSize];

		double* pdIt = &peepholes->derivs.front();
		for (int b = 0; b < numBlocks; ++b, pdIt += peepsPerBlock)
		{
			int cellStart = b * cellsPerBlock;
			int cellEnd = cellStart + cellsPerBlock;
			int errorOffset = unitsPerBlock * b;
			double inGateError = errorBegin[errorOffset];
			for (int d = 0; d < (int)dimensions.size(); ++d)
			{
				int actStep = actSteps[d];
				if (!onBoundary(false, actSteps, coords, dimensions, d, overlap))
				{
					const double* oldStates = &stateBuffer[((offset + actStep) * numCells)];

					//in gate
					for (int c = cellStart; c < cellEnd; ++c)
					{
						pdIt[c - cellStart] += inGateError * oldStates[c];
					}
					double forgGateError = errorBegin[errorOffset + d + 1];
					for (int c = cellStart; c < cellEnd; ++c)
					{
						pdIt[(c - cellStart) + ((d + 1) * cellsPerBlock)] += forgGateError * oldStates[c];
					}
				}
			}

			double outGateError = errorBegin[errorOffset + unitsPerBlock - 1];
			for (int c = cellStart; c < cellEnd; ++c)
			{
				pdIt[(c - cellStart) + peepsPerBlock - cellsPerBlock] += outGateError * stateBegin[c];
			}
		}
#endif
		const double* errBegin = &errorBuffer[offset * totalSize];
		const double* errEnd = errBegin + totalSize;
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
	int inputSize() const
	{
		return totalSize;
	}

	int outputSize() const
	{
		return numCells;
	}

	void getErrors(const double** errBegin, const double** errEnd, int offset) const
	{
		int totalOffset = offset * totalSize;
		if (offset < 0 || totalOffset >= (int)errorBuffer.size())
		{
			cerr << "NDimLstmLayer::getErrors offset " << offset << " out of range, exiting" <<endl;
			exit(0);
		}
		*errBegin = &errorBuffer[totalOffset];
		*errEnd = *errBegin + totalSize;
	}

	void getActs(const double** actBegin, const double** actEnd, int offset) const
	{
		int totalOffset = offset * numCells;
		if (offset < 0 || totalOffset >= (int)actBuffer.size())
		{
			cerr << "NDimLstmLayer::getActs offset " << offset << " out of range, exiting" <<endl;
			exit(0);
		}
		*actBegin = &actBuffer[totalOffset];
		*actEnd = *actBegin + numCells;
	}

	//TODO get rid of this
	void save(ostream& out) const
	{
		out << "<Lstm";
		out << " size=\"" << numBlocks << "\"";
		out << " actFn=\"" << actFn << "\"";
		out << "/>";
	}

	void print(ostream& out) const
	{
		save(out);
		out << endl;
		dataExporter.save("", out);
		out << endl;
	}

	void printInputConns(ostream& out) const 
	{
		InputConnsImpl::printInputConns(out);
#ifdef PEEPS
		peepholes->print(out);
#endif
	}


	//forwarded functions
	void build() {InputConnsImpl::build();}
	void addInputConn(Connection* conn) {InputConnsImpl::addInputConn(conn);}
	void addOutputConn(Connection* conn) {OutputConnsImpl::addOutputConn(conn);}
	const string& getName() const {return name;}
};


#endif
