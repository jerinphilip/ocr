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

#include "Connection.h"
#include "NnMath.h"

#ifdef DELAY_IN_CONN
Connection::Connection(Layer* f, Layer* t, int suffixNumber, const vector<int>* delay):
		from(f),
		to(t),
		dataExporter(0)
{
	if (delay)
	{
		delayCoords = *delay;
	}
	stringstream s;
	s << from->getName() << "_to_" << to->getName()	<< "_conn";
	if (suffixNumber >= 0)
	{
		s << "_" << suffixNumber;
	}
	name = s.str().c_str();
	dataExporter = new DataExporter(name);	
	dataExporter->exportVal("weights", weights);
	to->addInputConn(this);
	from->addOutputConn(this);
}

#else
Connection::Connection(Layer* f, Layer* t, int suffixNumber):
		from(f),
		to(t),
		dataExporter(0)
{
	stringstream s;
	s << from->getName() << "_to_" << to->getName()	<< "_conn";
	if (suffixNumber >= 0)
	{
		s << "_" << suffixNumber;
	}
	name = s.str().c_str();
	dataExporter = new DataExporter(name);	
	dataExporter->exportVal("weights", weights);
	to->addInputConn(this);
	from->addOutputConn(this);
}
#endif

Connection::~Connection()
{
	delete dataExporter;
}

int Connection::getInputSize() const
{
	return from->outputSize();
}

// const vector<double>& Connection::getWeights() const
// {
// 	return weights;
// }

void Connection::resetDerivs()
{
	fill(derivs.begin(),derivs.end(),0);
}

#ifdef DELAY_IN_CONN
int Connection::getAlteredCoord(bool backwards, int coord, int dim) const
{
	if (delayCoords.empty())
	{
		return coord;
	}
	else
	{
		if (backwards)
		{
			return coord - delayCoords[dim];
		}
		else
		{
			return  coord + delayCoords[dim];
		}
	}
}

bool Connection::withinBoundary(bool backwards, const vector<int>& coords, const vector<int>& seqDims) const
{
	for (int d = 0; d < coords.size(); ++d)
	{
		int newCoord = getAlteredCoord(backwards, coords[d], d);
		if (newCoord < 0 || newCoord >= seqDims[d])
		{
			return false;
		}
	}
	return true;
}

int Connection::getDelayOffset(bool backwards, const vector<int>& coords, const vector<int>& dimProducts) const
{
	int offset = 0;
	for (int d = 0; d < coords.size(); ++d)
	{
		offset += getAlteredCoord(backwards, coords[d], d)  * dimProducts[d];
	}
	return offset;
}

void Connection::updateDerivs(const double* errBegin, const double* errEnd, const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts)
{
	if (withinBoundary(false, coords, seqDims))
	{
		updateDerivs(errBegin, errEnd, getDelayOffset(false, coords, dimProducts));
	}
}

void Connection::feedForward (double* actBegin, double* actEnd, const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts) const
{
	if (withinBoundary(false, coords, seqDims))
	{
		feedForward(actBegin, actEnd, getDelayOffset(false, coords, dimProducts));
	}
}

void Connection::feedBack (double* errBegin, double* errEnd, const vector<int>& coords, const vector<int>& seqDims, const vector<int>& dimProducts) const
{
	if (withinBoundary(true, coords, seqDims))
	{
		feedBack(errBegin, errEnd, getDelayOffset(true, coords, dimProducts));
	}
}
#endif
void Connection::updateDerivs(const double* errBegin, const double* errEnd, int offset)
{
#ifdef ZERO_INPUT_CHECK
	if (from->active(offset))
#endif
	{
		const double* inActBegin = 0;
		const double* inActEnd = 0;
		from->getActs(&inActBegin, &inActEnd, offset);
		matrixMultVect(inActBegin, inActEnd, &derivs.front(), errBegin, errEnd);
	}
}

void Connection::feedForward (double* actBegin, double* actEnd, int offset) const
{
#ifdef ZERO_INPUT_CHECK
	if (from->active(offset))
#endif
	{	
		const double* inActBegin = 0;
		const double* inActEnd = 0;
		from->getActs(&inActBegin, &inActEnd, offset);
        //cerr << "[debug] [Connection] "<<weights.size()<<endl;
		matrixAddMult(actBegin, actEnd, &weights.front(), inActBegin, inActEnd);
	}
}

void Connection::feedBack (double* errBegin, double* errEnd, int offset) const
{
#ifdef ZERO_INPUT_CHECK
	if (from->active(offset))
#endif
	{
		const double* outErrBegin = 0;
		const double* outErrEnd = 0;
		to->getErrors(&outErrBegin, &outErrEnd, offset);
		matrixAddMultTranspose(errBegin, errEnd, &weights.front(), outErrBegin, outErrEnd);
	}
}

void Connection::build()
{
	int numWts = (int)from->outputSize() * (int)to->inputSize();
	weights.resize(numWts);
	derivs.resize(numWts);
}

void Connection::print(ostream& out) const
{
	out << "Connection " << name;
	out << " (" << (int)weights.size() << " wts" << ")";
#ifdef DELAY_IN_CONN
	if (!delayCoords.empty())
	{
		out << " (delay coords ";
		copy(delayCoords.begin(), delayCoords.end(), ostream_iterator<int>(out, " "));
		out << ")";
	}
#endif
	out << endl;
}

const string& Connection::getName() const
{
	return name;
}

vector<double>& Connection::getTrainableWeights()
{
	return weights;
}

vector<double>& Connection::getDerivs()
{
	return derivs;
}
