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

#ifndef _INCLUDED_NDimLayer_h  
#define _INCLUDED_NDimLayer_h  

#include "Layer.h"
#include "InputConnsImpl.h"
#include "ActivationFunctions.h"
#include "Helpers.h"

//TODO get rid of NDim names. All layers, nets, connections etc to be n dimensional by default
//TODO get rid of NDimLayer entirely: put everything in level
class NDimLayer: 
		public Layer
{

public:

	virtual void save(ostream& out) const = 0;
	virtual void resizeBuffers (int size) = 0;
	virtual	void build() = 0;
	virtual void printInputConns(ostream& out) const = 0;
#ifdef DELAY_IN_CONN 
	virtual void feedForward( const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps) = 0;
	virtual void feedBack(const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps) = 0;
	virtual void updateDerivs(const vector<int>& coords, const vector<int>& dimensions, const vector<int>& actSteps) = 0;
#endif
	virtual void feedForward(int offset, const vector<int>& coords, const vector<int>& dimensions, 
							const vector<int>& actSteps, int overlap) = 0;
	virtual void feedBack(int offset, const vector<int>& coords, const vector<int>& dimensions, 
							const vector<int>& actSteps, int overlap) = 0;
	virtual void updateDerivs(int offset, const vector<int>& coords, const vector<int>& dimensions, 
							const vector<int>& actSteps, int overlap) = 0;
	static bool onBoundary(bool reverse, const vector<int>& steps, const vector<int>& coords, 
		const vector<int>& dimensions, int dim, int overlap)
	{
		if (overlap)
		{
			for (int d = 0; d < dim; ++d)
			{
				int overlapD = ((steps[d] < 0) ? overlap : -overlap);
				if (reverse)
				{
					overlapD = -overlapD;
				}
				int adjCoord = coords[d] + overlapD;
				if (adjCoord < 0 || adjCoord >= dimensions[d])
				{
					return true;
				}
			}
		}
		int step = reverse ? -steps[dim] : steps[dim];
		int coord = coords[dim];
		//int adjCoord = coords[dim] + jump;
		//return (adjCoord < 0 || adjCoord >= dimensions[dim]);
		return (((step < 0) && (coord == 0)) || ((step > 0) && (coord == (dimensions[dim] - 1))));
		//bool bdry = (((step < 0) && (coord == 0)) || ((step > 0) && (coord == (dimension - 1))));
		//return bdry;
	}
};

#endif

