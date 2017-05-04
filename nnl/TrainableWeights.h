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

#ifndef _INCLUDED_TrainableWeights_h  
#define _INCLUDED_TrainableWeights_h  

#include <vector>
#include <set>
#include "NamedObject.h"

using namespace std;


//TODO replace this with access to one big wt vector
class TrainableWeights:
		public NamedObject
{

public:
	
	static set<TrainableWeights*> container;

	TrainableWeights()
	{
		container.insert(this);
	}
	
	virtual ~TrainableWeights() 
	{
		container.erase(this);
	}

	virtual vector<double>& getTrainableWeights() = 0;
	virtual vector<double>& getDerivs() = 0;
	virtual void resetDerivs() = 0;
};

typedef set<TrainableWeights*>::iterator STWI;
typedef vector<TrainableWeights*>::iterator VTWI;
typedef vector<TrainableWeights*>::const_iterator VTWCI;

#endif
