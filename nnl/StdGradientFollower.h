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

#ifndef _INCLUDED_StdGradientFollower_h  
#define _INCLUDED_StdGradientFollower_h  

#include <algorithm>
#include "GradientFollower.h"
#include "Typedefs.h"
#include "Helpers.h"
#include "TrainableWeights.h"
#include "DataExporter.h"
#include "GradientDescentWeights.h"


class StdGradientFollower : public GradientFollower
{

private:

	DataExporter dataExporter;
	vector<GradientDescentWeights*> gdWeights;
	double learnRate;
	double momentum;

public:

	//double getRegError() const;
	int getNumWeights() const;
	void print(ostream& out = cout) const;
	void updateWeights();
	void resetDerivs();
	void randomiseWeights(double range); 
	GRADIENT_FOLLOWER_API StdGradientFollower(const string& name = "trainer");
	~StdGradientFollower();
};

typedef vector<GradientDescentWeights*>::iterator VPGDWI;
typedef vector<GradientDescentWeights*>::const_iterator VPGDWCI;

#endif
