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

#ifndef _INCLUDED_RpropGradientFollower_h  
#define _INCLUDED_RpropGradientFollower_h 

#include "GradientFollower.h"
#include "Typedefs.h"
#include "Helpers.h"
#include "TrainableWeights.h"
#include "DataExporter.h"

struct RpropWeights
{
	//data
	vector<double> deltas;
	vector<double> prevDerivs;
	TrainableWeights* trainableWeights;
	double wtDecay;
	
	//methods
	int size() const;
	void updateWeights(double etaPlus, double etaMin, double minDelta, double maxDelta);
	void randomiseWeights(double range, double initDelta);
	RpropWeights (TrainableWeights*tw, double initDelta);

};

class RpropGradientFollower:
		public GradientFollower
{

private:

	DataExporter dataExporter;
	vector<RpropWeights*> rpWeights;
	double etaChange;
	double etaMin;
	double etaPlus;
	double minDelta;
	double maxDelta;
	double initDelta;
	double prevAvgDelta;
	bool online;

public:

	int getNumWeights() const;
	void print(ostream& out = cout) const;
	void updateWeights();
	void resetDerivs();
	void randomiseWeights(double range); 
	GRADIENT_FOLLOWER_API RpropGradientFollower(bool online, const string& name = "trainer");
	~RpropGradientFollower();
	
};

typedef vector<RpropWeights*>::iterator VPRPWI;
typedef vector<RpropWeights*>::const_iterator VPRPWCI;

#endif
