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

#define GRADIENT_TRAINER_EXPORTS

#include <numeric>
#include "Random.h"
#include "RpropGradientFollower.h"


//FLORIANS PARAMS
// double d_0 = 0.01;         // delta_0
// double d_min = 1e-9;       // delta_min – must be small enough to allow for precise adaption
// double d_max = 0.2;        // delta_max – must be a lot smaller in order to prevent instability
// double eta_plus = 1.2;     //
// double eta_minus = 0.5;    //

RpropGradientFollower::RpropGradientFollower(bool onlin, const string& name):
		dataExporter(name),
		online(onlin),
		prevAvgDelta(0),
		etaMin(0.5),
		etaPlus(1.2),
		minDelta(1e-9),
		maxDelta(0.2),
		initDelta(0.01),
		etaChange(0.01)
{
	dataExporter.exportVal("etaMin",etaMin);
	dataExporter.exportVal("etaPlus",etaPlus);
	dataExporter.exportVal("minDelta",minDelta);
	dataExporter.exportVal("maxDelta",maxDelta);
	dataExporter.exportVal("initDelta",initDelta);
	if (online)
	{
		dataExporter.exportVal("etaChange",etaChange);
		dataExporter.exportVal("prevAvgDelta",prevAvgDelta);
	}
	
	for (STWI it=TrainableWeights::container.begin(); it!=TrainableWeights::container.end(); ++it)
	{
		RpropWeights* rpw = new RpropWeights(*it, initDelta);
		rpWeights.push_back(rpw);
	}
}

RpropGradientFollower::~RpropGradientFollower()
{
	for_each(rpWeights.begin(), rpWeights.end(), deleteT<RpropWeights>);
}

void RpropGradientFollower::print(ostream& out) const
{
	out << "RPROP gradient follower" << endl;
	dataExporter.save("", out);
	out << "total weights " << getNumWeights() << endl;
}

int RpropGradientFollower::getNumWeights() const
{
	int numWeights = 0;
	for (VPRPWCI it=rpWeights.begin(); it!=rpWeights.end();++it)
	{
		numWeights += (*it)->size();
	}	
	return numWeights;
}

void RpropGradientFollower::updateWeights()
{
	for (VPRPWCI it=rpWeights.begin(); it!=rpWeights.end();++it)
	{
		(*it)->updateWeights(etaPlus, etaMin, minDelta, maxDelta);
	}
	
	//use eta adaptations for online training (from Mike Schuster's thesis)
	if (online)	
	{
		double avgDelta = 0;
		for (VPRPWCI it=rpWeights.begin(); it!=rpWeights.end(); ++it)
		{
			avgDelta += accumulate((*it)->deltas.begin(), (*it)->deltas.end(), 0.0);
		}
		avgDelta /= getNumWeights();
		if (avgDelta > prevAvgDelta)
		{
			etaPlus = max (1.0, etaPlus - etaChange);
		}
		else
		{
			etaPlus += etaChange;
		}
		prevAvgDelta = avgDelta;
	}
}

void RpropGradientFollower::randomiseWeights(double range)
{
	for (VPRPWCI it=rpWeights.begin(); it!=rpWeights.end();++it)
	{
		(*it)->randomiseWeights(range, initDelta);
	}
	prevAvgDelta = initDelta;
}

void RpropGradientFollower::resetDerivs()
{
	for (VPRPWI it=rpWeights.begin(); it!=rpWeights.end();++it)
	{
		(*it)->trainableWeights->resetDerivs();
	}
}

RpropWeights::RpropWeights (TrainableWeights*tw, double initDelta):
		trainableWeights(tw),
		wtDecay(0)
{
	DataExporter* dataExporter = DataExportHandler::instance().getExporter(trainableWeights->getName());
	dataExporter->exportVal("deltas", deltas);
	dataExporter->exportVal("prevDerivs", prevDerivs);
	dataExporter->exportVal("wtDecay", wtDecay);
	int numWts = trainableWeights->getTrainableWeights().size();
	deltas.resize(numWts, initDelta);
	prevDerivs.resize(numWts, 0);
}	

void RpropWeights::updateWeights(double etaPlus, double etaMin, double minDelta, double maxDelta)
{
	vector<double>& derivs = trainableWeights->getDerivs();
	vector<double>& weights = trainableWeights->getTrainableWeights();
	for (VDI wtIt = weights.begin(), derivIt = derivs.begin(), prevDerivIt = prevDerivs.begin(), deltaIt = deltas.begin(); 
		    wtIt!=weights.end(); 
		    ++wtIt, ++derivIt, ++prevDerivIt, ++deltaIt)
	{
		double deriv = *derivIt;
		if (wtDecay)
		{
			deriv -= fabs(*wtIt);
		}
		double derivTimesPrevDeriv =  deriv * *prevDerivIt;
		if(derivTimesPrevDeriv > 0)
		{
			*deltaIt = constrain(minDelta, maxDelta, *deltaIt * etaPlus);
			*wtIt -= sign(deriv) * *deltaIt;
			*prevDerivIt = deriv;
		}
		else if(derivTimesPrevDeriv < 0)
		{
			*deltaIt = constrain(minDelta, maxDelta, *deltaIt * etaMin);
			*prevDerivIt = 0;
		}
		else
		{
			*wtIt -= sign(deriv) * *deltaIt;
			*prevDerivIt = deriv;
		}
	}
}

void RpropWeights::randomiseWeights(double range, double initDelta)
{
	vector<double>& weights = trainableWeights->getTrainableWeights();

	//pick weights with gaussian distribution, std dev range, mean 0
	generate(weights.begin(), weights.end(), Random::gaussRand);
	transform(weights.begin(), weights.end(),weights.begin(), bind2nd(multiplies<double>(), range));

	//reset the weight changes
	fill(deltas.begin(),deltas.end(),initDelta);
	fill(prevDerivs.begin(),prevDerivs.end(),0);
}	

int RpropWeights::size() const
{
	return (int) deltas.size();
}
