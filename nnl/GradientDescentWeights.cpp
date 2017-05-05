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

#include "GradientDescentWeights.h"
#include "Random.h"
#include "Typedefs.h"
#include "Helpers.h"
#include "DataExporter.h"

//global debug switch
//bool sequenceDebugOutput=false;;
//global debug switch
//bool sequenceDebugOutput = false;

GradientDescentWeights::GradientDescentWeights (TrainableWeights*tw):
		trainableWeights(tw),
		wtDecay(0)
{
	DataExporter* dataExporter = DataExportHandler::instance().getExporter(trainableWeights->getName());
	dataExporter->exportVal("wtChanges", wtChanges);
	dataExporter->exportVal("wtDecay", wtDecay);
	wtChanges.resize(trainableWeights->getTrainableWeights().size());
}	

void GradientDescentWeights::updateWeights(double learnRate, double momentum)
{
	vector<double>& derivs = trainableWeights->getDerivs();
	vector<double>& weights = trainableWeights->getTrainableWeights();

#ifndef _WIN32
    /*
	if (sequenceDebugOutput)
	{
		//print max and min weights
		cout << trainableWeights->getName() << endl;
		cout << " derivs: max " << *max_element(derivs.begin(), derivs.end());
		cout << " min " << *min_element(derivs.begin(), derivs.end());
		cout << " mean " << mean<vector<double>::iterator, double>(derivs.begin(), derivs.end());
		cout << " std dev " << stdDev<vector<double>::iterator, double>(derivs.begin(), derivs.end());
		cout << " abs mean " << absMean<vector<double>::iterator, double>(derivs.begin(), derivs.end());
		cout << endl;
		cout << " prev weights: max " << *max_element(weights.begin(), weights.end());
		cout << " min " << *min_element(weights.begin(), weights.end());
		cout << " mean " << mean<vector<double>::iterator, double>(weights.begin(), weights.end());
		cout << " std dev " << stdDev<vector<double>::iterator, double>(weights.begin(), weights.end());
		cout << " abs mean " << absMean<vector<double>::iterator, double>(weights.begin(), weights.end());
		cout << endl;
	}
    */
#endif

	//apply gradient descent with momentum
	for (VDI wtChIt = wtChanges.begin(), wtIt = weights.begin(), derivIt = derivs.begin(); 
						wtChIt!=wtChanges.end(); 
						++wtChIt, ++wtIt, ++derivIt)
	{
		double deriv = *derivIt;
		if (wtDecay)
		{
			deriv -= fabs(*wtIt);
		}
		double wtChange = (momentum * (*wtChIt)) - (learnRate * deriv);
		*wtChIt = wtChange;
		*wtIt += wtChange;
	}

#ifndef _WIN32
    /*
	if (sequenceDebugOutput)
	{
		cout << " new weights: max " << *max_element(weights.begin(), weights.end());
		cout << " min " << *min_element(weights.begin(), weights.end());
		cout << " mean " << mean<vector<double>::iterator, double>(weights.begin(), weights.end());
		cout << " std dev " << stdDev<vector<double>::iterator, double>(weights.begin(), weights.end());
		cout << " abs mean " << absMean<vector<double>::iterator, double>(weights.begin(), weights.end());
		cout << endl;
	}
    */
#endif
}	


void GradientDescentWeights::randomiseWeights(double range)
{
	vector<double>& weights = trainableWeights->getTrainableWeights();

	//pick weights with gaussian distribution, std dev range 
	generate(weights.begin(), weights.end(), Random::gaussRand);
	transform(weights.begin(), weights.end(),weights.begin(), bind2nd(multiplies<double>(), range));

	//reset the weight changes
	fill(wtChanges.begin(),wtChanges.end(),0);
}	

int GradientDescentWeights::size() const
{
	return (int) wtChanges.size();
}
