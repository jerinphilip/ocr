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

#define GRADIENT_TEST_EXPORTS

#include "GradientTest.h"
#include "TrainableWeights.h"
#include "DataExportHandler.h"
#include "Typedefs.h"

//TODO get rid of this (make error map optional for inject errors)
static map<const string, pair<int,double> > dummyErrorMap;

GradientTest::GradientTest(Net* n, const DataSequence& s, double p, ostream& o):
		net(n),
		seq(s),
		perturbation(p),
		out(o)
{
	//set constrain errors to false
	vector<ExportVal*> constrainErrVals;
	DataExportHandler::instance().getExportVals(constrainErrVals, "constrainErrors");
	for (vector<ExportVal*>::iterator it = constrainErrVals.begin(); it != constrainErrVals.end(); ++it)
	{
		(*it)->set(0);
	}

	//calculate numerical pds
	cout << "calculating numerical pds" << endl;
	vector<vector<double> > numPds(TrainableWeights::container.size());
	VVDI numPdIt = numPds.begin();
	for (STWI it=TrainableWeights::container.begin(); it!= TrainableWeights::container.end(); ++it, ++numPdIt)
	{
		vector<double>& weights = (*it)->getTrainableWeights();
		differentiateErrorWrtWeights(*numPdIt, weights);
	}

	//calculate algorithmic pds
	cout << "calculating algorithmic pds" << endl;
	net->calculateGradient(dummyErrorMap, seq);

	//print comparison
	numPdIt = numPds.begin();
	vector<double> differences;
	double overallMaxDiff = 0;
	for (STWI it=TrainableWeights::container.begin(); it!= TrainableWeights::container.end(); ++it, ++numPdIt)
	{
		out << (*it)->getName() << endl;
		differences.resize(numPdIt->size());
		printGradientDiffs(*numPdIt, (*it)->getDerivs(), differences);
		double maxDiff = *max_element(differences.begin(), differences.end());
		double minDiff = *min_element(differences.begin(), differences.end());
		double meanDiff = accumulate(differences.begin(), differences.end(), 0.0) / (double)differences.size();
		out << "max diff " << maxDiff << "\tmin diff " << minDiff << "\tmean diff " << meanDiff << endl;
		out << endl;
		if (maxDiff > overallMaxDiff)
		{
			overallMaxDiff = maxDiff;
		}
	}
	out << "overall max diff " << overallMaxDiff << endl << endl;
}

void GradientTest::differentiateErrorWrtWeights(vector<double>& pds, vector<double>& weights)
{
	for (VDI wtIt=weights.begin(); wtIt!=weights.end(); ++wtIt)
	{
		//store original weight
		double oldWt = *wtIt;

		//add positive perturbation and compute error
		*wtIt = oldWt + perturbation;
		double plusErr = net->calculateError(dummyErrorMap, seq);

		//add negative perturbation and compute error
		*wtIt = oldWt - perturbation;
		double minusErr = net->calculateError(dummyErrorMap, seq);

		//store symmetric difference
		pds.push_back((plusErr-minusErr)/(2*perturbation));

		//restore original weight
		*wtIt = oldWt;
	}
}

void GradientTest::printGradientDiffs(const vector<double>& numPds, const vector<double>& algPds, vector<double>& differences)
{
	out << "numerical pds" << endl;
	copy(numPds.begin(), numPds.end(), ostream_iterator<double>(out, " "));
	out << endl;
	out << "algorithmic pds" << endl;
	copy(algPds.begin(), algPds.end(), ostream_iterator<double>(out, " "));
	out << endl;
	for (int i = 0; i < (int)numPds.size(); ++i)
	{
		differences[i] = fabs(numPds[i] - algPds[i]);
	}
}

