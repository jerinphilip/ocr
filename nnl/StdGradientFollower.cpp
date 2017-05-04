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

#include "StdGradientFollower.h"


StdGradientFollower::StdGradientFollower(const string& name):
		dataExporter(name),
		learnRate(1e-4),
		momentum(0.9)
{
	dataExporter.exportVal("learnRate",learnRate);
	dataExporter.exportVal("momentum",momentum);
	for (STWI it=TrainableWeights::container.begin(); it!=TrainableWeights::container.end(); ++it)
	{
		GradientDescentWeights* gdw = new GradientDescentWeights(*it);
		gdWeights.push_back(gdw);
	}
}

StdGradientFollower::~StdGradientFollower()
{
	for_each(gdWeights.begin(), gdWeights.end(), deleteT<GradientDescentWeights>);
}

void StdGradientFollower::print(ostream& out) const
{
	dataExporter.save("", out);
	out << "total weights " << getNumWeights() << endl;
}

int StdGradientFollower::getNumWeights() const
{
	int numWeights = 0;
	for (VPGDWCI it=gdWeights.begin(); it!=gdWeights.end();++it)
	{
		numWeights += (*it)->size();
	}	
	return numWeights;
}

void StdGradientFollower::updateWeights()
{
	for (VPGDWI it=gdWeights.begin(); it!=gdWeights.end();++it)
	{
		(*it)->updateWeights(learnRate, momentum);
	}	
}

void StdGradientFollower::resetDerivs()
{
	for (VPGDWI it=gdWeights.begin(); it!=gdWeights.end();++it)
	{
		(*it)->trainableWeights->resetDerivs();
	}
}

void StdGradientFollower::randomiseWeights(double range)
{
	for_each(gdWeights.begin(),gdWeights.end(),bind2nd(mem_fun(&GradientDescentWeights::randomiseWeights),range));
}
