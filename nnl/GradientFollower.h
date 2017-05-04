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

#ifndef _INCLUDED_GradientFollower_h  
#define _INCLUDED_GradientFollower_h  

#include <iostream>
#include "Net.h"
#include "DataSequence.h"

using namespace std;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef GRADIENT_TRAINER_EXPORTS
#define GRADIENT_FOLLOWER_API DLLEXPORT
#else
#define GRADIENT_FOLLOWER_API DLLIMPORT
#endif


class GradientFollower
{

public:	
	virtual int getNumWeights() const = 0;
//	GRADIENT_FOLLOWER_API virtual void calculateGradient(Net& net, map<const string, pair<int,double> >& errorMap, const DataSequence& seq) const = 0;
	GRADIENT_FOLLOWER_API virtual void print(ostream& out = cout) const = 0;
	GRADIENT_FOLLOWER_API virtual void updateWeights() = 0;
	GRADIENT_FOLLOWER_API virtual void resetDerivs() = 0;
	GRADIENT_FOLLOWER_API virtual void randomiseWeights(double range) = 0; 
	GRADIENT_FOLLOWER_API virtual ~GradientFollower(){};

};

#endif
