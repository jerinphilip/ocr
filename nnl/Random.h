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

#ifndef _INCLUDED_Random_h  
#define _INCLUDED_Random_h  

#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <limits>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef RANDOM_DLL_EXPORTS
#define RANDOM_API DLLEXPORT
#else
#define RANDOM_API DLLIMPORT
#endif


const double pi = 3.14159265358979323846;

namespace Random
{ 

	//RANDOM_API double cauchyRand();
	RANDOM_API double gaussRand();								//gaussian double, mean 0 std dev 1
	RANDOM_API double gaussRandP(double mean, double stdDev);	//gaussian double, mean mean std dev stdDev
	RANDOM_API void setRandSeed(unsigned int seed);
	RANDOM_API double flatRnd(double range);					//flat double from (-range, range)
	RANDOM_API double rnd();									//flat double from (0,1)
	RANDOM_API int randInt(int max);							//flat int from [0,max-1]
	RANDOM_API int randInt(int min,int max);					//flat int from [min,max]
	RANDOM_API int randBit();									//flat int from {0,1}

}

#endif

