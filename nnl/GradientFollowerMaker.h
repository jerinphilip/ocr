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

#ifndef _INCLUDED_GradientFollowerMaker_h  
#define _INCLUDED_GradientFollowerMaker_h 

#include <string>
#include "GradientFollower.h"

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef TRAINER_MAKER_EXPORTS
#define TRAINER_MAKER_API DLLEXPORT
#else
#define TRAINER_MAKER_API DLLIMPORT
#endif


class GradientFollowerMaker
{

public:
	
	TRAINER_MAKER_API static GradientFollower* makeStdGradientFollower(const string& name = "trainer");
	TRAINER_MAKER_API static GradientFollower* makeRpropGradientFollower(bool online, const string& name = "trainer");

};

#endif

