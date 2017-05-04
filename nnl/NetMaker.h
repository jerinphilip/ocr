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

#ifndef _INCLUDED_NetMaker_h  
#define _INCLUDED_NetMaker_h 

#include "Net.h"

using namespace std;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef NET_MAKER_EXPORTS
#define NET_MAKER_API DLLEXPORT
#else
#define NET_MAKER_API DLLIMPORT
#endif


class NetMaker
{

public:

	NET_MAKER_API static Net* makeNet (const DOMElement* element, const DataDimensions& dims, vector<string>& criteria);

};



#endif
