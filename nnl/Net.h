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

#ifndef _INCLUDED_Net_h  
#define _INCLUDED_Net_h  

#include <map>
#include <iostream>
#include <string>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/util/XMLString.hpp>
#include "DataSequence.h"
#include "ExportVal.h"

using namespace std;
XERCES_CPP_NAMESPACE_USE


#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef NET_EXPORTS
#define NET_API DLLEXPORT
#else
#define NET_API DLLIMPORT
#endif

class Net
{

public:
	
	//net operation functions
	NET_API virtual double calculateGradient(map<const string, pair<int,double> >& errorMap, const DataSequence& seq) = 0;
	NET_API virtual double calculateError(map<const string, pair<int,double> >& errorMap, const DataSequence& seq) = 0;

	//io functions
	NET_API virtual void save(const string& indent, ostream&out=cout) const = 0;
	NET_API virtual void print(ostream& out = cout) const = 0;
	NET_API virtual void outputInternalVariables(const string& path) const = 0;
	NET_API virtual void outputJacobian(const DataSequence& seq, const string& path, int timestep, int output) = 0;
	NET_API virtual void printOutputs(ostream& out = cout) = 0; 
	NET_API virtual void build() = 0;

	//constrction / destruction
	NET_API Net(){}
	NET_API virtual ~Net(){}
};

typedef vector<Net*>::iterator VPNI;

#endif
