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

#ifndef _INCLUDED_DataExportHandler_h  
#define _INCLUDED_DataExportHandler_h  

class DataExportHandler;

#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/util/XMLString.hpp>
#include <sstream>
#include "DataExporter.h"

using namespace std;
XERCES_CPP_NAMESPACE_USE

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define DLLIMPORT __declspec(dllimport)
#else 
#define DLLEXPORT 
#define DLLIMPORT
#endif

#ifdef DATA_EXPORT_HANDLER_EXPORTS
#define DATA_EXPORT_HANDLER_API DLLEXPORT
#else
#define DATA_EXPORT_HANDLER_API DLLIMPORT
#endif

//singleton class: only one instance possible
class DataExportHandler  
{

private:

	map <const string,DataExporter*> dataExporters;

protected:

	DataExportHandler(){}
	DataExportHandler(const DataExportHandler&){}
	DataExportHandler& operator= (const DataExportHandler&){}

public: 

	DATA_EXPORT_HANDLER_API void save(const string& indent, ostream& out=cout) const;
	void addExporter(const string& name, DataExporter* exporter);
	void remExporter(const string& name);
	DATA_EXPORT_HANDLER_API DataExporter* getExporter(const string& name);
	DATA_EXPORT_HANDLER_API void getExportVals(vector<ExportVal*>& vals, const string& valName);
	DATA_EXPORT_HANDLER_API void load(const DOMElement* dataElement);
	DATA_EXPORT_HANDLER_API static DataExportHandler& instance();
};


typedef map<const string, DataExporter*>::iterator MSPDEI;
typedef map<const string, DataExporter*>::const_iterator MSPDECI;

#endif
