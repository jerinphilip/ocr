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

#define DATA_EXPORT_HANDLER_EXPORTS

#include "DataExportHandler.h"

//DataExportHandler DataExportHandler::dataHandler;

DATA_EXPORT_HANDLER_API DataExportHandler& DataExportHandler::instance()
{
	static DataExportHandler inst;
	return inst;
}

void DataExportHandler::addExporter(const string& name, DataExporter* exporter)
{
	dataExporters[name] = exporter;
}

void DataExportHandler::getExportVals(vector<ExportVal*>& vals, const string& valName)
{
	for (MSPDECI it=dataExporters.begin(); it!=dataExporters.end() ; ++it)
	{
		ExportVal* ev = it->second->getVal(valName);
		if (ev)
		{
			vals.push_back(ev);
		}
	}
}

DataExporter* DataExportHandler::getExporter(const string& name)
{
	MSPDEI it = dataExporters.find(name);
	if (it != dataExporters.end())
	{
		return it->second;
	}
	else
	{
		return 0;
	}
}

void DataExportHandler::remExporter(const string& name)
{
	dataExporters.erase(name);
}

void DataExportHandler::save(const string& indent, ostream& out) const
{
	out << indent << "<ExportedData>" << endl;
	for (MSPDECI it=dataExporters.begin(); it!=dataExporters.end() ; ++it)
	{
		out << indent << "\t<Object name=\"" << it->first << "\">" << endl;
		it->second->save(indent + "\t\t", out);
		out << indent << "\t</Object>" << endl;
	}
	out << indent << "</ExportedData>" << endl;
}

void DataExportHandler::load(const DOMElement* dataElement)
{
    /* Gather all Objects */
	const DOMNodeList* objectNodes = dataElement->getElementsByTagName(XMLString::transcode("Object"));
	int numObjects = objectNodes->getLength();
	for (int i = 0; i < numObjects; ++i)
	{
		const DOMElement* objectElement = (DOMElement*)objectNodes->item(i);
		const DOMNamedNodeMap* objAttributes = objectElement->getAttributes();
		const string objName = XMLString::transcode(objAttributes->getNamedItem(XMLString::transcode("name"))->getNodeValue());
		DataExporter* d = getExporter(objName);
		if (d)
		{
			const DOMNodeList* paramNodes = objectElement->getElementsByTagName(XMLString::transcode("Param"));
			int numParams = paramNodes->getLength();
			for (int j = 0; j < numParams; ++j)
			{	
				const DOMNode* paramNode = paramNodes->item(j);
				DOMNamedNodeMap* parAttributes = paramNode->getAttributes();
				string parName = XMLString::transcode(parAttributes->getNamedItem(XMLString::transcode("name"))->getNodeValue());
				double val = atof(XMLString::transcode(parAttributes->getNamedItem(XMLString::transcode("val"))->getNodeValue()));
				ExportVal* p = d->getVal(parName);
				if (p)
				{
					p->set(val);
				}
			}

			const DOMNodeList* vectNodes = objectElement->getElementsByTagName(XMLString::transcode("Vect"));
			int numVects = vectNodes->getLength();
			for (int k = 0; k < numVects; ++k)
			{	
				const DOMNode* VectNode = vectNodes->item(k);
				DOMNamedNodeMap* vectAttributes = VectNode->getAttributes();
				string vectName = XMLString::transcode(vectAttributes->getNamedItem(XMLString::transcode("name"))->getNodeValue());
				int size = atoi(XMLString::transcode(vectAttributes->getNamedItem(XMLString::transcode("size"))->getNodeValue()));
				ExportVal* a = d->getVal(vectName);
				if (a && a->size() == size)
				{
					const char* text = XMLString::transcode(VectNode->getFirstChild()->getNodeValue());
					stringstream s;
					s << text;
                    //cout<<"Reading: " << text << endl;
					a->set(s);
				}
			}
            //cerr <<  "[debug] Read ObjName: "<<objName <<endl;
		}
		else
		{
			cerr << "failed to load data object " << objName << endl;
		}
	}
}
