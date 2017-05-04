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

#define DATA_EXPORTER_EXPORTS

#include "DataExporter.h"


DataExporter::DataExporter(const string &n):
		name(n)
{
	DataExportHandler::instance().addExporter(name, this);
}

const string& DataExporter::getName() const
{
	return name;
}

DataExporter::~DataExporter()
{
	for (MSPEV it=vals.begin(); it!=vals.end(); ++it)
	{
		delete it->second;
	}
	DataExportHandler::instance().remExporter(name);
}

ExportVal* DataExporter::getVal (const string& valName)
{
	MSPEV it = vals.find(valName);
	if (it != vals.end())
	{
		return it->second;
	}
	else
	{
		return 0;
	}
}

void DataExporter::save(const string& indent, ostream& out) const
{
	if (vals.size())
	{
		for (MSPCEV it=vals.begin(); it!=vals.end(); ++it)
		{
			it->second->save(out, it->first, indent);
		}
	}
}
