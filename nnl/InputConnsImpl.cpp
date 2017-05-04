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

#include "InputConnsImpl.h"
#include "Helpers.h"

InputConnsImpl::~InputConnsImpl()
{
	for_each(inputs.begin(), inputs.end(), deleteT<Connection>);
}

void InputConnsImpl::printInputConns(ostream& out) const
{
	for (VPBCCI connIt=inputs.begin(); connIt!=inputs.end(); ++connIt)
	{
		(*connIt)->print(out);
	}
}

void InputConnsImpl::addInputConn(Connection* conn)
{
	inputs.push_back(conn);
}

#ifdef DELAY_IN_CONN
void InputConnsImpl::updateDerivs(const double* errBegin, const double* errEnd, const vector<int>& coords, 
				  const vector<int>& seqDims, const vector<int>& dimProducts) const
{
	for (VPBCCI connIt=inputs.begin(); connIt!=inputs.end(); ++connIt)
	{
		(*connIt)->updateDerivs(errBegin, errEnd, coords, seqDims, dimProducts);
	}
}
#endif
void InputConnsImpl::updateDerivs(const double* errBegin, const double* errEnd, int offset) const
{
	for (VPBCCI connIt=inputs.begin(); connIt!=inputs.end(); ++connIt)
	{
		(*connIt)->updateDerivs(errBegin, errEnd, offset);
	}
}

void InputConnsImpl::build()
{
	for_each(inputs.begin(),inputs.end(),mem_fun(&Connection::build));
}
