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

#include "BiasLayer.h"

vector<double> BiasLayer::dummy;

BiasLayer::BiasLayer():
		name("bias"),
		act(1)
{
}

BiasLayer::~BiasLayer()
{
}

int BiasLayer::outputSize() const
{
	return 1;
}

void BiasLayer::getActs(const double** actBegin, const double** actEnd, int offset) const
{
	*actBegin = &act;
	*actEnd = *actBegin + 1;
}

const string& BiasLayer::getName() const
{
	return name;
}
