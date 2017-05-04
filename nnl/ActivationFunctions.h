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

#ifndef _INCLUDED_ActivationFunctions_h  
#define _INCLUDED_ActivationFunctions_h  

#include <numeric>
#include <limits>

using namespace std;


struct Tanh
{
	static double fn(double x);
	static double deriv(double y);
};

struct Identity
{
	static double fn(double x);
	static double deriv(double y);
};

struct Logistic
{
	static double fn(double x);
	static double deriv(double y);
};

struct Maxmin2
{
	static double fn(double x);
	static double deriv(double y);
};

struct Maxmin1
{
	static double fn(double x);	
	static double deriv(double y);
};

struct Max2min0
{
	static double fn(double x);
	static double deriv(double y);
};

#endif
