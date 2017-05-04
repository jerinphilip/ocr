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

#include "ActivationFunctions.h"
#include "NnMath.h"

//activation functions and derivatives
//TODO bound all functions within reasonable limits
double Logistic::fn(double x)
{ 
	if (x < -expLimit)
	{
		x = -expLimit;
	}
	else if (x > expLimit)
	{
		x = expLimit;
	}
	return 1.0 / (1.0 + exp(-x));
}

double Logistic::deriv(double y)  
{ 
	return y*(1.0-y); 
}

double Maxmin2::fn(double x)   
{ 
	if (x < -expLimit)
	{
		x = -expLimit;
	}
	else if (x > expLimit)
	{
		x = expLimit;
	}
	return (4 * Logistic::fn(x)) - 2;
}

double Maxmin2::deriv(double y)  
{ 
	if (y==-2 || y==2)
	{
		return 0;
	}
	return (4 - (y * y)) / 4.0;
}

double Maxmin1::fn(double x)   
{ 
	if (x < -expLimit)
	{
		x = -expLimit;
	}
	else if (x > expLimit)
	{ 
		x = expLimit;
	}
	return (2 * Logistic::fn(x)) - 1;
}

double Maxmin1::deriv(double y)  
{ 
	if (y==-1 || y==1)
	{
		return 0;
	}
	return (1.0 - (y * y)) / 2.0;
}

double Max2min0::fn(double x)   
{ 
	if (x < -expLimit)
	{
		x = -expLimit;
	}
	else if (x > expLimit)
	{ 
		x = expLimit;
	}
	return (2 * Logistic::fn(x));
}

double Max2min0::deriv(double y)  
{ 
	if (y==-1 || y==1)
	{
		return 0;
	}
	return y * (1 - (y / 2.0));
}

double Tanh::fn(double x)  
{  
	return Maxmin1::fn(2*x);
}

double Tanh::deriv(double y)  
{ 
	return 1.0 - (y *  y); 
}

double Identity::fn(double x)
{
	return x;
}

double Identity::deriv(double y)
{
	return 1;
}
