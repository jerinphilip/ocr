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

#ifndef _INCLUDED_NnMath_h  
#define _INCLUDED_NnMath_h 

#include <limits>
#include <functional>
#include <math.h>

using namespace std;

//TODO: put this in namespace
static const double doubleMax = numeric_limits<double>::max() / 1000;
static const double doubleMin = numeric_limits<double>::min() * 1000;
static const double expLimit = log(doubleMax);		//x < -expLimit => exp(x) = 0; x> expLimit => exp(x) = doubleMax
static const double logMin = log(numeric_limits<double>::min());//numeric_limits<double>::min() * 1e100;

const double *matrixAddMult(double *tgtIt,
							double *tgtEnd,
							const double *matrixIt,
							const double *vectBegin,
							const double *vectEnd);

double *matrixMultVect(const double *vect1Begin,
					   const double *vect1End,
					   double *matrixIt,
					   const double *vect2It,
					   const double *vect2End);

const double *matrixAddMultTranspose(double *tgtBegin,
									 double *tgtEnd,
									 const double *matrixIt,
									 const double *inputIt,
									 const double *inputEnd);
double safeLog (double x);
double safeExp (double x);
double bdedExp (double x);
struct less_mag : public binary_function<double, double, bool>
{
	bool operator()(double x, double y) 
	{ 
		return fabs(x) < fabs(y);
	}
};
double logAdd(double x, double y);
double logMinus(double x, double y);

//class LogDouble
//{
//
//private:
//
//	double val;
//
//public:
//
//	LogDouble(double v):
//			val(v)
//	{
//	}
//
//	LogDouble& operator += (LogDouble log)
//	{
//		val = logAdd(val, log.val);
//		return *this;
//	}
//
//	LogDouble& operator -= (LogDouble log)
//	{
//		val = logMinus(val, log.val);
//		return *this;
//	}
//
//	LogDouble& operator *= (LogDouble log)
//	{
//		val += log.val;
//		return *this;
//	}
//
//	LogDouble& operator /= (LogDouble log)
//	{
//		val -= log.val;
//		return *this;
//	}
//}
//
//LogDouble operator+ (LogDouble log1, LogDouble log2)
//{
//	return LogDouble(logAdd(log1.val, log2.val));
//}
//
//LogDouble operator- (LogDouble log1, LogDouble log2)
//{
//	return LogDouble(logMinus(log1.val, log2.val));
//}
//
//LogDouble operator* (LogDouble log1, LogDouble log2)
//{
//	return LogDouble(log1.val + log2.val);
//}
//
//LogDouble operator/ (LogDouble log1, LogDouble log2)
//{
//	return LogDouble(log1.val - log2.val);
//}


#endif
