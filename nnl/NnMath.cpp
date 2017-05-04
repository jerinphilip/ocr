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

#include "NnMath.h"
#include <iostream>

#if HAVE_CONFIG_H
#include <config.h>
#endif 

//#define HAVE_LIBBLAS
#if HAVE_LIBBLAS 
#include <cblas.h>
#endif


// matrix = matrix + vect2(x)vect1 
// vect1 is inner loop
double *matrixMultVect(const double *vect1Begin,
					   const double *vect1End,
					   double *matrixIt,
					   const double *vect2It,
					   const double *vect2End)
{
#if HAVE_LIBBLAS 
	int m = vect2End - vect2It;
	int n = vect1End - vect1Begin;
	cblas_dger(CblasRowMajor, m, n, 1.0, vect2It, 1, vect1Begin, 1, matrixIt, n);
	return matrixIt + (m*n);
#else
	for (;vect2It != vect2End; ++vect2It)
	{
		double input = *vect2It;
		for (const double *vect1It = vect1Begin; vect1It != vect1End; ++vect1It, ++matrixIt)
		{
			*matrixIt += *vect1It * input;
		}
	}
	return matrixIt;
#endif
}

//tgt = matrix input + tgt
const double *matrixAddMult(double *tgtIt,
							double *tgtEnd,
							const double *matrixIt,
							const double *inputBegin,
							const double *inputEnd)
{
#if HAVE_LIBBLAS 
	int m = tgtEnd-tgtIt;
	int n = inputEnd-inputBegin;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, matrixIt, n, inputBegin, 1, 1.0, tgtIt, 1);
	return matrixIt + (m*n);
#else
	for (;tgtIt != tgtEnd; ++tgtIt)
	{
		double sum = 0;
		for (const double *inputIt = inputBegin; inputIt != inputEnd; ++inputIt, ++matrixIt)
		{ 
			sum += *matrixIt * (*inputIt);
		}
		*tgtIt += sum; 
	} 
	return matrixIt;
#endif
}

//tgt = matrix^T input + tgt
const double *matrixAddMultTranspose(double *tgtBegin,
									 double *tgtEnd, 
									 const double *matrixIt,
									 const double *inputIt,
									 const double *inputEnd)
{
#if HAVE_LIBBLAS 
	int n = tgtEnd-tgtBegin;
	int m = inputEnd-inputIt;
	cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1.0, matrixIt, n, inputIt, 1, 1.0, tgtBegin, 1);
	return matrixIt + (m*n);
#else
	for (;inputIt != inputEnd; ++inputIt)
	{
		double input = *inputIt;
		for (double *tgtIt = tgtBegin; tgtIt != tgtEnd; ++tgtIt, ++matrixIt)
		{
			*tgtIt += *matrixIt * input;
		}
	}
	return matrixIt;
#endif
}

double safeLog (double x)
{
	if (x < numeric_limits<double>::min())
	{
		return logMin;
	}
	else
	{
		return log(x);
	}
}

double safeExp (double x)
{
	if (x > expLimit)
	{
		return doubleMax;
	}
	return exp(x);
}

double bdedExp (double x)
{
	if (x < -expLimit)
	{
		return doubleMin;
	}
	else if (x > expLimit)
	{
		return doubleMax;
	}
	return exp(x);
}

#define LZERO  (-1.0E10)   /* ~log(0) */
#define LSMALL (-0.5E10)   /* log values < LSMALL are set to LZERO */
static const double minLogExp = -log(-LZERO);
//typedef double LogDouble;
/* EXPORT->LAdd: Return sum x + y on log scale,
                sum < LSMALL is floored to LZERO */
double logAdd(double x, double y)
{
	if (x < y) 
	{
		double temp = x; 
		x = y; 
		y = temp;
	}
	double diff = y - x;
	if (diff < -expLimit)
	{
		return x;
		//return (x < LSMALL) ? LZERO : x;
	}
	else 
	{
		double z = safeExp(diff);
		return x + log(1.0 + z);
	}
}

double logMinus(double x, double y)
{
	if (y > x)
	{
		cout << "logMinus: error x " << x << " y " << y << endl;
		return -doubleMax;
	}
	double diff = y - x;
	if (diff < -expLimit)
	{
		return x;
	}
	else 
	{
		double z = safeExp(diff);
		return x + log(1.0 - z);
	}
}

