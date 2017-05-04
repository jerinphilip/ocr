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

#define RANDOM_DLL_EXPORTS

#include "Random.h"
//#include "../../contrib/randoma.h"

using namespace std;

//TODO find a good random number library
static int idum;
double ran2 () 
{
    const int IM1 = 2147483563, IM2 = 2147483399;
    const double AM=(1.0/IM1);
    const int IMM1 = IM1-1;
    const int IA1 = 40014, IA2 = 40692, IQ1 = 53668, IQ2 = 52774;
    const int IR1 = 12211, IR2 = 3791, NTAB = 32;
    const int NDIV = 1+IMM1/NTAB;
    const double EPS = 3.0e-16, RNMX = 1.0-EPS;

    int j, k;
    static int idum2=123456789, iy = 0;
    static int iv[NTAB];
    double temp;

    if (idum <= 0) {
        idum = (idum == 0 ? 1 : -idum);
        idum2=idum;
        for (j=NTAB+7;j>=0;j--) {
            k=idum/IQ1;
            idum=IA1*(idum-k*IQ1)-k*IR1;
            if (idum < 0) idum += IM1;
            if (j < NTAB) iv[j] = idum;
        }
        iy=iv[0];
    }
    k=idum/IQ1;
    idum=IA1*(idum-k*IQ1)-k*IR1;
    if (idum < 0) idum += IM1;
    k=idum2/IQ2;
    idum2=IA2*(idum2-k*IQ2)-k*IR2;
    if (idum2 < 0) idum2 += IM2;
    j=iy/NDIV;
    iy=iv[j]-idum2;
    iv[j] = idum;
    if (iy < 1) iy += IMM1;
    if ((temp=AM*iy) > RNMX) return RNMX;
    else return temp;
}

double Random::gaussRandP(double mean, double stdDev)
{
	return gaussRand()*stdDev + mean;
}

double Random::gaussRand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if(phase == 0) {
		do {
			//double U1 = WRandom();
			double U1 = ran2();
			//double U2 = WRandom();
			double U2 = ran2();

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}

void Random::setRandSeed(unsigned int seed)
{
	if (seed == 0)
	{
		seed = (unsigned)time(NULL);
	}
	cout << "setting random seed to " << seed << endl;
	//WRandomInit(seed);
	srand(seed);
	idum = -(int)seed;
}

double Random::flatRnd(double range)
{
	//return (WRandom()*2*range) - range;
	return (ran2()*2*range) - range;
}

double Random::rnd()
{
	//return WRandom();
	return ran2();
}

int Random::randInt(int max)
{
	//return WIRandom (0, max-1);

	//todo: switch this to better random number generator
	return rand()%max;
}

int Random::randInt(int min,int max)
{
	//return WIRandom (min, max);
	return min+randInt(max-min+1);
}


int Random::randBit()
{
	return rand() & 1; 
//	static unsigned int bits=0;
//	if (!bits)
//	{
//		bits = WBRandom ();
//	}
//	else
//	{
//		bits >>= 1;
//	}
//	return bits & 1;
}
