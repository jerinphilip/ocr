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

#ifndef _INCLUDED_Typedefs_h  
#define _INCLUDED_Typedefs_h  

#include <string>
#include <vector>
#include <map>

using namespace std;

typedef vector<double>::iterator VDI;
typedef vector<double>::const_iterator VDCI;
typedef vector<double>::reverse_iterator VDRI;
typedef string::iterator SI;
typedef string::const_iterator SCI;
typedef vector<int>::iterator VII;
typedef vector<string>::iterator VSI;
typedef vector<string>::const_iterator VSCI;
typedef vector<int>::reverse_iterator VIRI;
typedef vector<vector<int> >::reverse_iterator VVIRI;
typedef vector<int>::const_iterator VICI;
typedef vector<bool>::iterator VBI;
typedef vector<float>::iterator VFI;
typedef vector<float>::const_iterator VFCI;
typedef vector<vector<double> >::iterator VVDI;
typedef vector<vector<double> >::const_iterator VVDCI;
typedef vector<vector<int> >::iterator VVII;
typedef vector<vector<int> >::const_iterator VVICI;
typedef vector<unsigned int>::iterator VUII;
typedef vector<vector<float> >::iterator VVFI;
typedef map<const string, double>::iterator MCSDI;
typedef map<const string, double>::const_iterator MCSDCI;
typedef map<const string, pair<int,double> >::iterator MCSPIDI;
typedef map<const string, pair<int,double> >::const_iterator MCSPIDCI;
typedef vector< map<const string, pair<int,double> > >::const_iterator VMCSDCI;
typedef vector<map<const string, pair<int,double> > >::iterator VMCSDI;
typedef vector<map<const string, pair<int,double> > >::reverse_iterator VMCSDRI;
typedef map<const string, int>::iterator MCSII;
typedef map<const string, int>::const_iterator MCSICI;
typedef map<int, int>::iterator MIII;
typedef map<int, int>::const_iterator MIICI;
typedef vector<vector<int> >::const_reverse_iterator VVIRCI;
typedef vector<int>::const_reverse_iterator VIRCI;
typedef vector<const float*>::const_iterator VPCFCI;
typedef vector<const float*>::iterator VPCFI;
typedef vector<const float*>::const_reverse_iterator VPCFCRI;
typedef vector<bool>::const_iterator VBCI;
typedef vector<bool>::iterator VBI;
typedef map <const string, pair<double, int> >::iterator MCSPDII;
typedef map <const string, pair<double, int> >::const_iterator MCSPDICI;

#endif
