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

#ifndef _INCLUDED_Helpers_h  
#define _INCLUDED_Helpers_h  

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <set>
#include <algorithm>
#include <iterator>

using namespace std;

template<class T> void deleteT(T* p) 
{
   delete p;
}

template<class T1, class T2> void operator+= (pair<T1, T2>& a, const pair<T1, T2>& b)
{
	a.first += b.first;
	a.second += b.second;
}

template<class T> class objectStore
{

private:

	vector<T*> store;

public:

	void deleteObject(T* object)
	{
		store.push_back(object);
	}

	T* newObject()
	{
		T* object;
		if (store.empty())
		{
			object = new T;
		}
		else
		{
			object = store.back();
			store.pop_back();
		}
		return object;
	}

	T* copy(T* tOld)
	{
		if (tOld)
		{
			T* tNew = newObject();
			*tNew = *tOld;
			return tNew;
		}
		else
		{
			return 0;
		}
	}

	~objectStore()
	{
		for_each(store.begin(), store.end(), deleteT<T>);
	}
};

template <class T> T constrain (const T min, const T max, const T t)
{
	if (t > max)
	{
		return max;
	}
	else if (t < min)
	{
		return min;
	}
	return t;
}

template<class T1, class T2> ostream& operator << (ostream& out, const pair<T1, T2>& p)
{
	out << p.first << ' ' << p.second;
	return out;
}

template<class T1> T1 maxDistance (const vector<T1>& v1, const vector<T1>& v2)
{
	T1 maxDist = 0;
	for (int i = 0; i < (int)v1.size(); ++i)
	{
		T1 dist = fabs(v1[i] - v2[i]);
		if (dist > maxDist)
		{
			maxDist = dist;
		}
	}
	return maxDist;
}

template <class InputIterator, class T> T mean(InputIterator first, InputIterator last)
{
	T m = accumulate(first, last, (T)0) / (T) distance(first, last);
	return m;
}

template <class InputIterator1, class InputIterator2> double 
	squaredEuclideanDistance(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
	double d = 0;
	for (; first1 != last1; ++first1, ++first2)
	{
		double diff = *first1 - *first2;
		d += (diff * diff);
	}
	return d;
}

template <class InputIterator, class T>T stdDev(InputIterator first, InputIterator last)
{
	T m = mean<InputIterator,T>(first, last);
	T dev = (T)0;
	for (InputIterator it = first; it != last; ++it)
	{
		T diff = *it - m;
		dev += diff * diff;
	}
	dev /= (T) distance(first, last);
	return sqrt(dev);
}

template <class InputIterator, class T>T absMean(InputIterator first, InputIterator last)
{
	T am = (T)0;
	for (InputIterator it = first; it != last; ++it)
	{
		if (*it > (T)0)
		{
			am += *it;
		}
		else
		{
			am -= *it;
		}
	}
	am /= (T) distance(first, last);
	return am;
}

template<class T1> void arrayMean (vector<T1>& out, const vector<vector<T1> >& in, int start, int end)
{
	for (int i = start; i < end; ++i)
	{
		transform(in[i].begin(), in[i].end(), out.begin(), out.begin(), plus<T1>());
	}
	transform(out.begin(), out.begin() + in.front().size(), out.begin(), bind2nd(divides<T1>(), end - start));
}

template<class T> void printBufferToStream (const vector<T>& buffer, ostream& out, int bufferWidth)
{
	int bufferHeight = (int)buffer.size() / bufferWidth;
	for (int x = 0; x < bufferWidth; ++x)
	{
		for (int y = 0; y < bufferHeight; ++y)
		{
			out << buffer[x + (y*bufferWidth)] << " ";
		}
		out << endl;
	}
}

template<class T> void printBufferToStreamT (const vector<T>& buffer, ostream& out, int w)
{
	for (int i = 0; i + w <= (int)buffer.size(); i += w)
	{
		copy(&buffer[i], &buffer[i+w], ostream_iterator<T>(out, " "));
		out << endl;
	}
}

template<class T> void printBufferToFile (const vector<T>& buffer, const string& filename, int bufferWidth, 
					  const vector<int>* dims = 0, const vector<string>* labels = 0)
{
	ofstream out (filename.c_str());
	if (out.is_open())
	{
		if (dims)
		{
			out << "DIMENSIONS: ";
			copy(dims->begin(), dims->end(), ostream_iterator<int>(out, " "));
			out << endl;
		}
		if (labels)
		{
			out << "LABELS: ";
			copy(labels->begin(), labels->end(), ostream_iterator<string>(out, " "));
			out << endl;
		}
		printBufferToStream<T>(buffer, out, bufferWidth);
	}
	else
	{
		cerr << "file " << filename << " cannot be created" << endl;
	}
}

template<class T> int sign(T t)
{
	if (t < 0)
	{
		return -1;
	}
	else if (t > 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

#endif
