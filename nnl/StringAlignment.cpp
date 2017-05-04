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

#include "StringAlignment.h"




StringAlignment::StringAlignment (int sp, int dp, int ip):
		subPenalty(sp),
		delPenalty(dp),
		insPenalty(ip),
		m(0),
		n(0)
{
}

// Optimal alignment using Levenshtein distance
// Copyright (C) 2005 Santiago Fernandez <santiago@idsia.ch>
StringAlignment::StringAlignment (const vector<int>& reference_sequence,const vector<int>& test_sequence,
								  bool backtrace,int sp, int dp, int ip):
		subPenalty(sp),
		delPenalty(dp),
		insPenalty(ip),
		m(0),
		n(0)
{
	// Levenshtein distance
	// by Anders Sewerin Johansen (http://www.merriampark.com/ldcpp.htm)
	// with minor additions/modifications
	// Step 1
	alignStrings(reference_sequence, test_sequence, backtrace);
}

void StringAlignment::allocateMemory (int refSeqSize, int testSeqSize)
{
	n = refSeqSize;
	m = testSeqSize;
	if (n != 0 && m != 00)
	{
		matrix.resize(n+1);

		// Size the vectors in the 2.nd dimension. Unfortunately C++ doesn't
		// allow for allocation on declaration of 2.nd dimension of vec of vec
		for (int i = 0; i <= n; i++) 
		{
			matrix[i].resize(m+1);
		}
	}
}

int StringAlignment::alignStrings (const vector<int>& reference_sequence,const vector<int>& test_sequence, bool backtrace)
{
	const int refSeqSize = (int)reference_sequence.size();
	const int testSeqSize = (int)test_sequence.size();
	if (refSeqSize != n || testSeqSize != m)
	{
		allocateMemory(refSeqSize, testSeqSize);
	}
	if (n == 0)
	{
		substitutions = 0;
		deletions = 0;
		insertions = m;
		distance = m;
	}
	else if (m == 0)
	{
		substitutions = 0;
		deletions = n;
		insertions = 0;
		distance = n;
	}
	else
	{
		typedef std::vector< std::vector<int> > Tmatrix; 
		Tmatrix matrix(n+1);

		// Size the vectors in the 2.nd dimension. Unfortunately C++ doesn't
		// allow for allocation on declaration of 2.nd dimension of vec of vec
		for (int i = 0; i <= n; i++) 
		{
			matrix[i].resize(m+1);
		}

		// Step 2
		for (int i = 0; i <= n; i++) 
		{
			matrix[i][0]=i;
		}

		for (int j = 0; j <= m; j++) 
		{
			matrix[0][j]=j;
		}

		// Step 3
		for (int i = 1; i <= n; i++) 
		{
			const int s_i = reference_sequence[i-1];

			// Step 4
			for (int j = 1; j <= m; j++) 
			{
				const int t_j = test_sequence[j-1];

				// Step 5

				int cost;
				if (s_i == t_j)
				{
					cost = 0;
				}
				else
				{
					cost = 1;
				}

				// Step 6

				const int above = matrix[i-1][j];
				const int left = matrix[i][j-1];
				const int diag = matrix[i-1][j-1];
				const int cell = min(above + 1,			// deletion
								min(left + 1,			// insertion
									diag + cost));		// substitution

				// Step 6A: Cover transposition, in addition to deletion,
				// insertion and substitution. This step is taken from:
				// Berghel, Hal ; Roach, David : "An Extension of Ukkonen's 
				// Enhanced Dynamic Programming ASM Algorithm"
				// (http://www.acm.org/~hlb/publications/asm/asm.html)
				matrix[i][j]=cell;
			}
		}

		//N.B sub,ins and del penalties are all set to 1 if backtrace is ignored
		if (backtrace)
		{
			// Step 7
			// Usually, there are many possible alignments with the same
			// Levenshtein distance. Each one of them involves a different
			// number of substitutions, deletions or insertions to transform
			// one sequence into the other. We consider only one of the possible
			// transformations.
			//
			// Using pseudo-code from:
			// http://www-igm.univ-mlv.fr/~lecroq/seqcomp/seqcomp.ps.gz
			// with minor additions/modifications

			std::vector<int>::size_type i = n;
			std::vector<int>::size_type j = m;
			substitutions = 0;
			deletions = 0;
			insertions = 0;
			std::vector<int>::size_type length = n; // length of the reference sequence

			// Backtracking
			while (i != 0 && j != 0) 
			{
				if (matrix[i][j] == matrix[i-1][j-1]) 
				{
					--i;
					--j;
				}
				else if (matrix[i][j] == matrix[i-1][j-1] + 1)
				{
					++substitutions;
					--i;
					--j;
				}
				else if (matrix[i][j] == matrix[i-1][j] + 1)
				{
					++deletions;
					--i;
				}
				else 
				{
					++insertions;
					--j;
				}
			}

			while (i != 0) 
			{
				++deletions;
				--i;
			}

			while (j != 0) 
			{
				++insertions;
				--j;
			}

			// Sanity check:
			if ((substitutions + deletions + insertions) != matrix[n][m]) 
			{
				std::cerr <<
					"Found path with distance " <<
					substitutions + deletions + insertions <<
					" but Levenshtein distance is " <<
					matrix[n][m] << std::endl;
			}

			//scale individual errors by penalties
			distance = (subPenalty*substitutions) + (delPenalty*deletions) + (insPenalty*insertions);
		}
		else
		{
			distance = matrix[n][m];
		}
	}
	return distance;
}

int StringAlignment::getSubstitutions () const
{
	return substitutions;
}

int StringAlignment::getInsertions () const
{
	return insertions;
}

int StringAlignment::getDeletions () const
{
	return deletions;
}

int StringAlignment::getDistance () const
{
	return distance;
}
