//////////////////////////////////////////////////////////////////////
// STLUtil.h: Interface
//
// Copyright (c) 2011, Jeffrey Chadwick
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//////////////////////////////////////////////////////////////////////

#ifndef STL_UTIL_H
#define STL_UTIL_H

#include <VECTOR.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <vector>

//////////////////////////////////////////////////////////////////////
// Some random utility functions that I needed a place for
//////////////////////////////////////////////////////////////////////

// Deletes all contents in a vector of pointers
template <class T>
inline void clearVectorContents( std::vector<T *> &data )
{
  for ( int i = 0; i < data.size(); i++ )
  {
    delete data[ i ];
  }
}

// Writes vector data to a file
inline void writeRealVector(const char *filename, const vector<double> &data)
{
  FILE* file;
  file = fopen(filename, "wb");

  if( file == NULL )
  {
	  printf( "** WARNING ** Could not write vector to %s\n", filename );
	  return;
  }

  int size = data.size();

  // write dimensions
  fwrite((void*)&size, sizeof(int), 1, file);

  // write data
  fwrite((void*)data.data(), sizeof(double), size, file);
  fclose(file);
}

// Reads vector data from a file
inline void readRealVector(const char *filename, vector<double> &data)
{
  FILE *file;
  file = fopen(filename, "rb");

  if ( file == NULL )
  {
    printf( "** WARNING ** Could not read vector %s\n", filename );
    return;
  }

  int size;

  // Read size
  fread((void*)&size, sizeof(int), 1, file);

  data.resize( size );

  // Read data
  fread((void*)data.data(), sizeof(double), size, file);
  fclose(file);
}

// Sum of entries in a vector
template <class T>
inline T vectorSum( const std::vector<T> &data )
{
  T sum = 0;

  for ( int i = 0; i < data.size(); i++ )
  {
    sum += data[ i ];
  }
}

#endif
