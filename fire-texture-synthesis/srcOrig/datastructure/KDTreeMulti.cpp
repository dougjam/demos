//////////////////////////////////////////////////////////////////////
// KDTreeMulti.cpp: Implementation of the KDTreeMulti class
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

#include "KDTreeMulti.h"

#include <algorithm>

#include <trace.h>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTreeMulti<T>::KDTreeMulti( int numDimensions )
	: _root( NULL ),
    _numDimensions( numDimensions )
{
}

//////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTreeMulti<T>::~KDTreeMulti()
{
	delete _root;
}

//////////////////////////////////////////////////////////////////////
// Build's a KD tree out of the given data
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTreeMulti<T>::build( vector<T *> &data )
{
	if ( !_root )
	{
		delete _root;
	}

  _root = new KDNode( 0, _numDimensions );

	_root->build( data );
}

//////////////////////////////////////////////////////////////////////
// Returns the nearest neighbour to point x in the tree
//////////////////////////////////////////////////////////////////////
template <class T>
T *KDTreeMulti<T>::nearestNeighbour( const VECTOR &x ) const
{
  T           *nearest = NULL;
  Real         distance = FLT_MAX;

  TRACE_ASSERT( x.size() == _numDimensions, "Dimension mismatch" );

  if ( _root )
  {
    nearest = _root->nearestNeighbour( x, distance );
  }

  return nearest;
}

//////////////////////////////////////////////////////////////////////
// KDNode class definitions
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTreeMulti<T>::KDNode::KDNode( int split, int numDimensions )
	: _split( split ),
    _numDimensions( numDimensions ),
		_left( NULL ),
		_right( NULL ),
		_data( 0 )
{
}

//////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTreeMulti<T>::KDNode::~KDNode()
{
	delete _left;
	delete _right;
}

//////////////////////////////////////////////////////////////////////
// Build's a tree with the given data
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTreeMulti<T>::KDNode::build( vector<T *> &data )
{
	_numElements = data.size();

	// If we are down to one node, then we're done
	if ( data.size() == 1 )
	{
		_data = data[ 0 ];
		return;
	}

	int										 splitPoint;
	vector<T *>						 leftData, rightData;

	// Sort the data along the split axis direction
	KDCompare	             compare( _split );

	sort( data.begin(), data.end(), compare );

	splitPoint = data.size() / 2;

	leftData.resize( splitPoint );
	rightData.resize( data.size() - splitPoint - 1 );

	// Create data arrays for our left and right children,
	// and grow our bounding box while we're at it.
	for ( int i = 0; i < splitPoint; i++ )
	{
		leftData[i] = data[i];
	}

  _data = data[ splitPoint ];

	for ( int i = splitPoint + 1; i < data.size(); i++ )
	{
		rightData[i - splitPoint - 1] = data[i];
	}

	// If we haven't reduced the size of our subtrees, then
	// we stop here (don't want to recurse indefinitely, after all).
	if ( leftData.size() == data.size() )
	{
    cerr << "ERROR:  Tree size not reduced" << endl;
    abort();
	}

	// Create children for ourselves
  if ( leftData.size() > 0 )
  {
    _left = new KDNode( ( _split + 1 ) % _numDimensions, _numDimensions );

    _left->build( leftData );
  }

  if ( rightData.size() > 0 )
  {
    _right = new KDNode( ( _split + 1 ) % _numDimensions, _numDimensions );

    _right->build( rightData );
  }
}

//////////////////////////////////////////////////////////////////////
// Returns the nearest neighbour for this branch to the given
// point and lying within the given distance.  If no neighbour
// within that distance can be found, returns null
//////////////////////////////////////////////////////////////////////
template <class T>
T *KDTreeMulti<T>::KDNode::nearestNeighbour( const VECTOR &x,
                                             Real &maxDistance ) const
{
  Real           nodeDistance = ( x - _data->pos() ).norm2();
  Real           splitDistance = x( _split ) - _data->pos()[ _split ];

  T             *nearest = ( nodeDistance < maxDistance ) ? _data : NULL;
  T             *childNearest = NULL;

  KDNode        *first = ( splitDistance > 0 ) ? _right : _left;
  KDNode        *second = ( splitDistance > 0 ) ? _left : _right;

  maxDistance = min( maxDistance, nodeDistance );

  if ( first )
  {
    childNearest = first->nearestNeighbour( x, maxDistance );

    if ( childNearest )
    {
      // Must be closer than the current nearest node
      nearest = childNearest;
    }
  }

  // Only examine the other child node if it is possible
  // to have a closer neighbour in this node
  if ( maxDistance >= abs( splitDistance ) && second )
  {
    childNearest = second->nearestNeighbour( x, maxDistance );

    if ( childNearest )
    {
      // Must be closer than the current nearest node
      nearest = childNearest;
    }
  }

  return nearest;
}
