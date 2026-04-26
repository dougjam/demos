//////////////////////////////////////////////////////////////////////
// KDTree.cpp: Implementation of the KDTree class
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

#include "KDTree.h"

#include <algorithm>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTree<T>::KDTree()
	: _root( NULL )
{
}

//////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTree<T>::~KDTree()
{
	delete _root;
}

//////////////////////////////////////////////////////////////////////
// Build's a KD tree out of the given data
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTree<T>::build( vector<T *> &data )
{
	if ( !_root )
	{
		delete _root;
	}

	_root = new KDTree<T>::KDNode( KDTree::SPLIT_X );

	_root->build( data );
}

//////////////////////////////////////////////////////////////////////
// Intersect the given point with this tree, and append to a
// set of T objects (possibly) intersecting with this point
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTree<T>::intersect( const VEC3F &x, set<T *> &entries )
{
	if ( _root )
	{
		_root->intersect( x, entries );
	}
}

//////////////////////////////////////////////////////////////////////
// Same as the above, except that elements are put in to a
// vector.  Note that this allows duplicate entries.
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTree<T>::intersect( const VEC3F &x, vector<T *> &entries )
{
	if ( _root )
	{
		_root->intersect( x, entries );
	}
}

//////////////////////////////////////////////////////////////////////
// KDNode class definitions
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTree<T>::KDNode::KDNode( KDSplit split )
	: _split( split ),
		_left( NULL ),
		_right( NULL ),
		_data( 0 )
{
}

//////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////
template <class T>
KDTree<T>::KDNode::~KDNode()
{
	delete _left;
	delete _right;
}

//////////////////////////////////////////////////////////////////////
// Build's a tree with the given data
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTree<T>::KDNode::build( vector<T *> &data )
{
	_numElements = data.size();

	// Grow the bounding box for this node to accomodate
	// its data
	for ( int i = 0; i < data.size(); i++ )
	{
		_bbox += data[i]->bbox();
	}

	// If we are down to one node, then we're done
	if ( data.size() == 1 )
	{
		_data = data;
		return;
	}

	int										 splitPoint;
	vector<T *>						 leftData, rightData;

	// Sort the data along the split axis direction
	KDTree<T>::KDCompare	 compare( _split );

	sort( data.begin(), data.end(), compare );

	splitPoint = data.size() / 2;

	leftData.resize( splitPoint );
	rightData.resize( data.size() - splitPoint );

	// Create data arrays for our left and right children,
	// and grow our bounding box while we're at it.
	for ( int i = 0; i < splitPoint; i++ )
	{
		leftData[i] = data[i];
	}

	for ( int i = splitPoint; i < data.size(); i++ )
	{
		rightData[i - splitPoint] = data[i];
	}

	// If we haven't reduced the size of our subtrees, then
	// we stop here (don't want to recurse indefinitely, after all).
	if ( leftData.size() == data.size() )
	{
		_data = data;
		return;
	}

	// Create children for ourselves
	_left = new KDTree<T>::KDNode( (KDSplit)( ((int)_split + 1) % 3 ) );
	_right = new KDTree<T>::KDNode( (KDSplit)( ((int)_split + 1) % 3 ) );

	_left->build( leftData );
	_right->build( rightData );
}

//////////////////////////////////////////////////////////////////////
// Intersect a point with this KD node, returning a set of T pointers
// which (possibly) intersect with the point
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTree<T>::KDNode::intersect( const VEC3F &x, set<T *> &entries )
{
	// If we are at a leaf node, check all entries against the
	// point and insert in to the set, if necessary.
	if ( _data.size() != 0 )
	{
		for ( int i = 0; i < _data.size(); i++ )
		{
			if ( _data[i]->bbox().isInside( x ) )
			{
				entries.insert( _data[i] );
			}
		}

		return;
	}

	// Figure out which children we have to intersect with.
	if ( _left->bbox().isInside( x ) )
	{
		_left->intersect( x, entries );
	}

	if ( _right->bbox().isInside( x ) )
	{
		_right->intersect( x, entries );
	}
}

//////////////////////////////////////////////////////////////////////
// Same as the above, except that elements are put in to a
// vector.  Note that this allows duplicate entries.
//////////////////////////////////////////////////////////////////////
template <class T>
void KDTree<T>::KDNode::intersect( const VEC3F &x, vector<T *> &entries )
{
	// If we are at a leaf node, check all entries against the
	// point and insert in to the set, if necessary.
	if ( _data.size() != 0 )
	{
		for ( int i = 0; i < _data.size(); i++ )
		{
			if ( _data[i]->bbox().isInside( x ) )
			{
				entries.push_back( _data[i] );
			}
		}

		return;
	}

	// Figure out which children we have to intersect with.
	if ( _left->bbox().isInside( x ) )
	{
		_left->intersect( x, entries );
	}

	if ( _right->bbox().isInside( x ) )
	{
		_right->intersect( x, entries );
	}
}
