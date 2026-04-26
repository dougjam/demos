//////////////////////////////////////////////////////////////////////
// KDTreeMulti.h: Interface for the KDTreeMulti class
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

#ifndef KD_TREE_MULTI_H
#define KD_TREE_MULTI_H

#include <VECTOR.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <BoundingBox.h>

#include <set>
#include <vector>

//////////////////////////////////////////////////////////////////////
// KDTreeMulti class
//
// A standard KD-tree for partitioning objects with point locations
// in a multidimensional space
//
// All classes T used in this template must provide the function
//    const VECTOR &pos() const
// which returns the location of the object in a multi dimensional
// space
//////////////////////////////////////////////////////////////////////
template <class T>
class KDTreeMulti {
	public:
		class KDCompare {
			public:
				KDCompare( int split )
					: _split( split )
				{
				}

				virtual ~KDCompare() {}

				inline bool								 operator()( T *t1, T *t2 )
																	 {
                                     return ( t1->pos()[ _split ]
                                            < t2->pos()[ _split ] );
																	 }

			private:
				// The index to compare on
				int		      						   _split;
		};

		// Tree nodes.  These do most of the work.
		class KDNode {
			public:
				KDNode( int split, int numDimensions );

				// Destructor
				virtual ~KDNode();

				// Build a new node given a set of objects
				void										 build( std::vector<T *> &data );

        // Returns the nearest neighbour for this branch to the given
        // point and lying within the given distance.  If no neighbour
        // within that distance can be found, returns null
        T                       *nearestNeighbour( const VECTOR &x,
                                                   Real &maxDistance ) const;

			private:
				// This node's splitting direction
				int	    								 _split;
        int                      _numDimensions;

				// Children
				KDNode									*_left;
				KDNode									*_right;

        // Data point stored at this node
        T                       *_data;

				// How many elements are stored beneath this node
				int											 _numElements;

		};

		KDTreeMulti( int numDimensions );

		// Destructor
		virtual ~KDTreeMulti();

		// Build's a KD tree out of the given data
		void												 build( std::vector<T *> &data );

    // Returns the nearest neighbour to point x in the tree
    T                           *nearestNeighbour( const VECTOR &x ) const;

	protected:

	private:
		KDNode											*_root;
    int                          _numDimensions;

};

#include "KDTreeMulti.cpp"

#endif
