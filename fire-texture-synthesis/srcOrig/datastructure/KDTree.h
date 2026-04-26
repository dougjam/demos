//////////////////////////////////////////////////////////////////////
// KDTree.h: Interface for the KDTree class
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

#ifndef KD_TREE_H
#define KD_TREE_H

#include <VECTOR.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <BoundingBox.h>

#include <set>
#include <vector>

//////////////////////////////////////////////////////////////////////
// KDTree class
//
// A standard KD-tree for partitioning objects (possibly volumetric
// objects like tetrahedra, etc.).
//
// All classes T used in this template must provide the function
// BoundingBox & bbox();
// which returns the bounding box for the object.
// Single points may simply return a degenerate bounding box.
//////////////////////////////////////////////////////////////////////
template <class T>
class KDTree {
	public:
		// KD Tree splitting directions.
		enum KDSplit {
			SPLIT_X = 0,
			SPLIT_Y,
			SPLIT_Z
		};

		class KDCompare {
			public:
				KDCompare( KDSplit split )
					: _split( split )
				{
				}

				virtual ~KDCompare() {}

				inline bool								 operator()( T *t1, T *t2 )
																	 {
																		 return ( t1->bbox().center()[_split]
																		 			< t2->bbox().center()[_split] );
																	 }

			private:
				// The index to compare on
				KDSplit									 _split;
		};

		// Tree nodes.  These do most of the work.
		class KDNode {
			public:
				KDNode( KDSplit split );

				// Destructor
				virtual ~KDNode();

				// Build a new node given a set of objects
				void										 build( std::vector<T *> &data );

				inline BoundingBox			&bbox()
																 {
																	 return _bbox;
																 }

				// Intersect the given point with this node, and append to a
				// set of T objects (possibly) intersecting with this point
				void										 intersect( const VEC3F &x,
																						std::set<T *> &entries );

				// Same as the above, except that elements are put in to a
				// vector.  Note that this allows duplicate entries.
				void										 intersect( const VEC3F &x,
																						std::vector<T *> &entries );

			private:
				// Bounds for all objects stored in this
				// node.
				BoundingBox							 _bbox;

				// This node's splitting direction
				KDSplit									 _split;

				// Children
				KDNode									*_left;
				KDNode									*_right;

				// List of data for this node - only non-empty at leaves
				std::vector<T *>				 _data;

				// How many elements are stored beneath this node
				int											 _numElements;

		};

		KDTree();

		// Destructor
		virtual ~KDTree();

		// Build's a KD tree out of the given data
		void												 build( std::vector<T *> &data );

		// Intersect the given point with this tree, and append to a
		// set of T objects (possibly) intersecting with this point
		void												 intersect( const VEC3F &x,
																						std::set<T *> &entries );

		// Same as the above, except that elements are put in to a
		// vector.  Note that this allows duplicate entries.
		void												 intersect( const VEC3F &x,
																						std::vector<T *> &entries );

	protected:

	private:
		KDNode											*_root;
};

#include "KDTree.cpp"

#endif
