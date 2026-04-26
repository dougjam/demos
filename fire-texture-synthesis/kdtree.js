// This file is a derivative work of the C++ implementation released
// alongside Chadwick and James, "Animating Fire with Sound," SIGGRAPH 2011.
// See https://www.cs.cornell.edu/projects/Sound/fire/ for the original.
//
// Original copyright notice (preserved per BSD 2-Clause):
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
// Hand-port of srcOrig/datastructure/KDTreeMulti.{h,cpp}. Exact-NN only.
// The data layout differs slightly from the C++: instead of storing
// pointers to objects with a .pos() accessor, we store a flat Float64Array
// of N*D coordinates (`points`) plus a parallel Int32Array of payload
// indices (`indices`) so the caller can recover their original ordering.

'use strict';

(function (global) {

  // In-place quickselect on a slice [lo, hi) of a typed-int permutation
  // array, comparing by `points[perm[i]*dim + axis]`. Positions perm so
  // that perm[k] is the rank-k element (k counted from index 0, not lo)
  // and perm[lo..k] <= perm[k] <= perm[k..hi). Lomuto partition with
  // median-of-three pivot. Linear time on average; worst-case quadratic
  // is vanishingly unlikely with median-of-three on real feature data.
  function _quickselectByAxis(perm, lo, hi, k, points, dim, axis) {
    while (hi - lo > 1) {
      const mid = (lo + hi) >>> 1;
      const last = hi - 1;
      const v0 = points[perm[lo]   * dim + axis];
      const v1 = points[perm[mid]  * dim + axis];
      const v2 = points[perm[last] * dim + axis];
      // Median-of-three: pick the index whose value is the median of v0,v1,v2.
      let pivotIdx, pivotVal;
      if (v0 < v1) {
        if (v1 < v2)      { pivotIdx = mid;  pivotVal = v1; }
        else if (v0 < v2) { pivotIdx = last; pivotVal = v2; }
        else              { pivotIdx = lo;   pivotVal = v0; }
      } else {
        if (v0 < v2)      { pivotIdx = lo;   pivotVal = v0; }
        else if (v1 < v2) { pivotIdx = last; pivotVal = v2; }
        else              { pivotIdx = mid;  pivotVal = v1; }
      }
      // Move pivot to the end of the range.
      { const t = perm[pivotIdx]; perm[pivotIdx] = perm[last]; perm[last] = t; }
      // Lomuto partition: collect elements < pivot at the front.
      let store = lo;
      for (let r = lo; r < last; r++) {
        if (points[perm[r] * dim + axis] < pivotVal) {
          const t = perm[r]; perm[r] = perm[store]; perm[store] = t;
          store++;
        }
      }
      // Move pivot to its final position.
      { const t = perm[store]; perm[store] = perm[last]; perm[last] = t; }
      if (store === k) return;
      if (store < k)  lo = store + 1;
      else            hi = store;
    }
  }

  class KDTreeMulti {
    constructor(points, dimensions, indices) {
      // points: Float64Array of length n_points * dimensions, row-major.
      // dimensions: integer feature dimension.
      // indices: Int32Array of length n_points; the value returned by
      //          `nearestNeighbour` is one of these (preserving the
      //          caller's original index space).
      const n = (points.length / dimensions) | 0;
      if (indices === undefined || indices === null) {
        indices = new Int32Array(n);
        for (let i = 0; i < n; i++) indices[i] = i;
      }
      this._points = points;
      this._dim = dimensions;
      this._indices = indices;
      // We work over a permutation of indices [0, n) and recursively
      // partition it; the tree's data is stored implicitly as
      // (split-axis, mid-index, left-range, right-range) at each node.
      this._perm = new Int32Array(n);
      for (let i = 0; i < n; i++) this._perm[i] = i;
      this._nodes = []; // flat array; each node is 5 ints below
      this._root = -1;
      if (n > 0) this._root = this._build(0, n, 0);
    }

    // Build a node covering this._perm[start..end) with split axis `split`.
    // Returns the node index.
    _build(start, end, split) {
      const n = end - start;
      const dim = this._dim;
      const points = this._points;
      const perm = this._perm;
      const node_idx = this._nodes.length;
      // Layout per node: [split, point_perm_idx, left_node, right_node, _]
      this._nodes.push(0, 0, -1, -1, 0);

      if (n === 1) {
        this._nodes[node_idx + 0] = split;
        this._nodes[node_idx + 1] = perm[start];
        return node_idx;
      }

      // Position the median element at perm[start + splitPoint] using
      // an in-place quickselect (Hoare-partition + median-of-3 pivot).
      // Quickselect is O(n) average vs O(n log n) for a full sort, and
      // we only need the median + a half-partition. The previous version
      // boxed the slice into a regular Array, sorted with a JS callback,
      // and copied back; this avoids all of that.
      const splitPoint = (n / 2) | 0;
      _quickselectByAxis(perm, start, end, start + splitPoint,
                          points, dim, split);
      const midIdx = perm[start + splitPoint];

      const nextSplit = (split + 1) % dim;
      const leftStart = start;
      const leftEnd = start + splitPoint;
      const rightStart = start + splitPoint + 1;
      const rightEnd = end;

      this._nodes[node_idx + 0] = split;
      this._nodes[node_idx + 1] = midIdx;

      if (leftEnd > leftStart) {
        this._nodes[node_idx + 2] = this._build(leftStart, leftEnd, nextSplit);
      }
      if (rightEnd > rightStart) {
        this._nodes[node_idx + 3] = this._build(rightStart, rightEnd, nextSplit);
      }
      return node_idx;
    }

    // Returns { index: int (original payload index), distance: float (L2) }.
    // If the tree is empty, returns { index: -1, distance: Infinity }.
    //
    // Optional `epsilon` (default 0) gives a (1+eps)-approximate search:
    // a neighbour at L2 distance d from the query is accepted iff its
    // d <= (1+eps) * (true nearest distance). epsilon=0 is exact NN
    // (matches scipy.spatial.cKDTree(eps=0)). Higher epsilon prunes
    // more aggressively, often 2-10x faster at the cost of a small
    // chance of missing the true nearest neighbour.
    nearestNeighbour(query, epsilon) {
      if (this._root < 0) return { index: -1, distance: Infinity };
      const eps = epsilon || 0;
      // Distance pruning is done in SQUARED L2 throughout (saves a
      // sqrt per node visit). The (1+eps) tolerance becomes (1+eps)^2
      // in squared form. We keep the public 'distance' field as the
      // true L2 (sqrt'd at the end).
      const state = {
        best: -1,
        bestDistSq: Infinity,
        scaleSq: (1 + eps) * (1 + eps),
      };
      this._nnRecurse(this._root, query, state);
      const idx = state.best >= 0 ? this._indices[state.best] : -1;
      return { index: idx, distance: Math.sqrt(state.bestDistSq) };
    }

    _nnRecurse(node_idx, query, state) {
      // Cache hot-loop locals.
      const nodes = this._nodes;
      const points = this._points;
      const dim = this._dim;
      const scaleSq = state.scaleSq;
      // Iterative recursion via a small explicit stack would be faster
      // still, but JS engines inline shallow recursion well enough that
      // a 30-deep stack is rarely the bottleneck.
      while (node_idx >= 0) {
        const split = nodes[node_idx];
        const pointIdx = nodes[node_idx + 1];
        const left = nodes[node_idx + 2];
        const right = nodes[node_idx + 3];
        const base = pointIdx * dim;

        // Squared L2 distance.
        let s = 0.0;
        for (let d = 0; d < dim; d++) {
          const diff = query[d] - points[base + d];
          s += diff * diff;
        }
        if (s < state.bestDistSq) {
          state.best = pointIdx;
          state.bestDistSq = s;
        }

        const splitDistance = query[split] - points[base + split];
        const splitDistSq = splitDistance * splitDistance;
        const first = splitDistance > 0 ? right : left;
        const second = splitDistance > 0 ? left : right;

        // Recurse on the near child; loop into it for tail-call savings.
        if (first >= 0) this._nnRecurse(first, query, state);

        // Visit far child only if it could contain a (1+eps)-close point.
        if (second >= 0 && state.bestDistSq >= scaleSq * splitDistSq) {
          node_idx = second;
        } else {
          return;
        }
      }
    }
  }

  global.KDTreeMulti = KDTreeMulti;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = KDTreeMulti;
  }
})(typeof self !== 'undefined' ? self : globalThis);
