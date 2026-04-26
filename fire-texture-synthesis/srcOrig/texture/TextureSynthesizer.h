//////////////////////////////////////////////////////////////////////
// TextureSynthesizer.h: Interface
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

#ifndef TEXTURE_SYNTHESIZER_H
#define TEXTURE_SYNTHESIZER_H

#include <VECTOR.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <vector>

#include <GaussianPyramid.h>

#include <KDTreeMulti.h>

#include <ANN/ANN.h>

//////////////////////////////////////////////////////////////////////
// TextureSynthesizer class
//
// Performs texture synthesis on a time signal using the given
// training signal
//////////////////////////////////////////////////////////////////////
class TextureSynthesizer {
	public:
		TextureSynthesizer( const FloatArray &trainingSignal,
                        const FloatArray &baseSignal,
                        int numLevels, int Fs );

		// Destructor
		virtual ~TextureSynthesizer();

    // Synthesizes a time domain signal from the base signal using
    // the stored training data.  Requires the nearest neighbour
    // half width at each level in the hierarchy as well as the
    // number of levels to preserve from the input signal.
    void synthesizeSignal( const IntArray &filterHalfWidths,
                           int numBaseLevels,
                           bool useANN = false, Real epsANN = 0.0,
                           Real falloff = 0.0 );

    // Same as the above, but does the synthesis by dividing each
    // level of the pyramids in to overlapping windows of the given
    // half width.  Synthesis is then done by building feature vectors
    // out of window powers.
    void synthesizeWindowedSignal( const IntArray &filterHalfWidths,
                                   int numBaseLevels,
                                   int windowHalfWidth );

    // Reconstructs a synthesized signal, assuming synthesizeSignal
    // has already been called.
    void reconstructSignal( FloatArray &finalSignal );

    // Picks the number of base levels to use according to the sampling
    // frequency for this synthesizer.  All levels with sampling frequencies
    // below the given cutoff will be retained.
    int pickNumBaseLevels( Real sampleCutoff );

    void writeAllLevels( const char *prefix );

  private:
    typedef vector<GaussianPyramid::FeaturePoint>      FeatureList;
    typedef vector<GaussianPyramid::FeaturePoint *>    FeaturePtrList;
    typedef KDTreeMulti<GaussianPyramid::FeaturePoint> FeatureTree;

    // Constructs the feature vectors at each sample and each
    // level in the training signal
    void buildLevelFeatures( const IntArray &filterHalfWidths,
                             std::vector<FeatureList> &levelFeatures,
                             std::vector<FeaturePtrList> &levelFeaturePtrs,
                             Real falloff );

    // Same as the above, but builds features using the windowed power
    // signals stored in the synthesis pyramids
    void buildPowerLevelFeatures(
                             const IntArray &filterHalfWidths,
                             std::vector<FeatureList> &levelFeatures,
                             std::vector<FeaturePtrList> &levelFeaturePtrs );

    // Constructs the nearest neighbour search tree for the
    // set of features stored at each level in the training
    // pyramid
    void buildLevelTrees( std::vector<FeaturePtrList> &levelFeaturePtrs,
                          std::vector<FeatureTree *> &levelTrees );

    // Builds trees using the ANN library
    void buildANNTrees( std::vector<FeatureList> &levelFeatures,
                        std::vector<ANNbd_tree *> &levelTrees,
                        std::vector<ANNpointArray> &ANNdataPoints );

    // Uses a texture synthesis procedure to extend the contents
    // of a level of the Gaussian pyramid
    void extendLevel( const IntArray &filterHalfWidths,
                      const std::vector<FeatureList> &levelFeatures,
                      const std::vector<FeatureTree *> &levelTrees,
                      int level,
                      bool useANN, Real epsANN,
                      std::vector<ANNbd_tree *> *annTrees,
                      Real falloff );

    // Same as the above, but does this with windowed power signals,
    // rather than copying input samples directly.
    void extendLevelWindowed( const IntArray &filterHalfWidths,
                              const std::vector<FeatureList> &levelFeatures,
                              const std::vector<FeatureTree *> &levelTrees,
                              int level );

    static void copyANNPoint( const VECTOR &input, ANNpoint output );

    void clear();

	protected:

	private:
    const FloatArray            &_trainingSignal;
    const FloatArray            &_baseSignal;

    int                          _numLevels;
    int                          _Fs;

    GaussianPyramid             *_trainingPyramid;
    GaussianPyramid             *_signalPyramid;

};

#endif
