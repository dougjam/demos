//////////////////////////////////////////////////////////////////////
// WindowSynthesizer.h: Interface
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

#ifndef WINDOW_SYNTHESIZER_H
#define WINDOW_SYNTHESIZER_H

#include <VECTOR.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <vector>

#include <GaussianPyramid.h>

#include <KDTreeMulti.h>

#include <ANN/ANN.h>

//////////////////////////////////////////////////////////////////////
// WindowSynthesizer class
//
// Performs texture synthesis on a time signal using the given
// training signal
//////////////////////////////////////////////////////////////////////
class WindowSynthesizer {
	public:
		WindowSynthesizer( const FloatArray &trainingSignal,
                       const FloatArray &baseSignal,
                       int numLevels, int Fs );

		// Destructor
		virtual ~WindowSynthesizer();

    // Synthesizes a time domain signal from the base signal using
    // the stored training data.  Requires the nearest neighbour
    // half width at each level in the hierarchy as well as the
    // number of levels to preserve from the input signal.
    void synthesizeSignal( const IntArray &windowHalfWidths,
                           const IntArray &filterHalfWidths,
                           int numBaseLevels,
                           bool useLaplacian = false,
                           bool useANN = false, Real epsANN = 0.0,
                           Real falloff = 0.0,
                           bool scaleCDF = false,
                           Real scalingAlpha = 1.0 );

    // This version also requires a lowpass filtered version of
    // the training signal.  It builds the base level of the 
    // signal out of this filtered version, and then all subsequent
    // levels are built out fo the difference signal, obtained by
    // subtracting the low pass filtered signal from the original.
    // This allows us to only synthesize the high frequency part
    // of a signal.
    void synthesizeSignal( const FloatArray &trainingSignalLP,
                           const IntArray &windowHalfWidths,
                           const IntArray &filterHalfWidths,
                           bool useANN = false, Real epsANN = 0.0,
                           Real falloff = 0.0,
                           bool scaleCDF = false,
                           Real scalingAlpha = 1.0 );

    // Reconstructs a synthesized signal, assuming synthesizeSignal
    // has already been called.
    void reconstructSignal( FloatArray &finalSignal );

    // Picks the number of base levels to use according to the sampling
    // frequency for this synthesizer.  All levels with sampling frequencies
    // below the given cutoff will be retained.
    int pickNumBaseLevels( Real sampleCutoff );

    void writeAllLevels( const char *outputPrefix, const char *inputPrefix );

  private:
    typedef vector<GaussianPyramid::FeaturePoint>      FeatureList;
    typedef vector<GaussianPyramid::FeaturePoint *>    FeaturePtrList;
    typedef KDTreeMulti<GaussianPyramid::FeaturePoint> FeatureTree;

    // Constructs the feature vectors at each sample and each
    // level in the training signal.
    //
    // filterHalfWidths we wish to consider around the
    // given synthesis window
    void buildLevelFeatures( const IntArray &windowHalfWidths,
                             const IntArray &filterHalfWidths,
                             std::vector<FeatureList> &levelFeatures,
                             std::vector<FeaturePtrList> &levelFeaturePtrs,
                             Real falloff );

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
    void extendLevel( const IntArray &windowHalfWidths,
                      const IntArray &filterHalfWidths,
                      const std::vector<FeatureList> &levelFeatures,
                      const std::vector<FeatureTree *> &levelTrees,
                      int level,
                      bool useANN = false, Real epsANN = 0.0,
                      std::vector<ANNbd_tree *> *annTrees = NULL,
                      Real falloff = 0.0,
                      bool scaleCDF = false,
                      Real scalingAlpha = 1.0 );

    // Blends the given window from an input signal in to the
    // given input of an output signal using a simple hat function
    static void blendWindowData( const FloatArray &input, int input_window,
                                 FloatArray &output, int output_window,
                                 int windowHalfWidth, Real scale = 1.0 );

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
