//////////////////////////////////////////////////////////////////////
// TextureStitcher.h: Interface
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

#ifndef TEXTURE_STITCHER_H
#define TEXTURE_STITCHER_H

#include <VECTOR.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <vector>

#include <GaussianPyramid.h>

#include <KDTreeMulti.h>

//////////////////////////////////////////////////////////////////////
// TextureStitcher class
//
// Synthesizes a time domain signal using a stitching procedure.
// This is done with a two-level procedure in which the training
// signal is decomposed in to low and high frequency components
//////////////////////////////////////////////////////////////////////
class TextureStitcher {
	public:
		TextureStitcher( const FloatArray &trainingSignal,
                     const FloatArray &trainingSignalLow,
                     const FloatArray &baseSignal,
                     int Fs );

    // Destructor
    virtual ~TextureStitcher();

    // Synthesizes a time domain signal by stitching
    // together high frequency components from the training
    // data.
    void synthesizeStitchedSignal( Real FsLow,
                                   Real windowLength, Real windowRate,
                                   Real blendLength,
                                   FloatArray &outputSignal );

  private:
    typedef vector<GaussianPyramid::FeaturePoint>      FeatureList;
    typedef vector<GaussianPyramid::FeaturePoint *>    FeaturePtrList;
    typedef KDTreeMulti<GaussianPyramid::FeaturePoint> FeatureTree;

    // Assuming that the input signal has the given sampling
    // rate, downsamples the signal until its sampling rate
    // lies below the given rate.
    //
    // Downsampling is done by reducing the sampling rate by
    // half at each level and using Gaussian filter coefficients
    static void downSampleSignal( const FloatArray &input,
                                  FloatArray &output,
                                  Real FsInput, Real FsLow,
                                  Real &FsFinal, int &downSampleLevels );

    // Given a time window length, determine the appropriate window
    // length in the low frequency signal
    static void chooseWindowLength( Real windowLength, Real windowRate,
                                    Real FsLow,
                                    int downSampleLevels,
                                    int &windowHalfWidthLow,
                                    int &windowHalfWidthHigh,
                                    int &windowStepLow, int &windowStepHigh );

    // Builds features from the low frequency training signal
    //
    // Only builds features fully contatined within the given
    // start and end index
    static void buildWindowFeatures(
                          int windowHalfWidthLow,
                          int windowStepLow,
                          int downSampleLevels,
                          FeatureList &features,
                          const FloatArray &trainingSignalLowDownSampled,
                          int start_idx, int end_idx );

    // Adds the given window from the training sample data to the
    // specified window location in the output signal.  Blending is
    // done over the specified blending width, and the blend location
    // is chosen to minimize error between the two adjacent signals.
    static void appendWindow( int blendLengthHigh,
                              int windowHalfWidthHigh, int windowRateHigh,
                              int window_center_training, int window_idx_output,
                              const FloatArray &trainingSignalPadded,
                              const FloatArray &trainingSignalLowPadded,
                              FloatArray &outputSignal );

    // Builds the output signal by adding one window at a time
    // and stitching each new window in to the signal as needed.
    static void buildOutputSignal( int blendLengthHigh,
                                   int windowHalfWidthLow,
                                   int windowHalfWidthHigh, int windowRateHigh,
                                   int start_idx, int end_idx,
                                   const FloatArray &trainingSignalPadded,
                                   const FloatArray &trainingSignalLowPadded,
                                   const FloatArray &baseSignalDownSampled,
                                   const FeatureTree &trainingFeatureTree,
                                   FloatArray &outputSignal );

    // Constructs a feature for the given window of a signal
    static void buildWindowFeature( const FloatArray &signal,
                                    int windowHalfWidth, int window_idx,
                                    VECTOR &feature );

    void clear();

	protected:

	private:
    const FloatArray            &_trainingSignal;
    const FloatArray            &_trainingSignalLow;
    const FloatArray            &_baseSignal;

    int                          _Fs;
#if 0
    Real                         _FsLow;

    // Downsampled versions of the lowpass signals, as well as the
    // number of levels of down sampling applied to them.
    FloatArray                   _trainingSignalLowDownSampled;
    FloatArray                   _baseSignalDownSampled;
    int                          _downSampleLevels;

    int                          _windowHalfWidthLow;
    int                          _windowHalfWidthHigh;
#endif

};

#endif
