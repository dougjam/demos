//////////////////////////////////////////////////////////////////////
// TextureStitcher.cpp: Implementation
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

#include "TextureStitcher.h"

#include <IO.h>
#include <trace.h>

#include <STLUtil.h>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
TextureStitcher::TextureStitcher( const FloatArray &trainingSignal,
                                  const FloatArray &trainingSignalLow,
                                  const FloatArray &baseSignal,
                                  int Fs )
  : _trainingSignal( trainingSignal ),
    _trainingSignalLow( trainingSignalLow ),
    _baseSignal( baseSignal ),
    _Fs( Fs )
{
}

//////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////
TextureStitcher::~TextureStitcher()
{
  clear();
}

//////////////////////////////////////////////////////////////////////
// Synthesizes a time domain signal by stitching
// together high frequency components from the training
// data.
//
// Window rate effectively controls how many samples we will
// actually compute from the training data.
//////////////////////////////////////////////////////////////////////
void TextureStitcher::synthesizeStitchedSignal( Real FsLow,
                                                Real windowLength,
                                                Real windowRate,
                                                Real blendLength,
                                                FloatArray &outputSignal )
{
  FloatArray         trainingSignalPadded;
  FloatArray         trainingSignalLowPadded;
  FloatArray         trainingSignalLowDownSampled;
  FloatArray         baseSignalPadded;
  FloatArray         baseSignalDownSampled;

  int                downSampleLevels;
  int                windowHalfWidthLow, windowHalfWidthHigh;
  int                blendLengthHigh = (int)( blendLength * _Fs );
  int                windowStepLow, windowStepHigh;
  Real               FsLowFinal;

  int                start_idx_training, end_idx_training;
  int                start_idx_base, end_idx_base;

  FeatureList        windowFeatures;
  FeaturePtrList     windowFeaturePtrs;
  FeatureTree       *windowFeatureTree;

  printf( "Building padded input signals\n" );

  // Pad the training samples with reflective boundary conditions
  GaussianPyramid::padInputSignal( _trainingSignal,
                                   trainingSignalPadded,
                                   start_idx_training, end_idx_training,
                                   true );

  GaussianPyramid::padInputSignal( _trainingSignalLow,
                                   trainingSignalLowPadded,
                                   start_idx_training, end_idx_training,
                                   true );

  // Pad the base signal with zero boundary conditions
  GaussianPyramid::padInputSignal( _baseSignal,
                                   baseSignalPadded,
                                   start_idx_base, end_idx_base,
                                   false );

  outputSignal.resize( baseSignalPadded.size(), 0.0 );

  printf( "Down sampling signals\n" );

  // Start by down sampling the signals
  downSampleSignal( trainingSignalLowPadded, trainingSignalLowDownSampled,
                    (Real)_Fs, FsLow, FsLowFinal, downSampleLevels );

  downSampleSignal( baseSignalPadded, baseSignalDownSampled,
                    (Real)_Fs, FsLow, FsLowFinal, downSampleLevels );

  chooseWindowLength( windowLength, windowRate, FsLowFinal, downSampleLevels,
                      windowHalfWidthLow, windowHalfWidthHigh,
                      windowStepLow, windowStepHigh );

  printf( "Building window features\n" );

  buildWindowFeatures( windowHalfWidthLow, windowStepLow, downSampleLevels,
                       windowFeatures, trainingSignalLowDownSampled,
                       start_idx_training, end_idx_training );

  for ( int i = 0; i < windowFeatures.size(); i++ )
  {
    windowFeaturePtrs.push_back( &windowFeatures[ i ] );
  }

  printf( "Got %d features with dimension %d\n",
          (int)windowFeatures.size(), windowFeatures[ 0 ].pos().size() );

  printf( "Building window feature tree\n" );

  windowFeatureTree = new FeatureTree( 2 * windowHalfWidthLow + 1 );
  windowFeatureTree->build( windowFeaturePtrs );

  printf( "Building output signal\n" );

  cout << SDUMP( trainingSignalPadded.size() ) << endl;
  cout << SDUMP( trainingSignalLowPadded.size() ) << endl;
  cout << SDUMP( baseSignalDownSampled.size() ) << endl;
  cout << SDUMP( trainingSignalLowDownSampled.size() ) << endl;

  buildOutputSignal( blendLengthHigh, windowHalfWidthLow,
                     windowHalfWidthHigh, windowStepHigh,
                     start_idx_base, end_idx_base,
                     trainingSignalPadded, trainingSignalLowPadded,
                     baseSignalDownSampled, *windowFeatureTree,
                     outputSignal );

  for ( int i = 0; i < baseSignalPadded.size(); i++ )
  {
    outputSignal[ i ] += baseSignalPadded[ i ];
  }

  delete windowFeatureTree;
}

//////////////////////////////////////////////////////////////////////
// Assuming that the input signal has the given sampling
// rate, downsamples the signal until its sampling rate
// lies below the given rate.
//
// Downsampling is done by reducing the sampling rate by
// half at each level and using Gaussian filter coefficients
//////////////////////////////////////////////////////////////////////
void TextureStitcher::downSampleSignal( const FloatArray &input,
                                        FloatArray &output,
                                        Real FsInput, Real FsLow,
                                        Real &FsFinal, int &downSampleLevels )
{
  FloatArray         tempSignal;

  FloatArray        *s1, *s2, *temp;

  tempSignal = input;

  s1 = &tempSignal;
  s2 = &output;

  FsFinal = (Real)FsInput;

  downSampleLevels = 0;

  while ( FsFinal > FsLow )
  {
    GaussianPyramid::buildGaussianLevel( *s1, *s2 );

    FsFinal /= 2.0;

    temp = s1;
    s1 = s2;
    s2 = temp;

    downSampleLevels++;
  }

  output = *s1;
}

//////////////////////////////////////////////////////////////////////
// Given a time window length, determine the appropriate window
// length in the low frequency signal
//////////////////////////////////////////////////////////////////////
void TextureStitcher::chooseWindowLength( Real windowLength, Real windowRate,
                                          Real FsLow,
                                          int downSampleLevels,
                                          int &windowHalfWidthLow,
                                          int &windowHalfWidthHigh,
                                          int &windowStepLow,
                                          int &windowStepHigh )
{
  int            windowFullWidth;
  
  windowFullWidth = (int)ceil( windowLength * FsLow );

  windowHalfWidthLow = windowFullWidth / 2;
  windowHalfWidthLow = max( 0, windowHalfWidthLow );
  windowHalfWidthHigh = windowHalfWidthLow;

  windowStepLow = (int)( windowRate * FsLow );
  windowStepLow = max( 1, windowStepLow );
  windowStepHigh = windowStepLow;

  for ( int i = 0; i < downSampleLevels; i++ )
  {
    windowHalfWidthHigh *= 2;
    windowStepHigh *= 2;
  }
}

//////////////////////////////////////////////////////////////////////
// Builds features from the low frequency training signal
//////////////////////////////////////////////////////////////////////
void TextureStitcher::buildWindowFeatures(
                          int windowHalfWidthLow,
                          int windowStepLow,
                          int downSampleLevels,
                          FeatureList &features,
                          const FloatArray &trainingSignalLowDownSampled,
                          int start_idx, int end_idx )
{
  int            window_middle, window_start, window_end;
  int            feature_idx;
  int            featureDimensions;
  Real           percentDone;

  // The given start and end indices correspond to the signal
  // at its full sampling rate, so we have to reduce them
  // to the down sampled rate
  for ( int i = 0; i < downSampleLevels; i++ )
  {
    if ( start_idx % 2 == 1 )
    {
      start_idx = start_idx / 2 + 1;
    }
    else
    {
      start_idx = start_idx / 2;
    }

    end_idx = end_idx / 2;
  }

  features.clear();

  window_middle = 0;
  window_start = window_middle - windowHalfWidthLow;
  window_end = window_middle + windowHalfWidthLow;

  featureDimensions = 2 * windowHalfWidthLow + 1;

  while ( window_start < (int)trainingSignalLowDownSampled.size() )
  {
    percentDone = (Real)window_middle
                / (Real)trainingSignalLowDownSampled.size();

    printf( "Building window features: %2.2f done\r", percentDone );

    // Get the sample index for the center of this window
    // from the full resolution signal
    //feature_idx = window_middle / windowHalfWidthLow;

    // Skip anything outside of the given range
    if ( window_start < start_idx || window_start >= end_idx
      || window_end < start_idx || window_end >= end_idx )
    {
      window_middle += windowStepLow;
      window_start = window_middle - windowHalfWidthLow;
      window_end = window_middle + windowHalfWidthLow;

      continue;
    }

    // Get the center index for this window in the full
    // resolution signal
    feature_idx = window_middle;

    for ( int i = 0; i < downSampleLevels; i++ )
    {
      feature_idx *= 2;
    }

    features.push_back( GaussianPyramid::FeaturePoint( feature_idx ) );

    GaussianPyramid::FeaturePoint &featurePoint = features.back();


    featurePoint.pos().resizeAndWipe( featureDimensions );

    for ( int sample_idx = window_start; sample_idx <= window_end;
          sample_idx++ )
    {
      if ( sample_idx < 0
        || sample_idx >= (int)trainingSignalLowDownSampled.size() )
      {
        featurePoint.pos()[ sample_idx - window_start ] = 0.0;
      }
      else
      {
        featurePoint.pos()[ sample_idx - window_start ]
          = trainingSignalLowDownSampled[ sample_idx ];
      }
    }

    window_middle += windowStepLow;
    window_start = window_middle - windowHalfWidthLow;
    window_end = window_middle + windowHalfWidthLow;
  }
  printf( "\n" );
}

//////////////////////////////////////////////////////////////////////
// Adds the given window from the training sample data to the
// specified window location in the output signal.  Blending is
// done over the specified blending width, and the blend location
// is chosen to minimize error between the two adjacent signals.
//////////////////////////////////////////////////////////////////////
void TextureStitcher::appendWindow(
                   int blendLengthHigh,
                   int windowHalfWidthHigh, int windowRateHigh,
                   int window_center_training, int window_idx_output,
                   const FloatArray &trainingSignalPadded,
                   const FloatArray &trainingSignalLowPadded,
                   FloatArray &outputSignal )
{
  Real             min_error = FLT_MAX;
  int              windowLength = 2 * windowHalfWidthHigh + 1;
  Real             errorCurrent = 0.0;
  Real             diff;

  int              window_start_training;
  int              window_start_output, window_end_output;

  int              blend_start_max = windowHalfWidthHigh - blendLengthHigh + 2;
  int              blend_start = 0;
  
  window_start_training = window_center_training - windowHalfWidthHigh;

  window_start_output = ( window_idx_output - 1 ) * windowHalfWidthHigh;
  window_end_output = window_start_output + 2 * windowHalfWidthHigh;

  // Do nothing if the output window oversteps the bounds
  // of the signal
  if ( window_start_output < 0 || window_end_output >= outputSignal.size() )
  {
    return;
  }

  // FIXME
  for ( int i = 0; i < windowLength; i++ )
  {
    Real blendValue = (Real)abs( i - windowHalfWidthHigh );
    blendValue /= (Real)windowHalfWidthHigh;
    blendValue = 1.0 - blendValue;

#if 0
    outputSignal[ window_start_output + i ]
      = ( trainingSignalPadded[ window_start_training + i ]
        - trainingSignalLowPadded[ window_start_training + i ] );
#endif
    outputSignal[ window_start_output + i ]
      += blendValue * ( trainingSignalPadded[ window_start_training + i ]
                      - trainingSignalLowPadded[ window_start_training + i ] );
  }

  return;

  // Initialize the error for the first possible blend
  // position
  //
  // TODO: figure out boundary cases...
  for ( int i = 0; i <= blendLengthHigh; i++ )
  {
    diff = outputSignal[ window_start_output + i ];
    diff -= ( trainingSignalPadded[ window_start_training + i ]
            - trainingSignalLowPadded[ window_start_training + i ] );

    errorCurrent += diff * diff;
  }

  min_error = errorCurrent;

  // Figure out what the error is for each possible blending range
  for ( int i = 1; i <= blend_start_max; i++ )
  {
    diff = outputSignal[ window_start_output + i - 1 ];
    diff -= ( trainingSignalPadded[ window_start_training + i - 1 ]
            - trainingSignalLowPadded[ window_start_training + i - 1 ] );

    errorCurrent -= diff * diff;

    int imax = i + blendLengthHigh;

    diff = 0.0;

    if ( window_start_output + imax < outputSignal.size() )
    {
      diff = outputSignal[ window_start_output + imax ];
    }
    if ( window_start_training + imax < trainingSignalPadded.size() )
    {
      diff -= ( trainingSignalPadded[ window_start_training + imax ]
              - trainingSignalLowPadded[ window_start_training + imax ] );
    }

    errorCurrent += diff * diff;

    if ( errorCurrent < min_error )
    {
      blend_start = i;
      min_error = errorCurrent;
    }
  }

  cout << "Blending with error " << min_error << endl;

  // Fill in the blended range
  for ( int i = blend_start; i <= blend_start + blendLengthHigh; i++ )
  {
    Real blend2 = (Real)( i - blend_start ) / (Real)blendLengthHigh;
    Real blend1 = 1.0 - blend2;

    outputSignal[ window_start_output + i ]
     =   blend1 * outputSignal[ window_start_output + i ]
       + blend2 * ( trainingSignalPadded[ window_start_training + i ]
                  - trainingSignalLowPadded[ window_start_training + i ] );
  }

  // After the blended range, we just draw directly from the
  // training signal
  for ( int i = blend_start + blendLengthHigh + 1; i <= windowLength; i++ )
  {
    outputSignal[ window_start_output + i ]
      = ( trainingSignalPadded[ window_start_training + i ]
        - trainingSignalLowPadded[ window_start_training + i ] );
  }
}

//////////////////////////////////////////////////////////////////////
// Builds the output signal by adding one window at a time
// and stitching each new window in to the signal as needed.
//////////////////////////////////////////////////////////////////////
void TextureStitcher::buildOutputSignal(
                        int blendLengthHigh,
                        int windowHalfWidthLow,
                        int windowHalfWidthHigh, int windowRateHigh,
                        int start_idx, int end_idx,
                        const FloatArray &trainingSignalPadded,
                        const FloatArray &trainingSignalLowPadded,
                        const FloatArray &baseSignalDownSampled,
                        const FeatureTree &trainingFeatureTree,
                        FloatArray &outputSignal )
{
  VECTOR                         feature;
  GaussianPyramid::FeaturePoint *nearest;

  Real                           percentDone;
  int                            window_middle, window_start, window_end;

  // Process each window in the down sampled base signal
  for ( int window_idx = 0;
        window_idx * windowHalfWidthLow < baseSignalDownSampled.size();
        window_idx++ )
  {
    if ( window_idx % 10 == 0 )
    {
      percentDone = (Real)( window_idx * windowHalfWidthLow )
                  / (Real)( baseSignalDownSampled.size() );

      printf( "Building signal: %2.2f done\r", percentDone );
    }

    window_middle = window_idx * windowHalfWidthHigh;
    window_start = window_middle - windowHalfWidthHigh;
    window_end = window_middle + windowHalfWidthHigh;

    if ( window_middle < start_idx || window_middle >= end_idx
      || window_start < 0 || window_end >= outputSignal.size() )
    {
      continue;
    }

    buildWindowFeature( baseSignalDownSampled, windowHalfWidthLow,
                        window_idx, feature );

    nearest = trainingFeatureTree.nearestNeighbour( feature );

    cout << "Copying training window with center " << nearest->index() << endl;

    TRACE_ASSERT( nearest, "No nearest feature found" );

    appendWindow( blendLengthHigh, windowHalfWidthHigh, windowRateHigh,
                  nearest->index(), window_idx,
                  trainingSignalPadded, trainingSignalLowPadded,
                  outputSignal );
  }
  printf( "\n" );
}

//////////////////////////////////////////////////////////////////////
// Constructs a feature for the given window of a signal
//////////////////////////////////////////////////////////////////////
void TextureStitcher::buildWindowFeature( const FloatArray &signal,
                                          int windowHalfWidth, int window_idx,
                                          VECTOR &feature )
{
  int        featureSize = 2 * windowHalfWidth + 1;
  int        window_start, window_middle, window_end;

  if ( feature.size() != featureSize )
  {
    feature.resizeAndWipe( featureSize );
  }

  window_middle = window_idx * windowHalfWidth;
  window_start = window_middle - windowHalfWidth;
  window_end = window_middle + windowHalfWidth;

  for ( int i = window_start; i <= window_end; i++ )
  {
    feature( i - window_start ) = signal[ i ];
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void TextureStitcher::clear()
{
}
