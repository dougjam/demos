//////////////////////////////////////////////////////////////////////
// GaussianPyramid.cpp: Implementation for the GaussianPyramid
//                      class
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

#include "GaussianPyramid.h"

#include <algorithm>
#include <iostream>

#include <MathUtil.h>
#include <STLUtil.h>

#include <IO.h>

#include <math.h>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Stencils for computing higher levels of the pyramid
//////////////////////////////////////////////////////////////////////
int GaussianPyramid::GAUSSIAN_STENCIL_SZ = 5;

Real GaussianPyramid::GAUSSIAN_STENCIL[] = { 0.05, 0.25, 0.4, 0.25, 0.05 };

Real GaussianPyramid::AMPLITUDE_CUTOFF = 1e-4;

FloatArray GaussianPyramid::scalingVector = FloatArray();

MERSENNETWISTER GaussianPyramid::generator = MERSENNETWISTER();

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
GaussianPyramid::GaussianPyramid( const FloatArray &inputSignal,
                                  int numLevels, bool reflectBoundaries )
  : _levels( numLevels ),
    _startIndex( numLevels ),
    _endIndex( numLevels )
{
  copyInputSignal( inputSignal, reflectBoundaries );

  computeLevels();
}

//////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////
GaussianPyramid::~GaussianPyramid()
{
}

//////////////////////////////////////////////////////////////////////
// Returns the features for all points at a given level in
// the pyramid, using the given feature sizes for each level.
// Sizes are assumed to specify the half width of the feature
// in sample.
//
// Only computes features for points fully inside of the span
// of the original signal.
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::buildLevelFeatures( vector<FeaturePoint> &features,
                                          const IntArray &featureHalfWidths,
                                          int level, Real falloff )
{
  features.clear();

  for ( int i = 0; i < _levels[ level ].size(); i++ )
  {
    FeaturePoint         featurePoint( i );

    if ( computeEntryFeature( featurePoint, featureHalfWidths,
                              level, i, falloff ) )
    {
      features.push_back( featurePoint );
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Same as the above, but builds a list of features for entries
// in a windowed power signal.
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::buildPowerLevelFeatures( vector<FeaturePoint> &features,
                                               const IntArray &featureHalfWidths,
                                               int level )
{
  features.clear();

  for ( int i = 0; i < _levels[ level ].size(); i++ )
  {
    FeaturePoint         featurePoint( i );

    if ( computePowerEntryFeature( featurePoint, featureHalfWidths, level, i ) )
    {
      features.push_back( featurePoint );
    }
  }
}

// This version computes features for windows of the given
// half width.  Due to cost, features are only computed
// using the current level and the one directly beneath it.
void GaussianPyramid::buildWindowFeatures( vector<FeaturePoint> &features,
                                           const IntArray &windowHalfWidths,
                                           const IntArray &featureHalfWidths,
                                           int level, Real falloff )
{
  int              numWindows;

  features.clear();
  
  numWindows = _levels[ level ].size() / windowHalfWidths[ level ] + 1;

  for ( int window_idx = 0; window_idx < numWindows; window_idx++ )
  {
    FeaturePoint         featurePoint( window_idx );

    if ( computeWindowFeature( featurePoint, windowHalfWidths,
                               featureHalfWidths, level,
                               window_idx, falloff ) )
    {
      features.push_back( featurePoint );
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Computes a feature vector for the given entry in the pyramid
// Returns true only if all samples were drawn from within the
// span of the input signal.
//////////////////////////////////////////////////////////////////////
bool GaussianPyramid::computeEntryFeature( FeaturePoint &featurePoint,
                                           const IntArray &featureHalfWidths,
                                           int level, int entry, Real falloff )
{
  int              featureDimensions = featureHalfWidths[ level ];
  int              feature_idx = 0;
  int              sample_idx_fixed;
  bool             allInside = true;
  int              level_idx = level;
  IntArray         levelEntry( _levels.size() );

  levelEntry[ level ] = entry;

  for ( int i = level + 1; i < _levels.size(); i++ )
  {
    levelEntry[ i ] = levelEntry[ i - 1 ] / 2;

    featureDimensions += 2 * featureHalfWidths[ i ] + 1;
  }

  featurePoint.index() = entry;

  if ( featurePoint.pos().size() != featureDimensions )
  {
    featurePoint.pos().resizeAndWipe( featureDimensions );
  }

  // Start with levels highest in the pyramid.  These do
  // not have the causality requirement, so we can consider
  // a symmetric neighbourhood around the sample point.
  //
  // FIXME: We could probably use better sampling here.
  // Eg. once we start going to higher levels, we technically
  // end up with non-integral sample indices.  This could
  // be handled using interpolation perhaps?
  //for ( level_idx = level + 1; level_idx < _levels.size(); level_idx++ )
  for ( level_idx = _levels.size() - 1; level_idx > level; level_idx-- )
  {
    entry = levelEntry[ level_idx ];

    for ( int sample_idx = entry - featureHalfWidths[ level_idx ];
          sample_idx <= entry + featureHalfWidths[ level_idx ]; sample_idx++ )
    {
      allInside &= ( sample_idx >= _startIndex[ level_idx ] );
      allInside &= ( sample_idx < _endIndex[ level_idx ] );

      sample_idx_fixed = mod_fixed( sample_idx, _levels[ level_idx ].size() );

      featurePoint.pos()[ feature_idx ]
        = _levels[ level_idx ][ sample_idx_fixed ];

      // Apply exponential falloff
      Real weight = exp( -1.0 * falloff * (Real)abs( sample_idx - entry ) );

      featurePoint.pos()[ feature_idx ] *= weight;

      feature_idx++;
    }
  }

  // Finish with the contributions from the current level,
  // since this is the only special case.  In order to
  // preserve causality, only consider samples that came
  // before the current one.
  level_idx = level;
  entry = levelEntry[ level_idx ];

  for ( int sample_idx = entry - featureHalfWidths[ level_idx ];
        sample_idx < entry; sample_idx++ )
  {
    allInside &= ( sample_idx >= _startIndex[ level_idx ] );
    allInside &= ( sample_idx < _endIndex[ level_idx ] );

    sample_idx_fixed = mod_fixed( sample_idx, _levels[ level_idx ].size() );

    featurePoint.pos()[ feature_idx ]
      = _levels[ level_idx ][ sample_idx_fixed ];

    // Apply exponential falloff
    Real weight = exp( -1.0 * falloff * (Real)abs( sample_idx - entry ) );

    featurePoint.pos()[ feature_idx ] *= weight;

    feature_idx++;
  }

#if 0
  // Start with the contributions from the current level,
  // since this is the only special case.  In order to
  // preserve causality, only consider samples that came
  // before the current one.
  for ( int sample_idx = entry - featureHalfWidths[ level_idx ];
        sample_idx < entry; sample_idx++ )
  {
    allInside &= ( sample_idx >= _startIndex[ level_idx ] );
    allInside &= ( sample_idx < _endIndex[ level_idx ] );

    sample_idx_fixed = mod_fixed( sample_idx, _levels[ level_idx ].size() );

    featurePoint.pos()[ feature_idx ]
      = _levels[ level_idx ][ sample_idx_fixed ];

    feature_idx++;
  }

  // Next, handle the levels higher in the pyramid.  These do
  // not have the causality requirement, so we can consider
  // a symmetric neighbourhood around the sample point.
  //
  // FIXME: We could probably use better sampling here.
  // Eg. once we start going to higher levels, we technically
  // end up with non-integral sample indices.  This could
  // be handled using interpolation perhaps?
  for ( level_idx = level + 1; level_idx < _levels.size(); level_idx++ )
  {
    entry /= 2;

    for ( int sample_idx = entry - featureHalfWidths[ level_idx ];
          sample_idx <= entry + featureHalfWidths[ level_idx ]; sample_idx++ )
    {
      allInside &= ( sample_idx >= _startIndex[ level_idx ] );
      allInside &= ( sample_idx < _endIndex[ level_idx ] );

      sample_idx_fixed = mod_fixed( sample_idx, _levels[ level_idx ].size() );

      featurePoint.pos()[ feature_idx ]
        = _levels[ level_idx ][ sample_idx_fixed ];

      feature_idx++;
    }
  }
#endif

  return allInside;
}

//////////////////////////////////////////////////////////////////////
// Same as the above, but computes features on entries of
// the window power signal, rather than the signal itself
//////////////////////////////////////////////////////////////////////
bool GaussianPyramid::computePowerEntryFeature(
                                          FeaturePoint &featurePoint,
                                          const IntArray &featureHalfWidths,
                                          int level, int entry )
{
  int              featureDimensions = featureHalfWidths[ level ];
  int              feature_idx = 0;
  int              sample_idx_fixed;
  bool             allInside = true;
  int              level_idx = level;
  IntArray         levelEntry( _levels.size() );

  TRACE_ASSERT( _levels.size() == _windowLevels.size(),
                "Windowed signal not computed" );

  levelEntry[ level ] = entry;

  for ( int i = level + 1; i < _levels.size(); i++ )
  {
    levelEntry[ i ] = levelEntry[ i - 1 ] / 2;

    featureDimensions += 2 * featureHalfWidths[ i ] + 1;
  }

  featurePoint.index() = entry;

  if ( featurePoint.pos().size() != featureDimensions )
  {
    featurePoint.pos().resizeAndWipe( featureDimensions );
  }

  // Start with levels highest in the pyramid.  These do
  // not have the causality requirement, so we can consider
  // a symmetric neighbourhood around the sample point.
  //
  // FIXME: We could probably use better sampling here.
  // Eg. once we start going to higher levels, we technically
  // end up with non-integral sample indices.  This could
  // be handled using interpolation perhaps?
  //for ( level_idx = level + 1; level_idx < _levels.size(); level_idx++ )
  for ( level_idx = _levels.size() - 1; level_idx > level; level_idx-- )
  {
    entry = levelEntry[ level_idx ];

    for ( int sample_idx = entry - featureHalfWidths[ level_idx ];
          sample_idx <= entry + featureHalfWidths[ level_idx ]; sample_idx++ )
    {
      allInside &= ( sample_idx >= _windowStartIndex[ level_idx ] );
      allInside &= ( sample_idx <= _windowEndIndex[ level_idx ] );

      sample_idx_fixed = mod_fixed( sample_idx,
                                    _windowLevels[ level_idx ].size() );

      featurePoint.pos()[ feature_idx ]
        = _windowLevels[ level_idx ][ sample_idx_fixed ];

      feature_idx++;
    }
  }

  // Finish with the contributions from the current level,
  // since this is the only special case.  In order to
  // preserve causality, only consider samples that came
  // before the current one.
  level_idx = level;
  entry = levelEntry[ level_idx ];

  for ( int sample_idx = entry - featureHalfWidths[ level_idx ];
        sample_idx < entry; sample_idx++ )
  {
    allInside &= ( sample_idx >= _startIndex[ level_idx ] );
    allInside &= ( sample_idx <= _endIndex[ level_idx ] );

    sample_idx_fixed = mod_fixed( sample_idx,
                                  _windowLevels[ level_idx ].size() );

    featurePoint.pos()[ feature_idx ]
      = _windowLevels[ level_idx ][ sample_idx_fixed ];

    feature_idx++;
  }

  return allInside;
}

//////////////////////////////////////////////////////////////////////
// This version computes a feature vector for a window
// using only the current level and the one directly
// beneath it.
//////////////////////////////////////////////////////////////////////
bool GaussianPyramid::computeWindowFeature( FeaturePoint &featurePoint,
                                            const IntArray &windowHalfWidths,
                                            const IntArray &featureHalfWidths,
                                            int level, int window_idx,
                                            Real falloff,
                                            const FloatArray *inputCDF,
                                            const FloatArray *outputCDF,
                                            Real *scale,
                                            Real alpha )
{
  int              featureDimensions = 0;
  int              windowHW = windowHalfWidths[ level ];
  int              featureHW = featureHalfWidths[ level ];
  int              window_middle = window_idx * windowHW;
  int              feature_start, feature_end;
  int              feature_idx = 0;
  bool             allInside = true;

  // Need real-valued indices to properly sample the
  // level above this one
  Real             window_middle_real;
  Real             feature_start_real;
  int              feature_length;

  Real             averageMagnitude = 0.0;

  // Figure out how many feature dimensions there should be.
  // Start with contributions from the current level.
  feature_start = window_middle - windowHW * ( featureHW + 1 );
  feature_end = window_middle - windowHW;

  featureDimensions += feature_end - feature_start + 1;

  if ( level < _levels.size() - 1 )
  {
    // Get contributions from the next level
    windowHW = windowHalfWidths[ level + 1 ];
    featureHW = featureHalfWidths[ level + 1 ];

    featureDimensions += 2 * windowHW * ( featureHW + 1 ) + 1;
  }

  if ( featurePoint.pos().size() != featureDimensions )
  {
    featurePoint.pos().resizeAndWipe( featureDimensions );
  }

  featurePoint.pos().clear();

  // Fill in the feature, starting with components
  // at the current level
  windowHW = windowHalfWidths[ level ];
  featureHW = featureHalfWidths[ level ];

  window_middle = window_idx * windowHW;
  feature_start = window_middle - windowHW * ( featureHW + 1 );
  feature_end = window_middle - windowHW;

  for ( int i = feature_start; i <= feature_end; i++ )
  {
    allInside &= ( i >= _startIndex[ level ] && i < _endIndex[ level ] );

    if ( i >= 0 && i < _levels[ level ].size() )
    {
      featurePoint.pos()[ feature_idx ] = _levels[ level ][ i ];
    }
    else
    {
      featurePoint.pos()[ feature_idx ] = 0.0;
    }

    // Apply exponential falloff
    Real weight = exp( -1.0 * falloff * (Real)abs( i - window_middle ) );

    featurePoint.pos()[ feature_idx ] *= weight;

    feature_idx++;
  }

  if ( level == _levels.size() - 1 )
  {
    return allInside;
  }

  // Fill in components using the level above us
  windowHW = windowHalfWidths[ level + 1 ];
  featureHW = featureHalfWidths[ level + 1 ];

  window_middle_real = (Real)window_middle / 2.0;
  feature_start_real = window_middle_real
                     - (Real)( windowHW * ( featureHW + 1 ) );
  feature_length = 2 * windowHW * ( featureHW + 1 ) + 1;

  bool inside;

  for ( int i = 0; i < feature_length; i++ )
  {
    featurePoint.pos()[ feature_idx ]
      = sampleSignal( _levels[ level + 1 ],
                      feature_start_real + (Real)i, inside );

    allInside &= inside;

    // Apply exponential falloff
    Real weight = exp( -1.0 * falloff
          * abs( feature_start_real + (Real)i - window_middle_real ) );

    featurePoint.pos()[ feature_idx ] *= weight;

    averageMagnitude += abs( featurePoint.pos()[ feature_idx ] );

    feature_idx++;
  }

  averageMagnitude /= (Real)feature_length;

  // If these feature entries came from the top level and
  // if we have CDF information to do scaling, do that now.
  if ( level + 1 == _levels.size() - 1 && inputCDF && outputCDF
       && averageMagnitude > AMPLITUDE_CUTOFF )
  {
    Real         outputFraction = sampleInverseCDF( averageMagnitude,
                                                    *outputCDF );
    Real         inputMagnitude = sampleCDF( outputFraction, *inputCDF );

    Real         scaling = inputMagnitude / averageMagnitude;

    // FIXME
    scaling = 1.0 - alpha + alpha * scaling;

    scalingVector.push_back( scaling );

    (*scale) = 1.0 / scaling;

    for ( int i = 0; i < feature_length; i++ )
    {
      feature_idx--;

      featurePoint.pos()[ feature_idx ] *= scaling;
    }
  }
  else if ( level + 1 == _levels.size() - 1 && inputCDF && outputCDF
            && averageMagnitude <= AMPLITUDE_CUTOFF )
  {
    scalingVector.push_back( 1.0 );
  }

#if 0
  if ( level + 1 == _levels.size() - 1 && inputCDF && outputCDF )
  {
    for ( int i = 0; i < feature_length; i++ )
    {
      feature_idx--;

      Real magnitude = abs( featurePoint.pos()[ feature_idx ] );

      Real outputFraction = sampleInverseCDF( averageMagnitude, 
                                              *outputCDF );
      Real inputMagnitude = sampleCDF( outputFraction, *inputCDF );

      if ( magnitude > AMPLITUDE_CUTOFF )
      {
        Real scaling = inputMagnitude / magnitude;

        featurePoint.pos()[ feature_idx ] *= scaling;
      }
    }
  }
#endif

  return allInside;
}

//////////////////////////////////////////////////////////////////////
// Reconstructs the input signal (just copy it from the bottom
// level for a Gaussian pyramid)
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::reconstructSignal( FloatArray &signal )
{
  signal.clear();

  for ( int i = _startIndex[ 0 ]; i < _endIndex[ 0 ]; i++ )
  {
    signal.push_back( _levels[ 0 ][ i ] );
  }
}

//////////////////////////////////////////////////////////////////////
// Sets contents of a level to zero
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::zeroLevel( int level )
{
  for ( int i = 0; i < _levels[ level ].size(); i++ )
  {
    _levels[ level ][ i ] = 0.0;
  }
}


//////////////////////////////////////////////////////////////////////
// Divides each level of the pyramid in to overlapping windows of the
// given half width and computes a signal, whose value for a given
// window is given by the signal power in that window.
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::computeWindowedSignals( int windowHalfWidth )
{
  _windowLevels.resize( _levels.size() );
  _windowStartIndex.resize( _levels.size() );
  _windowEndIndex.resize( _levels.size() );

  for ( int level_idx = 0; level_idx < _levels.size(); level_idx++ )
  {
    computeWindowPowerSignal( level_idx );
  }
}

//////////////////////////////////////////////////////////////////////
// Adds the signal from window window_input_idx in the input signal
// to the window window_output_idx in the signal at the given level
// in this pyramid.  Also updates the corresponding window power
// term.
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::addWindowContribution( const FloatArray &inputSignal,
                                             int level,
                                             int window_input_idx,
                                             int window_output_idx )
{
  int              input_start, input_middle, input_end, input_idx;
  int              output_start, output_middle, output_end;
  int              sz;
  Real             blendValue;

  TRACE_ASSERT( _windowLevels.size() == _levels.size(),
                "Power signals not initialized" );

  input_middle = _windowHalfWidth * window_input_idx;
  output_middle = _windowHalfWidth * window_output_idx;

  output_start = max( 0, output_middle - _windowHalfWidth );
  output_end = min( (int)( _levels[ level ].size() - 1 ),
                    output_middle + _windowHalfWidth );

  input_start = input_middle - ( output_middle - output_start );
  input_end = input_middle + ( output_end - output_middle );

  sz = input_end - input_start + 1;

  // Copy the signal over and apply the windowing function
  for ( int i = 0; i < sz; i++ )
  {
    input_idx = input_start + i;
    blendValue = (Real)( _windowHalfWidth - abs( input_idx - input_middle ) )
               / (Real)( _windowHalfWidth );

    _levels[ level ][ output_start + i ]
      += blendValue * inputSignal[ input_start + i ];
  }

  // Compute the power value for this window
  _windowLevels[ level ][ window_output_idx ]
    = windowPower( level, window_output_idx );
}

//////////////////////////////////////////////////////////////////////
// Conversion between Gaussian and Laplacian type
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::convertToLaplacian()
{
  if ( _isLaplacian )
    return;

  _isLaplacian = true;

  vector<FloatArray>       laplacianLevels( _levels.size() );
  int                      level_size;

  for ( int level_idx = 0; level_idx < _levels.size() - 1; level_idx++ )
  {
    level_size = _levels[ level_idx ].size();

    laplacianLevels[ level_idx ].resize( level_size, 0.0 );

    for ( int sample_idx = 0; sample_idx < level_size; sample_idx++ )
    {
      laplacianLevels[ level_idx ][ sample_idx ]
        = _levels[ level_idx ][ sample_idx ] - expand( level_idx, sample_idx );
    }
  }

  laplacianLevels[ _levels.size() - 1 ] = _levels[ _levels.size() - 1 ];

  _levels = laplacianLevels;
}

//////////////////////////////////////////////////////////////////////
// Conversion between Gaussian and Laplacian type
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::convertToGaussian()
{
  if ( !_isLaplacian )
    return;

  _isLaplacian = false;

  int                      level_size;

  for ( int level_idx = _levels.size() - 2; level_idx >= 0; level_idx-- )
  {
    level_size = _levels[ level_idx ].size();
    
    for ( int sample_idx = 0; sample_idx < level_size; sample_idx++ )
    {
      _levels[ level_idx ][ sample_idx ] += expand( level_idx, sample_idx );
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Figures out the contents of ( level + 1 ) at index sample_idx
// in level using the EXPAND operation from [Burt and Adelson]
//////////////////////////////////////////////////////////////////////
Real GaussianPyramid::expand( int level, int sample_idx )
{
  Real result = 0.0;

  for ( int n = -GAUSSIAN_STENCIL_SZ; n <= GAUSSIAN_STENCIL_SZ; n++ )
  {
    int index = 2 * sample_idx + n;

    if ( index % 2 == 1 )
      continue;

    index /= 2;

    if ( index >= 0 && index < _levels[ level + 1 ].size() )
    {
      result += _levels[ level + 1 ][ index ];
    }
  }

  return result;
}

//////////////////////////////////////////////////////////////////////
// Initializes a cumulative distribution function for the
// highest level in the pyramid.
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::initCDF()
{
  int              numLevels = _levels.size();
  int              cdf_size;
  int              cdf_idx = 0;

  cdf_size = _endIndex[ numLevels - 1 ] - _startIndex[ numLevels - 1 ];

  _CDF.clear();

  for ( int i = _startIndex[ numLevels - 1 ];
        i < _endIndex[ numLevels - 1 ]; i++ )
  {
    _CDF.push_back( abs( _levels[ numLevels - 1 ][ i ] ) );
  }

  std::sort( _CDF.begin(), _CDF.end() );

}

//////////////////////////////////////////////////////////////////////
// Samples the amplitude at the given CDF fraction
// which should be in the interval [0,1]
//////////////////////////////////////////////////////////////////////
Real GaussianPyramid::sampleCDF( Real fraction,
                                 const FloatArray &CDF )
{
  int                  idx1, idx2;
  Real                 idx_real, idx1_real, idx2_real;
  Real                 blend1, blend2;

  idx_real = fraction * (Real)( CDF.size() - 1 );
  
  idx1 = (int)( fraction * (Real)( CDF.size() - 1 ) );
  idx2 = idx1 + 1;

  idx1_real = (Real)idx1;
  idx2_real = (Real)idx2;

  // Clamp the indices
  idx1 = max( 0, idx1 );
  idx1 = min( (int)( CDF.size() - 1 ), idx1 );

  idx2 = max( 0, idx2 );
  idx2 = min( (int)( CDF.size() - 1 ), idx2 );

  blend1 = idx2_real - idx_real;
  blend2 = 1.0 - blend1;

  return ( blend1 * CDF[ idx1 ] + blend2 * CDF[ idx2 ] );
}

//////////////////////////////////////////////////////////////////////
// Evalutes the inverse of the CDF function
//////////////////////////////////////////////////////////////////////
Real GaussianPyramid::sampleInverseCDF( Real amplitude,
                                        const FloatArray &CDF )
{
  Real             amplitudeFraction;
  int              amplitude_idx;
  int              amplitude_idx1, amplitude_idx2;
  Real             amplitude1, amplitude2, amplitude_diff;
  Real             amplitude_idx1_real, amplitude_idx2_real;
  Real             blend1, blend2;

  if ( amplitude < CDF[ 0 ] )
  {
    amplitudeFraction = 0.0;
  }
  else if ( amplitude >= CDF.back() )
  {
    amplitudeFraction = 1.0;
  }
  else
  {
    amplitude_idx1 = binarySearch( CDF, amplitude, 0, CDF.size() - 1 );
    amplitude_idx2 = amplitude_idx1 + 1;

    TRACE_ASSERT( amplitude_idx1 < CDF.size() - 1 && amplitude_idx1 >= 0,
                  "Bad index" );

    amplitude_idx1_real = (Real)amplitude_idx1;
    amplitude_idx2_real = (Real)amplitude_idx2;

    amplitude_idx1_real /= (Real)( CDF.size() - 1 );
    amplitude_idx2_real /= (Real)( CDF.size() - 1 );

    amplitude1 = CDF[ amplitude_idx1 ];
    amplitude2 = CDF[ amplitude_idx2 ];

    amplitude_diff = amplitude2 - amplitude1;

    blend1 = ( amplitude2 - amplitude ) / amplitude_diff;
    blend2 = 1.0 - blend1;

    amplitudeFraction = blend1 * amplitude_idx1_real
                      + blend2 * amplitude_idx2_real;
  }

  TRACE_ASSERT( amplitudeFraction >= 0.0 && amplitudeFraction <= 1.0,
                "Invalid amplitude fraction" );

  return amplitudeFraction;
}

//////////////////////////////////////////////////////////////////////
// Builds the signal nextLevel from baseLevel using the Gaussian
// pyramid process.
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::buildGaussianLevel( const FloatArray &baseLevel,
                                          FloatArray &nextLevel )
{
  if ( ( baseLevel.size() - 1 ) % 2 != 0 )
  {
    cerr << "ERROR: Level size is not a power of 2" << endl;
    abort();
  }

  int            nextSize = ( baseLevel.size() - 1 ) / 2 + 1;
  int            halfWidth = GAUSSIAN_STENCIL_SZ / 2;
  int            start_idx, end_idx, middle_idx;
  int            filter_start_idx, filter_end_idx;
  int            num_components;

  nextLevel.resize( nextSize, 0.0 );

  for ( int i = 0; i < nextLevel.size(); i++ )
  {
    nextLevel[ i ] = 0.0;

    middle_idx = i * 2;
    start_idx = max( 0, middle_idx - halfWidth );
    end_idx = min( (int)(baseLevel.size() - 1), middle_idx + halfWidth );

    filter_start_idx = start_idx - middle_idx + halfWidth;
    filter_end_idx = end_idx - middle_idx + halfWidth;

    num_components = end_idx - start_idx + 1;

    for ( int j = 0; j <= num_components; j++ )
    {
      nextLevel[ i ] += baseLevel[ start_idx + j ]
                      * GAUSSIAN_STENCIL[ filter_start_idx + j ];
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Pads an input signal to make its length a power of two
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::padInputSignal( const FloatArray &inputSignal,
                                      FloatArray &outputSignal,
                                      int &start_idx, int &end_idx,
                                      bool reflectBoundaries)
{
  int              extendedLength = 1;

  while ( extendedLength + 1 < inputSignal.size() )
  {
    extendedLength *= 2;
  }

  extendedLength += 1;

  outputSignal.resize( extendedLength, 0.0 );

  start_idx = ( extendedLength - inputSignal.size() ) / 2;
  end_idx = start_idx + inputSignal.size();

  for ( int i = 0; i < inputSignal.size(); i++ )
  {
    outputSignal[ start_idx + i ] = inputSignal[ i ];
  }

  if ( reflectBoundaries )
  {
    // Reflect the end points of the signal
    for ( int i = 0; i < start_idx; i++ )
    {
      outputSignal[ i ] = inputSignal[ start_idx - i ];
    }

    for ( int i = end_idx; i < extendedLength; i++ )
    {
      outputSignal[ i ]
        = outputSignal[ end_idx - ( i - end_idx ) - 2 ];
    }
  }
  else
  {
    for ( int i = 0; i < start_idx; i++ )
    {
      outputSignal[ i ] = 0.0;
    }

    for ( int i = end_idx; i < extendedLength; i++ )
    {
      outputSignal[ i ] = 0.0;
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Samples a signal at a non-integer index using linear
// interpolation.
//
// Returns 0 if the input exceeds the signal range.
// TODO: I guess this could have roundoff error issues
// if we start to consider really large signals
//////////////////////////////////////////////////////////////////////
Real GaussianPyramid::sampleSignal( const FloatArray &signal,
                                    Real index, bool &inside )
{
  Real           sampleIndex1 = floor( index );
  Real           sampleIndex2 = sampleIndex1 + 1.0;
  int            intIndex1 = (int)index;
  int            intIndex2 = intIndex1 + 1;

  Real           samplePoint1;
  Real           samplePoint2;
  Real           samplePoint;
  
  samplePoint1 = ( intIndex1 >= 0 && intIndex1 < signal.size() ) ?
                   signal[ intIndex1 ] : 0.0;
  samplePoint2 = ( intIndex2 >= 0 && intIndex2 < signal.size() ) ?
                   signal[ intIndex2 ] : 0.0;

  inside = ( intIndex1 >= 0 && intIndex1 < signal.size() )
        && ( intIndex2 >= 0 && intIndex2 < signal.size() );

  // Blend between them
  samplePoint = ( 1.0 - ( index - sampleIndex1 ) ) * samplePoint1
              + ( index - sampleIndex1 ) * samplePoint2;

  return samplePoint;
}

//////////////////////////////////////////////////////////////////////
// Returns the index to the entry in the sorted array
// whose value lies immediately below the given
// amplitude.
//////////////////////////////////////////////////////////////////////
int GaussianPyramid::binarySearch( const FloatArray &data, Real value,
                                   int start_idx, int end_idx )
{
  while ( start_idx <= end_idx )
  {
    int mid = ( start_idx + end_idx ) / 2;      // Compute mid point
    
    if ( value >= data[ mid ] && value < data[ mid + 1 ] )
    {
      return mid;
    }
    else if ( value >= data[ mid ] )
    {
      start_idx = mid + 1;
    }
    else
    {
      end_idx = mid - 1;
    }
  }

  return start_idx;
}

//////////////////////////////////////////////////////////////////////
// Copies the input signal to the bottom level of the pyramid
// and extends it so that its length is a power of two
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::copyInputSignal( const FloatArray &inputSignal,
                                       bool reflectBoundaries )
{
  padInputSignal( inputSignal, _levels[ 0 ],
                  _startIndex[ 0 ], _endIndex[ 0 ], reflectBoundaries );

#if 0
  int              extendedLength = 1;

  while ( extendedLength + 1 < inputSignal.size() )
  {
    extendedLength *= 2;
  }

  extendedLength += 1;

  _levels[ 0 ].resize( extendedLength );

  _startIndex[ 0 ] = ( extendedLength - inputSignal.size() ) / 2;
  _endIndex[ 0 ] = _startIndex[ 0 ] + inputSignal.size();

  for ( int i = 0; i < inputSignal.size(); i++ )
  {
    _levels[ 0 ][ _startIndex[ 0 ] + i ] = inputSignal[ i ];
  }

  if ( reflectBoundaries )
  {
    // Reflect the end points of the signal
    for ( int i = 0; i < _startIndex[ 0 ]; i++ )
    {
      _levels[ 0 ][ i ] = inputSignal[ _startIndex[ 0 ] - i ];
    }

    for ( int i = _endIndex[ 0 ]; i < extendedLength; i++ )
    {
      _levels[ 0 ][ i ]
        = _levels[ 0 ][ _endIndex[ 0 ] - ( i - _endIndex[ 0 ] ) - 2 ];
    }
  }
  else
  {
    for ( int i = 0; i < _startIndex[ 0 ]; i++ )
    {
      _levels[ 0 ][ i ] = 0.0;
    }

    for ( int i = _endIndex[ 0 ]; i < extendedLength; i++ )
    {
      _levels[ 0 ][ i ] = 0.0;
    }
  }
#endif
}

//////////////////////////////////////////////////////////////////////
// Computes all pyramid levels from the base signal
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::computeLevels()
{
  for ( int i = 1; i < _levels.size(); i++ )
  {
    buildGaussianLevel( _levels[ i - 1 ], _levels[ i ] );

    _startIndex[ i ] = _startIndex[ i - 1 ] / 2;

    if ( _startIndex[ i - 1 ] % 2 != 0 )
    {
      _startIndex[ i ] += 1;
    }

    _endIndex[ i ] = _endIndex[ i - 1 ] / 2;

    if ( _endIndex[ i - 1 ] % 2 != 0 )
    {
      _endIndex[ i ] += 1;
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Computes the power of the signal at the given level within the
// given window
//////////////////////////////////////////////////////////////////////
Real GaussianPyramid::windowPower( int level, int window )
{
  int            middle_idx = window * _windowHalfWidth;
  int            start_idx = max( 0, middle_idx - _windowHalfWidth );
  int            end_idx = min( (int)( _levels[ level ].size() - 1 ),
                                middle_idx + _windowHalfWidth );

  int            N = end_idx - start_idx + 1;

  Real           power = 0.0;

  for ( int i = start_idx; i <= end_idx; i++ )
  {
    power += _levels[ level ][ i ] * _levels[ level ][ i ];
  }

  power /= (Real)N;

  return power;
}

//////////////////////////////////////////////////////////////////////
// Computes the windowed power signal for the given level
//////////////////////////////////////////////////////////////////////
void GaussianPyramid::computeWindowPowerSignal( int level )
{
  int            numWindows = _levels[ level ].size() / _windowHalfWidth + 1;
  int            start_idx, middle_idx, end_idx;

  _windowLevels[ level ].resize( numWindows );

  _windowStartIndex[ level ] = numWindows;
  _windowEndIndex[ level ] = 0;

  for ( int window_idx = 0; window_idx < _windowLevels[ level ].size();
        window_idx++ )
  {
    _windowLevels[ level ][ window_idx ] = windowPower( level, window_idx );

    middle_idx = window_idx * _windowHalfWidth;
    start_idx = max( 0, middle_idx - _windowHalfWidth );
    end_idx = min( (int)( _levels[ level ].size() - 1 ),
                    middle_idx + _windowHalfWidth );

    if ( start_idx >= _startIndex[ level ] )
    {
      _windowStartIndex[ level ] = min( _windowStartIndex[ level ],
                                        window_idx );
    }

    if ( end_idx < _endIndex[ level ] )
    {
      _windowEndIndex[ level ] = max( _windowEndIndex[ level ], window_idx );
    }
  }
}
