//////////////////////////////////////////////////////////////////////
// WindowSynthesizer.cpp: Implementation
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

#include "WindowSynthesizer.h"

#include <IO.h>
#include <trace.h>

#include <STLUtil.h>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
WindowSynthesizer::WindowSynthesizer( const FloatArray &trainingSignal,
                                      const FloatArray &baseSignal,
                                      int numLevels, int Fs )
  : _trainingSignal( trainingSignal ),
    _baseSignal( baseSignal ),
    _numLevels( numLevels ),
    _trainingPyramid( NULL ),
    _signalPyramid( NULL ),
    _Fs( Fs )
{
}

//////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////
WindowSynthesizer::~WindowSynthesizer()
{
  clear();
}

//////////////////////////////////////////////////////////////////////
// Synthesizes a time domain signal from the base signal using
// the stored training data.  Requires the nearest neighbour
// half width at each level in the hierarchy as well as the
// number of levels to preserve from the input signal.
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::synthesizeSignal( const IntArray &windowHalfWidths,
                                          const IntArray &filterHalfWidths,
                                          int numBaseLevels,
                                          bool useLaplacian,
                                          bool useANN, Real epsANN,
                                          Real falloff,
                                          bool scaleCDF,
                                          Real scalingAlpha )
{
  vector<FeatureList>        levelFeatures( _numLevels );
  vector<FeaturePtrList>     levelFeaturePtrs( _numLevels );
  vector<FeatureTree *>      levelFeatureTrees( _numLevels );
  vector<ANNbd_tree *>       levelFeatureTreesANN( _numLevels );
  vector<ANNpointArray>      levelPointsANN( _numLevels );

  clear();

  // Build our pyramids
  _trainingPyramid = new GaussianPyramid( _trainingSignal, _numLevels,
                                          true /* reflect boundaries */ );

  _signalPyramid = new GaussianPyramid( _baseSignal, _numLevels );

  if ( scaleCDF )
  {
    // Initialize cumulative distribution functions for
    // amplitudes at the top level in each pyramid.
    _trainingPyramid->initCDF();
    _signalPyramid->initCDF();
  }

  if ( useLaplacian )
  {
    _trainingPyramid->convertToLaplacian();
    _signalPyramid->convertToLaplacian();
  }

  buildLevelFeatures( windowHalfWidths, filterHalfWidths,
                      levelFeatures, levelFeaturePtrs, falloff );

  if ( useANN )
  {
    buildANNTrees( levelFeatures, levelFeatureTreesANN, levelPointsANN );
  }
  else
  {
    buildLevelTrees( levelFeaturePtrs, levelFeatureTrees );
  }

  // Clear all levels in the signal tree below the base levels
  for ( int level_idx = 0; level_idx < _numLevels - numBaseLevels; level_idx++ )
  {
    _signalPyramid->zeroLevel( level_idx );
  }

  // Extend each level
  for ( int level_idx = _numLevels - 1 - numBaseLevels;
        level_idx >= 0; level_idx-- )
  {
    printf( "WindowSynthesizer: Extending level %d\n", level_idx );

    extendLevel( windowHalfWidths, filterHalfWidths, levelFeatures,
                 levelFeatureTrees, level_idx,
                 useANN, epsANN, &levelFeatureTreesANN, falloff, scaleCDF,
                 scalingAlpha );
  }
  printf( "WindowSynthesizer: Done extending levels\n" );

  if ( useLaplacian )
  {
    _trainingPyramid->convertToGaussian();
    _signalPyramid->convertToGaussian();
  }

  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    if ( useANN )
    {
      delete levelFeatureTreesANN[ level_idx ];
    }
    else
    {
      delete levelFeatureTrees[ level_idx ];
    }
  }
}

//////////////////////////////////////////////////////////////////////
// This version also requires a lowpass filtered version of
// the training signal.  It builds the base level of the 
// signal out of this filtered version, and then all subsequent
// levels are built out fo the difference signal, obtained by
// subtracting the low pass filtered signal from the original.
// This allows us to only synthesize the high frequency part
// of a signal.
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::synthesizeSignal( const FloatArray &trainingSignalLP,
                                          const IntArray &windowHalfWidths,
                                          const IntArray &filterHalfWidths,
                                          bool useANN, Real epsANN,
                                          Real falloff,
                                          bool scaleCDF,
                                          Real scalingAlpha )
{
  vector<FeatureList>        levelFeatures( _numLevels );
  vector<FeaturePtrList>     levelFeaturePtrs( _numLevels );
  vector<FeatureTree *>      levelFeatureTrees( _numLevels );
  vector<ANNbd_tree *>       levelFeatureTreesANN( _numLevels );
  vector<ANNpointArray>      levelPointsANN( _numLevels );

  FloatArray                 trainingDifference;
  GaussianPyramid           *lowPassPyramid;

  const int                  numBaseLevels = 1;

  TRACE_ASSERT( trainingSignalLP.size() == _trainingSignal.size(),
                "Signal size mismatch" );

  trainingDifference.resize( trainingSignalLP.size(), 0.0 );

  for ( int i = 0; i < trainingSignalLP.size(); i++ )
  {
    trainingDifference[ i ] = _trainingSignal[ i ] - trainingSignalLP[ i ];
  }

  clear();

  // Build our pyramids
  _trainingPyramid = new GaussianPyramid( trainingDifference, _numLevels,
                                          true /* reflect boundaries */ );

  _signalPyramid = new GaussianPyramid( _baseSignal, _numLevels );

  lowPassPyramid = new GaussianPyramid( trainingSignalLP, _numLevels,
                                        true /* reflect boundaries */ );

  // Steal the top level from the low pass pyramid
  TRACE_ASSERT( lowPassPyramid->levels().back().size()
                == _trainingPyramid->levels().back().size(),
                "Top level mismatch" );

  for ( int i = 0; i < lowPassPyramid->levels().back().size(); i++ )
  {
    _trainingPyramid->levels().back()[ i ]
      = lowPassPyramid->levels().back()[ i ];
  }

  // Build CDF information *after* we have copied the
  // low frequency signal in to the base level
  if ( scaleCDF )
  {
    // Initialize cumulative distribution functions for
    // amplitudes at the top level in each pyramid.
    _trainingPyramid->initCDF();
    _signalPyramid->initCDF();
  }

  buildLevelFeatures( windowHalfWidths, filterHalfWidths,
                      levelFeatures, levelFeaturePtrs, falloff );

  if ( useANN )
  {
    buildANNTrees( levelFeatures, levelFeatureTreesANN, levelPointsANN );
  }
  else
  {
    buildLevelTrees( levelFeaturePtrs, levelFeatureTrees );
  }

  // Clear all levels in the signal tree below the base levels
  for ( int level_idx = 0; level_idx < _numLevels - numBaseLevels; level_idx++ )
  {
    _signalPyramid->zeroLevel( level_idx );
  }

  // Extend each level
  for ( int level_idx = _numLevels - 1 - numBaseLevels;
        level_idx >= 0; level_idx-- )
  {
    printf( "WindowSynthesizer: Extending level %d\n", level_idx );

    extendLevel( windowHalfWidths, filterHalfWidths, levelFeatures,
                 levelFeatureTrees, level_idx,
                 useANN, epsANN, &levelFeatureTreesANN, falloff, scaleCDF,
                 scalingAlpha );
  }
  printf( "WindowSynthesizer: Done extending levels\n" );

  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    if ( useANN )
    {
      delete levelFeatureTreesANN[ level_idx ];
    }
    else
    {
      delete levelFeatureTrees[ level_idx ];
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Reconstructs a synthesized signal, assuming synthesizeSignal
// has already been called.
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::reconstructSignal( FloatArray &finalSignal )
{
  // If we are using a Gaussian pyramid, this is done by simply
  // copying the signal from the bottom level of the pyramid.
  _signalPyramid->reconstructSignal( finalSignal );
}

//////////////////////////////////////////////////////////////////////
// Picks the number of base levels to use according to the sampling
// frequency for this synthesizer.  All levels with sampling frequencies
// below the given cutoff will be retained.
//////////////////////////////////////////////////////////////////////
int WindowSynthesizer::pickNumBaseLevels( Real sampleCutoff )
{
  int            baseLevels = 0;
  Real           Fs = (Real)_Fs;

  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    if ( Fs < sampleCutoff )
    {
      baseLevels++;
    }

    Fs /= 2.0;
  }

  return baseLevels;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::writeAllLevels( const char *outputPrefix,
                                        const char *inputPrefix )
{
  char           buf[ 1024 ];

  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    sprintf( buf, "%s_level_%d.vector", outputPrefix, level_idx );

    writeRealVector( buf, _signalPyramid->levels()[ level_idx ] );

    sprintf( buf, "%s_level_%d.vector", inputPrefix, level_idx );

    writeRealVector( buf, _trainingPyramid->levels()[ level_idx ] );
  }

  // Also write the top level, truncted to the start
  // and end indices
  FloatArray finalOutput, finalOutput_nontruncated;

  for ( int i = _signalPyramid->startIndex( 0 );
        i < _signalPyramid->endIndex( 0 ); i++ )
  {
    if ( abs( _baseSignal[ i - _signalPyramid->startIndex( 0 ) ] )
          < GaussianPyramid::AMPLITUDE_CUTOFF )
    {
      finalOutput.push_back( 0.0 );
    }
    else
    {
      finalOutput.push_back( _signalPyramid->levels()[ 0 ][ i ] );
    }

    finalOutput_nontruncated.push_back( _signalPyramid->levels()[ 0 ][ i ] );
  }

  sprintf( buf, "%s_final.vector", outputPrefix );


  sprintf( buf, "%s_final_noclip.vector", outputPrefix );

  writeRealVector( buf, finalOutput_nontruncated );

  writeRealVector( "scaling.vector", GaussianPyramid::scalingVector );

  writeRealVector( "signalCDF.vector", _signalPyramid->CDF() );
  writeRealVector( "trainingCDF.vector", _trainingPyramid->CDF() );
}

//////////////////////////////////////////////////////////////////////
// Constructs the feature vectors at each sample and each
// level in the training signal
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::buildLevelFeatures(
                            const IntArray &windowHalfWidths,
                            const IntArray &filterHalfWidths,
                            vector<FeatureList> &levelFeatures,
                            vector<FeaturePtrList> &levelFeaturePtrs,
                            Real falloff )
{
  // Build feature lists and trees for each level
  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    printf( "WindowSynthesizer: Building features for level %d\n", level_idx );

    _trainingPyramid->buildWindowFeatures( levelFeatures[ level_idx ],
                                           windowHalfWidths,
                                           filterHalfWidths, level_idx,
                                           falloff );

    for ( int feature_idx = 0; feature_idx < levelFeatures[ level_idx ].size();
          feature_idx++ )
    {
      levelFeaturePtrs[ level_idx ].push_back(
            &(levelFeatures[ level_idx ][ feature_idx ] ) );
    }
  }
  printf( "WindowSynthesizer: Done building features\n" );
}

//////////////////////////////////////////////////////////////////////
// Constructs the nearest neighbour search tree for the
// set of features stored at each level in the training
// pyramid
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::buildLevelTrees(
                            vector<FeaturePtrList> &levelFeaturePtrs,
                            vector<FeatureTree *> &levelTrees )
{
  // Build a tree for each level
  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    int featureDimensions = levelFeaturePtrs[ level_idx ][ 0 ]->pos().size();

    printf( "WindowSynthesizer: Building feature tree for level %d",
            level_idx );
    printf( " with %d samples", (int)levelFeaturePtrs[ level_idx ].size() );
    printf( " and %d dimensions\n", featureDimensions );

    levelTrees[ level_idx ] = new FeatureTree( featureDimensions );

    levelTrees[ level_idx ]->build( levelFeaturePtrs[ level_idx ] );
  }
  printf( "WindowSynthesizer: Done building trees\n" );
}

//////////////////////////////////////////////////////////////////////
// Builds trees using the ANN library
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::buildANNTrees( vector<FeatureList> &levelFeatures,
                                       vector<ANNbd_tree *> &levelTrees,
                                       vector<ANNpointArray> &ANNdataPoints )
{
  // Build a tree for each level
  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    int featureDimensions = levelFeatures[ level_idx ][ 0 ].pos().size();
    int levelSize = (int)levelFeatures[ level_idx ].size();

    printf( "WindowSynthesizer: Building ANN tree for level %d", level_idx );
    printf( " with %d samples", levelSize );
    printf( " and %d dimensions\n", featureDimensions );

    ANNdataPoints[ level_idx ] = annAllocPts( levelSize, featureDimensions );

    // Copy each point
    for ( int feature_idx = 0; feature_idx < levelSize; feature_idx++ )
    for ( int feature_dim = 0; feature_dim < featureDimensions; feature_dim++ )
    {
      ANNdataPoints[ level_idx ][ feature_idx ][ feature_dim ]
        = levelFeatures[ level_idx ][ feature_idx ].pos()( feature_dim );
    }

    levelTrees[ level_idx ] = new ANNbd_tree( ANNdataPoints[ level_idx ],
                                              levelSize,
                                              featureDimensions );
  }
}

//////////////////////////////////////////////////////////////////////
// Uses a texture synthesis procedure to extend the contents
// of a level of the Gaussian pyramid
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::extendLevel(
                  const IntArray &windowHalfWidths,
                  const IntArray &filterHalfWidths,
                  const vector<FeatureList> &levelFeatures,
                  const vector<FeatureTree *> &levelTrees,
                  int level,
                  bool useANN, Real epsANN,
                  std::vector<ANNbd_tree *> *annTrees,
                  Real falloff,
                  bool scaleCDF,
                  Real scalingAlpha )
{
  FloatArray        &levelData = _signalPyramid->levels()[ level ];
  const FloatArray  &trainingData = _trainingPyramid->levels()[ level ];
  const FeatureTree *neighbourTree = levelTrees[ level ];
  int                training_idx;
  Real               percentDone;
  ANNbd_tree        *neighbourTreeANN = NULL;
  ANNidxArray        nnIdx;
  ANNdistArray       nnDists;
  ANNpoint           annQuery;
  Real               scale;

  if ( useANN )
  {
    neighbourTreeANN = annTrees->at( level );
    nnIdx = new ANNidx[1];
    nnDists = new ANNdist[1];
    annQuery = annAllocPt( levelFeatures[ level ][ 0 ].pos().size() );
  }

  GaussianPyramid::FeaturePoint        featurePoint( -1 );
  const GaussianPyramid::FeaturePoint *nearest;

  int                numWindows, window_start, window_end;
  
  numWindows = levelData.size() / windowHalfWidths[ level ] + 1;

  for ( int window_idx = 0; window_idx < numWindows; window_idx++ )
  {
    if ( window_idx % 10 == 0 )
    {
      percentDone = (Real)window_idx / (Real)numWindows;
      percentDone *= 100.0;

      printf( "Synthesizing level %d: %2.2f %% done\r", level, percentDone );
    }

    scale = 1.0;

    // Build a feature vector for this sample
    if ( scaleCDF )
    {
      _signalPyramid->computeWindowFeature( featurePoint,
                                            windowHalfWidths, filterHalfWidths,
                                            level, window_idx, falloff,
                                            &( _trainingPyramid->CDF() ),
                                            &( _signalPyramid->CDF() ),
                                            &scale, scalingAlpha );

      if ( isnan( scale ) )
      {
        cout << "ERROR: Invalid scaling" << endl;
        scale = 1.0;
      }
    }
    else
    {
      _signalPyramid->computeWindowFeature( featurePoint,
                                            windowHalfWidths, filterHalfWidths,
                                            level, window_idx, falloff );
    }

    // Find the closest feature to this one in the
    // training data at this level
    if ( useANN )
    {
      // Copy the query point
      copyANNPoint( featurePoint.pos(), annQuery );

      neighbourTreeANN->annkSearch( annQuery, 1, nnIdx, nnDists, epsANN );

      TRACE_ASSERT( nnIdx[0] >= 0 && nnIdx[0] < levelFeatures[ level ].size(),
                    "Nearest neighbour out of range" );

      nearest = &( levelFeatures[ level ][ nnIdx[ 0 ] ] );
    }
    else
    {
      nearest = neighbourTree->nearestNeighbour( featurePoint.pos() );
    }

    TRACE_ASSERT( nearest, "Nearest neighbour returned NULL" );

    training_idx = nearest->index();

#if 0
    levelData[ sample_idx ] = trainingData[ training_idx ];
#endif

    blendWindowData( trainingData, training_idx, levelData, window_idx,
                     windowHalfWidths[ level ] );
  }
  printf( "\n" );

  if ( useANN )
  {
    delete[] nnIdx;
    delete[] nnDists;
  }
}

//////////////////////////////////////////////////////////////////////
// Blends the given window from an input signal in to the
// given input of an output signal using a simple hat function
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::blendWindowData(
                             const FloatArray &input, int input_window,
                             FloatArray &output, int output_window,
                             int windowHalfWidth, Real scale )
{
  int            window_start_input = windowHalfWidth * ( input_window - 1 );
  int            window_start_output = windowHalfWidth * ( output_window - 1 );
  int            windowSize = 2 * windowHalfWidth + 1;
  Real           blendValue;

  for ( int i = 0; i < windowSize; i++ )
  {
    blendValue = (Real)windowHalfWidth - (Real)abs( windowHalfWidth - i );
    blendValue /= (Real)windowHalfWidth;

    if ( window_start_output + i < 0 || window_start_output + i >= output.size()
      || window_start_input + i < 0 || window_start_input + i >= input.size() )
    {
      continue;
    }

    blendValue *= scale;

    output[ window_start_output + i ]
      += blendValue * input[ window_start_input + i ];
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::copyANNPoint( const VECTOR &input, ANNpoint output )
{
  for ( int i = 0; i < input.size(); i++ )
  {
    output[i] = input(i);
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void WindowSynthesizer::clear()
{
  delete _trainingPyramid;
  delete _signalPyramid;

  _trainingPyramid = NULL;
  _signalPyramid = NULL;
}
