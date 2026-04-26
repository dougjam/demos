//////////////////////////////////////////////////////////////////////
// TextureSynthesizer.cpp: Implementation
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

#include "TextureSynthesizer.h"

#include <IO.h>
#include <trace.h>

#include <STLUtil.h>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////
TextureSynthesizer::TextureSynthesizer( const FloatArray &trainingSignal,
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
TextureSynthesizer::~TextureSynthesizer()
{
  clear();
}

//////////////////////////////////////////////////////////////////////
// Synthesizes a time domain signal from the base signal using
// the stored training data.  Requires the nearest neighbour
// half width at each level in the hierarchy as well as the
// number of levels to preserve from the input signal.
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::synthesizeSignal( const IntArray &filterHalfWidths,
                                           int numBaseLevels,
                                           bool useANN, Real epsANN,
                                           Real falloff )
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

  buildLevelFeatures( filterHalfWidths, levelFeatures,
                      levelFeaturePtrs, falloff );

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
    printf( "TextureSynthesizer: Extending level %d\n", level_idx );

    extendLevel( filterHalfWidths, levelFeatures,
                 levelFeatureTrees, level_idx,
                 useANN, epsANN, &levelFeatureTreesANN, falloff );
  }
  printf( "TextureSynthesizer: Done extending levels\n" );

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
// Same as the above, but does the synthesis by dividing each
// level of the pyramids in to overlapping windows of the given
// half width.  Synthesis is then done by building feature vectors
// out of window powers.
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::synthesizeWindowedSignal(
                               const IntArray &filterHalfWidths,
                               int numBaseLevels,
                               int windowHalfWidth )
{
  vector<FeatureList>        levelFeatures( _numLevels );
  vector<FeaturePtrList>     levelFeaturePtrs( _numLevels );
  vector<FeatureTree *>      levelFeatureTrees( _numLevels );

  clear();

  // Build our pyramids
  _trainingPyramid = new GaussianPyramid( _trainingSignal, _numLevels,
                                          true /* reflect boundaries */ );

  _signalPyramid = new GaussianPyramid( _baseSignal, _numLevels );

  // Clear all levels in the signal tree below the base levels
  for ( int level_idx = 0; level_idx < _numLevels - numBaseLevels; level_idx++ )
  {
    _signalPyramid->zeroLevel( level_idx );
  }

  _trainingPyramid->computeWindowedSignals( windowHalfWidth );
  _signalPyramid->computeWindowedSignals( windowHalfWidth );

  buildPowerLevelFeatures( filterHalfWidths, levelFeatures, levelFeaturePtrs );

  buildLevelTrees( levelFeaturePtrs, levelFeatureTrees );

  // Extend each level
  for ( int level_idx = _numLevels - 1 - numBaseLevels;
        level_idx >= 0; level_idx-- )
  {
    printf( "TextureSynthesizer: Extending level %d\n", level_idx );

    extendLevelWindowed( filterHalfWidths, levelFeatures,
                         levelFeatureTrees, level_idx );
  }
  printf( "TextureSynthesizer: Done extending levels\n" );
}

//////////////////////////////////////////////////////////////////////
// Reconstructs a synthesized signal, assuming synthesizeSignal
// has already been called.
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::reconstructSignal( FloatArray &finalSignal )
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
int TextureSynthesizer::pickNumBaseLevels( Real sampleCutoff )
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
void TextureSynthesizer::writeAllLevels( const char *prefix )
{
  char           buf[ 1024 ];

  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    sprintf( buf, "%s_level_%d.vector", prefix, level_idx );

    writeRealVector( buf, _signalPyramid->levels()[ level_idx ] );
  }
}

//////////////////////////////////////////////////////////////////////
// Constructs the feature vectors at each sample and each
// level in the training signal
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::buildLevelFeatures(
                            const IntArray &filterHalfWidths,
                            vector<FeatureList> &levelFeatures,
                            vector<FeaturePtrList> &levelFeaturePtrs,
                            Real falloff )
{
  // Build feature lists and trees for each level
  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    printf( "TextureSynthesizer: Building features for level %d\n", level_idx );

    _trainingPyramid->buildLevelFeatures( levelFeatures[ level_idx ],
                                          filterHalfWidths, level_idx,
                                          falloff );

    for ( int feature_idx = 0; feature_idx < levelFeatures[ level_idx ].size();
          feature_idx++ )
    {
      levelFeaturePtrs[ level_idx ].push_back(
            &(levelFeatures[ level_idx ][ feature_idx ] ) );
    }
  }
  printf( "TextureSynthesizer: Done building features\n" );
}

//////////////////////////////////////////////////////////////////////
// Builds trees using the ANN library
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::buildANNTrees( vector<FeatureList> &levelFeatures,
                                        vector<ANNbd_tree *> &levelTrees,
                                        vector<ANNpointArray> &ANNdataPoints )
{
  // Build a tree for each level
  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    int featureDimensions = levelFeatures[ level_idx ][ 0 ].pos().size();
    int levelSize = (int)levelFeatures[ level_idx ].size();

    printf( "TextureSynthesizer: Building ANN tree for level %d", level_idx );
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
// Same as the above, but builds features using the windowed power
// signals stored in the synthesis pyramids
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::buildPowerLevelFeatures(
                         const IntArray &filterHalfWidths,
                         std::vector<FeatureList> &levelFeatures,
                         std::vector<FeaturePtrList> &levelFeaturePtrs )
{
  // Build feature lists and trees for each level
  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    printf( "TextureSynthesizer: Building features for level %d\n", level_idx );

    _trainingPyramid->buildPowerLevelFeatures( levelFeatures[ level_idx ],
                                               filterHalfWidths, level_idx );

    for ( int feature_idx = 0; feature_idx < levelFeatures[ level_idx ].size();
          feature_idx++ )
    {
      levelFeaturePtrs[ level_idx ].push_back(
            &(levelFeatures[ level_idx ][ feature_idx ] ) );
    }
  }
  printf( "TextureSynthesizer: Done building features\n" );
}

//////////////////////////////////////////////////////////////////////
// Constructs the nearest neighbour search tree for the
// set of features stored at each level in the training
// pyramid
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::buildLevelTrees(
                            vector<FeaturePtrList> &levelFeaturePtrs,
                            vector<FeatureTree *> &levelTrees )
{
  // Build a tree for each level
  for ( int level_idx = 0; level_idx < _numLevels; level_idx++ )
  {
    int featureDimensions = levelFeaturePtrs[ level_idx ][ 0 ]->pos().size();

    printf( "TextureSynthesizer: Building feature tree for level %d",
            level_idx );
    printf( " with %d samples", (int)levelFeaturePtrs[ level_idx ].size() );
    printf( " and %d dimensions\n", featureDimensions );

    levelTrees[ level_idx ] = new FeatureTree( featureDimensions );

    levelTrees[ level_idx ]->build( levelFeaturePtrs[ level_idx ] );
  }
  printf( "TextureSynthesizer: Done building trees\n" );
}

//////////////////////////////////////////////////////////////////////
// Uses a texture synthesis procedure to extend the contents
// of a level of the Gaussian pyramid
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::extendLevel(
                  const IntArray &filterHalfWidths,
                  const vector<FeatureList> &levelFeatures,
                  const vector<FeatureTree *> &levelTrees,
                  int level,
                  bool useANN, Real epsANN,
                  std::vector<ANNbd_tree *> *annTrees,
                  Real falloff )
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

  if ( useANN )
  {
    neighbourTreeANN = annTrees->at( level );
    nnIdx = new ANNidx[1];
    nnDists = new ANNdist[1];
    annQuery = annAllocPt( levelFeatures[ level ][ 0 ].pos().size() );
  }

  GaussianPyramid::FeaturePoint        featurePoint( -1 );
  const GaussianPyramid::FeaturePoint *nearest;

  for ( int sample_idx = 0; sample_idx < levelData.size(); sample_idx++ )
  {
    if ( sample_idx % 100 == 0 )
    {
      percentDone = (Real)sample_idx / (Real)levelData.size();
      percentDone *= 100.0;

      printf( "Synthesizing level %d: %2.2f %% done\r", level, percentDone );
    }

    // Build a feature vector for this sample
    _signalPyramid->computeEntryFeature( featurePoint, filterHalfWidths,
                                         level, sample_idx, falloff );

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

    levelData[ sample_idx ] = trainingData[ training_idx ];
  }
  printf( "\n" );
}

//////////////////////////////////////////////////////////////////////
// Same as the above, but does this with windowed power signals,
// rather than copying input samples directly.
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::extendLevelWindowed(
                          const IntArray &filterHalfWidths,
                          const std::vector<FeatureList> &levelFeatures,
                          const std::vector<FeatureTree *> &levelTrees,
                          int level )
{
  FloatArray        &levelData = _signalPyramid->windowLevels()[ level ];
  const FloatArray  &trainingData = _trainingPyramid->levels()[ level ];
  const FeatureTree *neighbourTree = levelTrees[ level ];
  int                training_window_idx;
  Real               percentDone;

  GaussianPyramid::FeaturePoint        featurePoint( -1 );
  const GaussianPyramid::FeaturePoint *nearest;

  for ( int sample_idx = 0; sample_idx < levelData.size(); sample_idx++ )
  {
#if 0
    if ( sample_idx % 100 == 0 )
    {
#endif
      percentDone = (Real)sample_idx / (Real)levelData.size();
      percentDone *= 100.0;

      printf( "Synthesizing level %d: %2.2f %% done\r", level, percentDone );
#if 0
    }
#endif

    // Build a feature vector for this sample
    _signalPyramid->computePowerEntryFeature(
                                         featurePoint, filterHalfWidths,
                                         level, sample_idx );

    // Find the closest feature to this one in the
    // training data at this level
    nearest = neighbourTree->nearestNeighbour( featurePoint.pos() );

    TRACE_ASSERT( nearest, "Nearest neighbour returned NULL" );

    training_window_idx = nearest->index();

    // TODO: Copy window data
    _signalPyramid->addWindowContribution( trainingData, level,
                                           training_window_idx, sample_idx );
  }
  printf( "\n" );
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::copyANNPoint( const VECTOR &input, ANNpoint output )
{
  for ( int i = 0; i < input.size(); i++ )
  {
    output[i] = input(i);
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void TextureSynthesizer::clear()
{
  delete _trainingPyramid;
  delete _signalPyramid;

  _trainingPyramid = NULL;
  _signalPyramid = NULL;
}
