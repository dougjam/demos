//////////////////////////////////////////////////////////////////////
// GaussianPyramid.h: Interface for the GaussianPyramid class
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

#ifndef GAUSSIAN_PYRAMID_H
#define GAUSSIAN_PYRAMID_H

#include <VECTOR.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <util/MERSENNETWISTER.h>

//////////////////////////////////////////////////////////////////////
// GaussianPyramid class
//
// Models a Gaussian pyramid for a one dimensional signal
//////////////////////////////////////////////////////////////////////
class GaussianPyramid {
	public:
		GaussianPyramid( const FloatArray &inputSignal, int numLevels,
                     bool reflectBoundaries = false );

		// Destructor
		virtual ~GaussianPyramid();

    const std::vector<FloatArray> &levels() const { return _levels; }

    std::vector<FloatArray> &levels() { return _levels; }

    const std::vector<FloatArray> &windowLevels() const { return _windowLevels; }

    std::vector<FloatArray> &windowLevels() { return _windowLevels; }

    // Helper class encoding a sample point somewhere
    // in the Gaussian pyramid, and a feature vector associated
    // with it.
    class FeaturePoint {
      public:
        FeaturePoint( int index )
          : _index( index )
        {
        }

        virtual ~FeaturePoint()
        {
        }

        int index() const
        {
          return _index;
        }

        int &index()
        {
          return _index;
        }

        const VECTOR &pos() const
        {
          return _features;
        }

        VECTOR &pos()
        {
          return _features;
        }

      private:
        int      _index;
        VECTOR   _features;
    };

    // Returns the features for all points at a given level in
    // the pyramid, using the given feature sizes for each level.
    // Sizes are assumed to specify the half width of the feature
    // in sample.
    //
    // Only computes features for points fully inside of the span
    // of the original signal.
    void buildLevelFeatures( vector<FeaturePoint> &features,
                             const IntArray &featureHalfWidths,
                             int level, Real falloff );

    // Same as the above, but builds a list of features for entries
    // in a windowed power signal.
    void buildPowerLevelFeatures( vector<FeaturePoint> &features,
                                  const IntArray &featureHalfWidths,
                                  int level );

    // This version computes features for windows of the given
    // half width.  Due to cost, features are only computed
    // using the current level and the one directly beneath it.
    void buildWindowFeatures( vector<FeaturePoint> &features,
                              const IntArray &windowHalfWidths,
                              const IntArray &featureHalfWidths,
                              int level, Real falloff = 0.0 );

    // Computes a feature vector for the given entry in the pyramid.
    // Returns true only if all samples were drawn from within the
    // span of the input signal.
    bool computeEntryFeature( FeaturePoint &featurePoint,
                              const IntArray &featureHalfWidths,
                              int level, int entry, Real falloff );

    // Same as the above, but computes features on entries of
    // the window power signal, rather than the signal itself
    bool computePowerEntryFeature( FeaturePoint &featurePoint,
                                   const IntArray &featureHalfWidths,
                                   int level, int entry );

    // This version computes a feature vector for a window
    // using only the current level and the one directly
    // beneath it.
    //
    // Falloff controls an optional exponential falloff
    // in sample weights as we move away from the
    // central sample index.
    //
    // If both inputCDF and outputCDF are non-null, we use
    // these to scale the amplitude for features entries, but only
    // if the feature entries occur at the bottom level of
    // the hierarchy
    bool computeWindowFeature( FeaturePoint &featurePoint,
                               const IntArray &windowHalfWidths,
                               const IntArray &featureHalfWidths,
                               int level, int window_idx,
                               Real falloff = 0.0,
                               const FloatArray *inputCDF = NULL,
                               const FloatArray *outputCDF = NULL,
                               Real *scaling = NULL,
                               Real alpha = 1.0 );

    // Reconstructs the input signal (just copy it from the bottom
    // level for a Gaussian pyramid)
    void reconstructSignal( FloatArray &signal );

    // Sets contents of a level to zero
    void zeroLevel( int level );

    // Divides each level of the pyramid in to overlapping windows of the
    // given half width and computes a signal, whose value for a given
    // window is given by the signal power in that window.
    void computeWindowedSignals( int windowHalfWidth );

    // Adds the signal from window window_input_idx in the input signal
    // to the window window_output_idx in the signal at the given level
    // in this pyramid.  Also updates the corresponding window power
    // term.
    void addWindowContribution( const FloatArray &inputSignal, int level,
                                int window_input_idx, int window_output_idx );

    // Conversion between Gaussian and Laplacian type
    void convertToLaplacian();

    void convertToGaussian();

    // Figures out the contents of ( level + 1 ) at index sample_idx
    // in level using the EXPAND operation from [Burt and Adelson]
    Real expand( int level, int sample_idx );

    // Initializes a cumulative distribution function for the
    // highest level in the pyramid.
    void initCDF();

    int startIndex( int level ) const
    {
      return _startIndex[ level ];
    }

    int endIndex( int level ) const
    {
      return _endIndex[ level ];
    }

    const FloatArray &CDF() const
    {
      return _CDF;
    }

    // Samples the amplitude at the given CDF fraction
    // which should be in the interval [0,1]
    static Real                sampleCDF( Real fraction,
                                          const FloatArray &CDF );

    // Evalutes the inverse of the CDF function
    static Real                sampleInverseCDF( Real amplitude,
                                                 const FloatArray &CDF );

    // Builds the signal nextLevel from baseLevel using the Gaussian
    // pyramid process.
    static void                buildGaussianLevel( const FloatArray &baseLevel,
                                                   FloatArray &nextLevel );

    // Pads an input signal to make its length a power of two
    static void                padInputSignal( const FloatArray &inputSignal,
                                               FloatArray &outputSignal,
                                               int &start_idx, int &end_idx,
                                               bool reflectBoundaries = false );

    // Samples a signal at a non-integer index using linear
    // interpolation.
    //
    // Returns 0 if the input exceeds the signal range.
    // TODO: I guess this could have roundoff error issues
    // if we start to consider really large signals
    static Real                sampleSignal( const FloatArray &signal,
                                             Real index, bool &inside );

    // Returns the index to the entry in the sorted array
    // whose value lies immediately below the given
    // amplitude.
    static int                 binarySearch( const FloatArray &data, Real value,
                                             int start_idx, int end_idx );

    // FIXME
    static FloatArray scalingVector;

	protected:

  private:
    // Copies the input signal to the bottom level of the pyramid
    // and extends it so that its length is a power of two
    void                       copyInputSignal( const FloatArray &inputSignal,
                                                bool reflectBoundaries = false );

    // Computes all pyramid levels from the base signal
    void                       computeLevels();

    // Computes the power of the signal at the given level within the
    // given window
    Real                       windowPower( int level, int window );

    // Computes the windowed power signal for the given level
    void                       computeWindowPowerSignal( int level );

	private:
    std::vector<FloatArray>    _levels;

    int                        _windowHalfWidth;
    std::vector<FloatArray>    _windowLevels;

    // Start and end sample indices for the original
    // signal (since we will extend it so that its
    // length is a power of 2)
    IntArray                   _startIndex;
    IntArray                   _endIndex;

    // Start and end indices for the windowed version
    // of the spectrum (ie. this specifies the range in
    // which windows are contained entirely inside the
    // original signal)
    IntArray                   _windowStartIndex;
    IntArray                   _windowEndIndex;

    // Whether or not this pyramid has been converted
    // to a Laplacian pyramid (ie. where each level stores
    // differences between scales in the signal).
    bool                       _isLaplacian;

    // Cumulative distribution function for the highest
    // pyramid level
    FloatArray                 _CDF;

    static int                 GAUSSIAN_STENCIL_SZ;
    static Real                GAUSSIAN_STENCIL[];

    static MERSENNETWISTER     generator;

  public:
    static Real                AMPLITUDE_CUTOFF;

};

#endif
