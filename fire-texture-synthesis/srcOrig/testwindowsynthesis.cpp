//////////////////////////////////////////////////////////////////////
// testwindowsynthesis.cpp: Windowed sound texture synthesis program
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

#include <SETTINGS.h>
#include <TYPES.h>

#include <STLUtil.h>

#include <GaussianPyramid.h>

#include <Parser.h>

#include <KDTreeMulti.h>

#include <WindowSynthesizer.h>

#include <IO.h>
#include <trace.h>

#include <fstream>
#include <iostream>

#include <vector>

using namespace std;

int main( int argc, const char **argv )
{
#if 0
  if ( argc < 3 )
  {
    printf( "Usage: %s <base signal vector> <training signal vector>\n",
            argv[0] );
    return 1;
  }
#endif

  const char          *configFile = ( argc > 1 ) ? argv[0] : "default.xml";

  Parser              *parser = Parser::buildParser( configFile );

  if ( !parser )
  {
    printf( "ERROR: Could not build parser from %s\n", configFile );
  }

  Parser::SynthesisParameters
                       synthesisParms = parser->getSynthesisParms();

  FloatArray           baseSignal, trainingSignal;
  FloatArray           outputSignal;

#if 0
  // Hard code this stuff for now
  //int                  numLevels = 11;
  int                  numLevels = 6;
  //int                  numLevels = 5;
  //int                  numLevels = 10;
  int                  Fs = 44100;
  //int                  Fs = 16000;
  int                  numBaseLevels;
  IntArray             filterHalfWidths( numLevels );
  IntArray             windowHalfWidths( numLevels );

  bool                 useLaplacian = false;
  bool                 useANN = true;
  Real                 epsANN = 1.0;

  Real                 falloff = 0.06;
#endif

  int                  numLevels = synthesisParms._numLevels;
  int                  Fs = synthesisParms._Fs;

  IntArray             filterHalfWidths( numLevels );
  IntArray             windowHalfWidths( numLevels );

  bool                 useLaplacian = false;
  bool                 useANN = true;
  Real                 epsANN = synthesisParms._epsANN;

  Real                 falloff = synthesisParms._falloff;

  bool                 scaleCDF = synthesisParms._scaleCDF;

  Real                 scalingAlpha = synthesisParms._scalingAlpha;

  int                  numBaseLevels = 1;

  // Create a result directory
  int                  id;
  char                 subdirName[ 1024 ];
  char                 copyCmd[ 1024 ];
  char                 outputPrefix[ 1024 ];
  char                 inputPrefix[ 1024 ];

  for ( id = 0; ; id++ )
  {
    sprintf( subdirName, "testwindowsynthesis-%03d", id );
    if ( IO::create_dir( subdirName ) == IO::OK )
      break;
  }

  sprintf( copyCmd, "cp %s %s", configFile, subdirName );
  system( copyCmd );

  ofstream reg;
  reg.open( "testwindowsynthesis_registry.txt", ios::app );
  if ( reg.fail() ) reg.open( "testwindowsynthesis_registry.txt", ios::out );
  TRACE_ASSERT( reg.good() );
  reg.width(3);
  reg.fill('0');
  reg << id << endl <<
    "\t\t" << SDUMP( numLevels ) << endl << 
    "\t\t" << SDUMP( Fs ) << endl << 
    "\t\t" << SDUMP( synthesisParms._windowHW ) << endl << 
    "\t\t" << SDUMP( synthesisParms._featureHW ) << endl << 
    "\t\t" << SDUMP( epsANN ) << endl << 
    "\t\t" << SDUMP( falloff ) << endl << 
    "\t\t" << SDUMP( synthesisParms._baseSignal ) << endl << 
    "\t\t" << SDUMP( synthesisParms._trainingSignal ) << endl <<
    "\t\t" << SDUMP( synthesisParms._scaleCDF ) << endl <<
    "\t\t" << SDUMP( synthesisParms._scalingAlpha ) << endl
    << endl << endl << endl;
  reg.close();

#if 0
  readRealVector( argv[1], baseSignal );
  readRealVector( argv[2], trainingSignal );
#endif
  readRealVector( synthesisParms._baseSignal.c_str(), baseSignal );
  readRealVector( synthesisParms._trainingSignal.c_str(), trainingSignal );

  printf( "Building synthesizer\n" );
  WindowSynthesizer synthesizer( trainingSignal, baseSignal, numLevels, Fs );

#if 0
  numBaseLevels = 1;
#endif

  cout << SDUMP( numBaseLevels ) << endl;

#if 0
  for ( int i = 0; i < numLevels; i++ )
  {
    windowHalfWidths[ i ] = 1;
    filterHalfWidths[ i ] = 12;
    //windowHalfWidths[ i ] = 2;
    //filterHalfWidths[ i ] = 6;
    //windowHalfWidths[ i ] = 3;
    //filterHalfWidths[ i ] = 3;
    //windowHalfWidths[ i ] = 4;
    //filterHalfWidths[ i ] = 2;
  }
  //filterHalfWidths[ numLevels - 1 ] = 1;
#endif
  for ( int i = 0; i < numLevels; i++ )
  {
    windowHalfWidths[ i ] = synthesisParms._windowHW;
    filterHalfWidths[ i ] = synthesisParms._featureHW;
  }

  printf( "Synthesizing signal\n" );
  synthesizer.synthesizeSignal( windowHalfWidths, filterHalfWidths,
                                numBaseLevels, useLaplacian,
                                useANN, epsANN, falloff, scaleCDF,
                                scalingAlpha );
  printf( "... done\n\n" );

  synthesizer.reconstructSignal( outputSignal );

  sprintf( outputPrefix, "%s/output", subdirName );
  sprintf( inputPrefix, "%s/input", subdirName );

  synthesizer.writeAllLevels( outputPrefix, inputPrefix );

  return 0;
}
