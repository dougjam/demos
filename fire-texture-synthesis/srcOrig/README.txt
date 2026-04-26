//////////////////////////////////////////////////////////////////////
// Compilation:
//////////////////////////////////////////////////////////////////////

On Linux systems with SCons 1.2 or later installed, SCons may be used with
the provided SConstruct file to compile the 'testwindowsynthesis' binary.
Running the 'scons' command in the project root directory should build the
'testwindowsynthesis' binary and place it in the work directory.

Compilation requires the ANN nearest neighbour search library, found here:
  http://www.cs.umd.edu/~mount/ANN/
The provided SConstruct file requires the following environment variables
for ANN:
  ANNINC=<ANN install directory>/include
  ANNLIB=<ANN install directory>/lib
Windows binaries for ANN are also available, and it should be straightforward
to include the necessary include paths and libraries in a Visual Studio
project.

The 'testwindowsynthesis' binary may be compiled using Intel C++ compilers
(if installed) by invoking 'scons icc=1'.

The Intel MKL library may optionally be used for vector operations by
invoking 'scons mkl=1'.  In this case, the user must set the following
environment variables:
  MKLINC=/opt/intel/mkl/10.3/mkl/include (for example)
  MKLLIB=/opt/intel/mkl/10.3/mkl/lib/intel64 (for example)
If compiling on Windows, use linearalgebra/VECTOR_FAST.cpp if the Intel MKL
library is installed.  Otherwise, use linearalgebra/VECTOR_DEBUG.cpp.

This software also makes use of the TinyXML library for reading configuration
files, but the source for this is included with our source code:
  http://www.grinninglizard.com/tinyxml/


//////////////////////////////////////////////////////////////////////
// Other useful tools:
//////////////////////////////////////////////////////////////////////

The 'testwindowsynthesis' binary only supports mono audio files.  We used
Audacity to convert stereo training inputs in to mono with the desired bit
rate.  Audacity may also be used to apply low- and high-pass filters to input
files as appropriate.  Audacity may be downloaded here:
  http://audacity.sourceforge.net/

The 'testwindowsynthesis' binary reads audio signals from a flat vector
binary format (see the linearalgebra/VECTOR* for details).  We use Matlab
scripts to convert between *.wav audio files and *.vector files readable
by 'testwindowsynthesis'.

Use the 'wavread' command in Matlab (or GNU Octave) to read in a wave file.
The provided 'read_vector' and 'write_vector' scripts in the 'matlab/'
directory can be used to write to and read from a format recognizable by
'testwindowsynthesis'.


//////////////////////////////////////////////////////////////////////
// Provided examples:
//////////////////////////////////////////////////////////////////////

We have provided the data necessary to synthesize all examples from the
paper.  In the 'work/' directory you will find subdirectories for each
example.  Navigate to a subdirectory and run './../testwindowsynthesis'.
Each run of the program will generate a subdirectory named
'testwindowsynthesis-xxx'.  In the generated directory, use Matlab
or Octave to generate the final output sound by running the following series
of commands (assuming that the provided read/write_vector scripts are in
your Matlab path):
  p = read_vector( 'output_final_noclip.vector' );
  wavwrite( p, 44100, 'test.wav' );

Some of the original audio clips used for training data can be found
in the 'work' subdirectory.  These clips work taken from the 'Ultimate
Fire' sound library:
  http://www.therecordist.com

Note that in order to use these clips as training inputs to our algorithm
we have downmixed them to 44100Hz mono audio clips, and converted them
in to binary vector format using the provided Matlab scripts.


//////////////////////////////////////////////////////////////////////
// Configuration files:
//////////////////////////////////////////////////////////////////////

Consult the 'default.xml' file in each example directory for instructions
on how to generate configuration files for 'testwindowsynthesis'.  The
following is a list of parameters recognized by the program (see Section 5
of the paper for details):
  Fs: Sampling rate for input and output audio files (44100 in all examples)

  numLevels: Number of Gaussian pyramid levels

  windowHW: Half-width of synthesis windows (h in the paper)

  featureHW: Feature half width (h_f in the paper)

  epsANN: Approximate nearest neighbour epsilon (see final paragraph of
          Section 5.2)

  scaleCDF: Whether or not to use the Dynamic Range Mapping technique
            described in section 5.3 (set to 1 to use the method, 0 otherwise).
