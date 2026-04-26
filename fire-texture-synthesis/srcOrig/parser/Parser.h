// Parser.h: Interface for the Parser class
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

#ifndef PARSER_H
#define PARSER_H

#include <tinyxml.h>
#include <tinystr.h>

#include <SETTINGS.h>
#include <TYPES.h>

#include <IO.h>

#include <string>

#include <iostream>

// A helper macro
#define FOR_CHILD_ELEMENTS( childVar, parentVar, childValue ) \
	for( TiXmlElement* childVar = parentVar->FirstChildElement(childValue); \
			childVar; \
			childVar = childVar->NextSiblingElement(childValue) )

//////////////////////////////////////////////////////////////////////
// Parser class
//
// A class which can parse input files and produce simulation meshes
// physical parameters for simulation, etc.
//////////////////////////////////////////////////////////////////////
class Parser {
	public:
    struct SynthesisParameters {
      std::string              _baseSignal;
      std::string              _trainingSignal;
      std::string              _trainingSignalLowpass;

      int                      _Fs;
      int                      _numLevels;

      int                      _windowHW;
      int                      _featureHW;

      Real                     _epsANN;
      Real                     _falloff;

      bool                     _scaleCDF;

      Real                     _scalingAlpha;
    };

		//////////////////////////////////////////////////////////////////////
		static Parser *buildParser(std::string filename);

    // Retrieves parameters for running sound texture synthesis
    SynthesisParameters getSynthesisParms();

		// If the attribute DNE, it will print out an error message, prompt
		// for input, and return "ERROR" for the value.
		static std::string queryRequiredAttr( TiXmlElement* node,
																					std::string attr );

		// If the attribute DNE, it will return the given defaultValue
		static std::string queryOptionalAttr( TiXmlElement* node,
																					std::string attr,
																					std::string defaultValue );

		// Lame implementation of simple xpath. Uses the first child if there are
		// multiple elements matching a given name.
		TiXmlNode* getNodeByPath( std::string pathStr );

		// Convenience function. Example:
		// return queryRequiredAttr( string("shell/farfieldsound"),
		// string("outputDir") );
		std::string queryRequiredAttr( std::string path, std::string attr );
		std::string queryOptionalAttr( std::string path, std::string attr,
																	 std::string defaultVal );

		std::string queryOptionalAttr( const char* path, const char* attr,
																	 const char* defaultVal )
		{
			return queryOptionalAttr( std::string(path), std::string(attr),
																std::string(defaultVal) );
		}

		std::string queryRequiredAttr( const char* path, const char* attr )
		{
			return queryRequiredAttr( std::string(path), std::string(attr) );
		}

		Real queryOptionalReal( const char* path, const char* attr,
														const char* defaultVal )
		{
			return atof( queryOptionalAttr( path, attr, defaultVal ).c_str() );
		}

		Real queryRequiredReal( const char* path, const char* attr )
		{
			return atof( queryRequiredAttr( path, attr ).c_str() );
		}

		int queryOptionalInt( const char* path, const char* attr,
													const char* defaultVal )
		{
			return atoi( queryOptionalAttr( path, attr, defaultVal ).c_str() );
		}

		int queryRequiredInt( const char* path, const char* attr )
		{
			return atoi( queryRequiredAttr( path, attr ).c_str() );
		}

		// Converts a string attribute in to 3 vector
		VEC3F queryRequiredVEC3( const char *path, const char *attr );
    Tuple3i queryRequiredTuple3( const char *path, const char *attr );

		// Error reporting
		void error_MissingNode( const char* path );
		void error_MissingAttribute( const char* path, const char* attrib );
		void error_InvalidAttribute( const char* path, const char* attrib );

		// If you want to walk the DOM on your own, use this
		TiXmlDocument* getDocument() { return document; }

		virtual ~Parser();

	private:
		// Construct a new parser given a tinyxml document.  We do
		// this privately so that we can make sure a file exists
		// before constructing the parser
		Parser(TiXmlDocument *document);

  private:
		TiXmlDocument *document;

		// Kept mainly for helpful error reporting
		std::string filename;
};

#endif
