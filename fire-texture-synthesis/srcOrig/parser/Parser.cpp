// Parser.cpp: Definition for theeParser class
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

#include "Parser.h"

#include <SETTINGS.h>

#include <IO.h>

#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Constructs a new parser based on the XML document stored in
// filename
//////////////////////////////////////////////////////////////////////
Parser *Parser::buildParser(string filename)
{
	// Construct a new tinyxml document and try to load the
	// given file
	TiXmlDocument *document = new TiXmlDocument();

	if (document->LoadFile(filename.c_str()))
	{
		Parser* parser = new Parser(document);
		parser->filename = filename;
		return parser;
	}
	else
	{
		return NULL;
	}
}

//////////////////////////////////////////////////////////////////////
// Construct a new parser
//////////////////////////////////////////////////////////////////////
Parser::Parser(TiXmlDocument *document)
	: document(document)
{
}

//////////////////////////////////////////////////////////////////////
// Clean up
//////////////////////////////////////////////////////////////////////
Parser::~Parser()
{
	delete document;
}

//////////////////////////////////////////////////////////////////////
// Retrieves parameters for running sound texture synthesis
//////////////////////////////////////////////////////////////////////
Parser::SynthesisParameters Parser::getSynthesisParms()
{
  SynthesisParameters        synthesisParms;

  synthesisParms._baseSignal = queryRequiredAttr( "synthesis", "basesignal" );
  synthesisParms._trainingSignal = queryRequiredAttr( "synthesis",
                                                     "trainingsignal" );
  synthesisParms._trainingSignalLowpass = queryOptionalAttr( "synthesis",
                                                      "trainingsignallowpass",
                                                      "" );

  synthesisParms._Fs = queryOptionalInt( "synthesis", "Fs", "44100" );
  synthesisParms._numLevels = queryOptionalInt( "synthesis", "numLevels", "6" );

  synthesisParms._windowHW = queryRequiredInt( "synthesis", "windowHW" );
  synthesisParms._featureHW = queryRequiredInt( "synthesis", "featureHW" );

  synthesisParms._epsANN = queryOptionalReal( "synthesis", "epsANN", "0.0" );
  synthesisParms._falloff = queryOptionalReal( "synthesis", "falloff", "0.0" );

  synthesisParms._scaleCDF
    = ( queryOptionalInt( "synthesis", "scaleCDF", "0" ) != 0 );

  synthesisParms._scalingAlpha = queryOptionalReal( "synthesis",
                                                    "scalingAlpha", "1.0" );

  return synthesisParms;
}

//////////////////////////////////////////////////////////////////////
// If the attribute DNE, it will print out an error message, prompt
// for input, and return "ERROR" for the value.
//////////////////////////////////////////////////////////////////////
string Parser::queryRequiredAttr( TiXmlElement *elem, std::string attr )
{
	const char* value = elem->Attribute(attr.c_str());

	if( value == NULL )
	{
		cerr << "[ERROR] Required attribute \"" << attr;
		cerr << "\" was not found in the XML elem: " << elem->Value() << endl;
		cerr << "Enter any character to continue, but you \
			should probably abort." << endl;
		char dummy;
		cin >> dummy;
		return string("ERROR");
	}
	else
	{
		return string( value );
	}
}

//////////////////////////////////////////////////////////////////////
// If the attribute DNE, it will return the given defaultValue
//////////////////////////////////////////////////////////////////////
string Parser::queryOptionalAttr( TiXmlElement *node, std::string attr,
																	std::string defaultValue )
{
	const char* value = node->Attribute(attr.c_str());

	if( value == NULL )
	{
		cout << "[STATUS] Optional attribute \"" << attr;
		cerr << "\" not found. Using default: " << defaultValue << endl;
		return defaultValue;
	}
	else
	{
		return string( value );
	}
}

//////////////////////////////////////////////////////////////////////
// Lame implementation of simple xpath. Uses the first child if there are
// multiple elements matching a given name.
//////////////////////////////////////////////////////////////////////
TiXmlNode* Parser::getNodeByPath( std::string pathStr )
{
	vector<string> path = IO::split( pathStr, string("/") );
	TiXmlNode* node = document;
	for( int i = 0; i < path.size(); i++ )
	{
		node = node->FirstChildElement( path[i].c_str() );
		if( node == NULL )
		{
			cerr << "[ERROR] The required XML path did not exist: " << flush << endl;
			for( int j = 0; j < path.size(); j++ )
			{
				cerr << j << " " << path[j] << endl;
				return NULL;
			}
		}
	}

	return node;
}

//////////////////////////////////////////////////////////////////////
// Convenience function. Example:
// return queryRequiredAttr( string("shell/farfieldsound"),
// string("outputDir") );
//////////////////////////////////////////////////////////////////////
string Parser::queryOptionalAttr( std::string path, std::string attr,
																	std::string defaultVal )
{
	TiXmlNode* node = getNodeByPath( path );
	TiXmlElement* elm = node->ToElement();
	string val = queryOptionalAttr( elm, attr, defaultVal );
	cout << "Queried " << attr << "=\"" << val << "\"" << endl;
	return val;
}

//////////////////////////////////////////////////////////////////////
// Convenience function. Example:
// return queryRequiredAttr( string("shell/farfieldsound"),
// string("outputDir") );
//////////////////////////////////////////////////////////////////////
string Parser::queryRequiredAttr( std::string path, std::string attr )
{
	TiXmlNode* node = getNodeByPath( path );
	TiXmlElement* elm = node->ToElement();
	string val = queryRequiredAttr( elm, attr );
	cout << "Queried " << attr << "=\"" << val << "\"" << endl;
	return val;
}

//////////////////////////////////////////////////////////////////////
// Converts a string attribute in to 3 vector
//////////////////////////////////////////////////////////////////////
VEC3F Parser::queryRequiredVEC3( const char *path, const char *attr )
{
	string							 str = queryRequiredAttr( path, attr );
	stringstream				 os( str );
	VEC3F								 result;

	for ( int i = 0; i < 3; i++ )
	{
		if ( !( os >> result[i] ) )
		{
			error_InvalidAttribute( path, attr );
			break;
		}
	}

	return result;
}

//////////////////////////////////////////////////////////////////////
// Converts a string attribute in to 3 tuple
//////////////////////////////////////////////////////////////////////
Tuple3i Parser::queryRequiredTuple3( const char *path, const char *attr )
{
	string							 str = queryRequiredAttr( path, attr );
	stringstream				 os( str );
	Tuple3i							 result;

	for ( int i = 0; i < 3; i++ )
	{
		if ( !( os >> result[i] ) )
		{
			error_InvalidAttribute( path, attr );
			break;
		}
	}

	return result;
}

//////////////////////////////////////////////////////////////////////
// Error reporting
//////////////////////////////////////////////////////////////////////
void Parser::error_MissingNode( const char* path )
{
	cerr << "[ERROR] Required node \"" << path;
	cerr << "\" not found" << endl;
	cerr << "Enter any character to continue, but you \
		should probably abort." << endl;
	char dummy;
	cin >> dummy;
}

//////////////////////////////////////////////////////////////////////
// Error reporting
//////////////////////////////////////////////////////////////////////
void Parser::error_MissingAttribute( const char* path, const char* attrib )
{
	cerr << "[ERROR] Required attribute \"" << attrib;
	cerr << "\" not found in node \"" << path << "\"" << endl;
	cerr << "Enter any character to continue, but you \
		should probably abort." << endl;
	char dummy;
	cin >> dummy;
}

//////////////////////////////////////////////////////////////////////
// Error reporting
//////////////////////////////////////////////////////////////////////
void Parser::error_InvalidAttribute( const char* path, const char* attrib )
{
	cerr << "[ERROR] Required attribute \"" << attrib;
	cerr << "\" in node \"" << path << "\"";
	cerr << " has invalid value " << endl;
	cerr << "Enter any character to continue, but you \
		should probably abort." << endl;
	char dummy;
	cin >> dummy;
}

