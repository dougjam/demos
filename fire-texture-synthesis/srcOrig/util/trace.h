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

#ifndef _TRACE_H_
#define _TRACE_H_

#include <cstdlib>
#include <stdio.h>
#ifdef WIN32
#include <conio.h>
#endif

//----------------------------------------
#define TRACE(priority,...) { \
	if (priority < 2) \
	{ \
		fprintf (stderr, "[ %s,%s,%d ] ", __FILE__, __FUNCTION__, __LINE__); \
		fprintf (stderr, __VA_ARGS__); \
		fprintf (stderr, "\n"); \
	} \
}

//----------------------------------------
#define TRACE_ERROR(...) { \
	fprintf (stderr, "[ ERROR @ %s,%s,%d ] ", __FILE__, __FUNCTION__, __LINE__); \
	fprintf (stderr, __VA_ARGS__); \
	fprintf (stderr, "\n"); \
	fprintf (stderr, "Quit? (1/0) >> "); \
	int _trace_error_quit, _trace_error_result; \
	_trace_error_result = scanf( "%d", &_trace_error_quit ); \
	if( _trace_error_quit != 0 ) exit (1); \
}

//----------------------------------------
#define TRACE_WARNING(...) { \
	fprintf (stderr, "[ WARNING ] %s: ", __FUNCTION__); \
	fprintf (stderr, __VA_ARGS__); \
	fprintf (stderr, "\n"); \
}

//----------------------------------------
#define TRACE_ASSERT(pred, ...)	\
	if( !(pred) )	\
	{	\
		TRACE_ERROR( "( " #pred " ) " __VA_ARGS__ );	\
	}

#endif

