% Copyright (c) 2011, Jeffrey Chadwick
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% - Redistributions of source code must retain the above copyright notice,
%   this list of conditions and the following disclaimer.
%
% - Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Builds linear blending functions for putting two
% frequency spectra together
function [b1, b2] = build_blending_function_linear( x, NFFT, f1, f2 )

  % Define windowing functions for the original and
  % extended spectra.
  function w = spectrumWeight( f )
    if ( f <= f1 )
      w = 1.0;
    elseif ( f >= f2 )
      w = 0.0;
    else
      f = ( f - f1 ) / ( f2 - f1 );
      w = 1.0 - f;
    end
  end

  function w = noiseWeight( f )
    if ( f <= f1 )
      w = 0.0;
    elseif ( f >= f2 )
      w = 1.0;
    else
      f = ( f - f1 ) / ( f2 - f1 );

      w = f;
    end
  end

  w1 = @spectrumWeight;
  w2 = @noiseWeight;

  b1 = build_blurring_function( x, w1, NFFT );
  b2 = build_blurring_function( x, w2, NFFT );
end
