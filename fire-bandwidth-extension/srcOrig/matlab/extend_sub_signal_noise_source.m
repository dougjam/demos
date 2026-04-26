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
% Takes an input signal and an input noise source, then
% chooses an appropriate scaling for the noise source
% so that the spectra of the two signals roughly line
% up around a given fit point
%
% Parameters:
%   psub: The original signal to extend
%   pnoise: The noise signal we want to add in
%   NFFT: the the size of the Fourier transform to take.
%   x: the frequency space to which the FFT should correspond.
%   f_blur: the blurring function for fitting two spectra
%   blend1, blend2: the blending functions for summing two spectra
%   noiseAmplitude: Additional scaling to apply to the noise
function [ psub, p_noiseScaled, p_filtered, alpha ] = ...
                           extend_sub_signal_envelope( ...
                                  psub, pnoise, NFFT, x, ...
                                  f_blur, blend1, blend2, ...
                                  noiseAmplitude )
  
  % Get spectra for the original signal and the windowed
  % noise signal
  Y = fft( psub, NFFT );
  Ynoise = fft( pnoise, NFFT );

  L = size( psub, 1 );

  % Apply windowing functions to the two spectra
  spectrum_signal = get_windowed_spectra( Y, blend1 );
  spectrum_noise = get_windowed_spectra( Ynoise, blend2 );

  % Figure out the correct scaling factor to apply to the
  % noise spectram
  alpha = fit_dual_power_spectra( x, spectrum_signal, spectrum_noise, Y, f_blur );

  %[ 'Got alpha ' num2str(alpha) ]

  % Construct the blended spectrum and convert back to the time domain
  p_extended = ifft( spectrum_signal + alpha * noiseAmplitude * spectrum_noise, NFFT );

  % Get other stuff for debugging purposes:
  %
  % Just the scaled noise signal
  p_noiseScaled = ifft( alpha * noiseAmplitude * spectrum_noise, NFFT );

  p_filtered = ifft( spectrum_signal, NFFT );
  p_filtered = p_filtered(1:L);

  % The final sub signal we want to add to the total signal
  psub = real( p_extended(1:L) );
end
