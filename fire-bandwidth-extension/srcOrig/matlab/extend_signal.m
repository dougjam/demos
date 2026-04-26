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
% Runs a bandwidth extension procedure on the given input
% signal in order to produce additional high-frequency
% content obeying a power-law: f^(-5/2)
%
% Parameters:
%   p:  The signal to extend
%   Fs: The sampling frequency of the signal
%   Fcutoff: The cutoff frequency used when applying lowpass
%            filters to the signal
%   halfWidth: The halfWidth of the windowing function
%   fitCenter: Point about which we fit a power law spectrum
%              to the existing sound spectrum (in Hz)
%   fitWidth: Approximate width of the smoothing kernel used
%             in the fit described above (in Hz)
%   blendStart, blendEnd: Range over which we blend the two
%                         spectra when recovering the final result
%   noiseScaling: Additional scaling to apply to the noise
%                 added to the signal
function [p_extended, p_lowpass, p_noiseFull, p_filtered, alphas] = ...
                                              extend_signal( ...
                                                  p, Fs, Fcutoff, halfWidth, ...
                                                  fitCenter, fitWidth, ...
                                                  blendStart, blendEnd, ...
                                                  noiseAmplitude, alpha )

  L = size( p, 1 );
  NFFT = 2^nextpow2(L);
  x = Fs * linspace(0, 1, NFFT);

  p_extended = zeros( L, 1 );
  p_filtered = zeros( L, 1 );
  p_noiseFull = zeros( L, 1 );

  alphas = zeros( 0, 1 );

  powerlawExponent = -1.0 * alpha / 2.0;

  % Start by getting a lowpass filtered version of the input signal
  p_lowpass = lowpass_filter( p, Fs, Fcutoff );

  % Windowing setup
  signalCenter = 1;
  signalStart = signalCenter - halfWidth;
  signalEnd = signalCenter + halfWidth;

  windowStart = 2 * L + 1;
  windowEnd = 3 * L;

  numWindows = ceil( double(L) / double(halfWidth) );
  windowNum = 1;

  % Precompute discretizations of the various functions
  % involved in the extension process
  f_blur = build_blurring_function_gaussian( x, fitCenter, fitWidth, NFFT );
  
  [ blend1, blend2 ] = build_blending_function_linear( ...
                                            x, NFFT, blendStart, blendEnd );

  powerlaw = build_powerlaw_spectrum( powerlawExponent, x, NFFT );

  % Scale the power law with the appropriate blending function
  % (otherwise we don't really get any high frequency noise)
  powerlaw = powerlaw .* blend2;

  windowFunction = build_window_function_linear( halfWidth, L );

  noise_unscaled = ifft( powerlaw, NFFT );

  %plot(real(noise_unscaled));
  %pause(4.0);

  % Extend the signal in each window
  while ( signalStart <= L )
    waitbar( double(windowNum) / double(numWindows) );

    startIndex = max( 1, signalStart );
    endIndex = min( L, signalEnd );

    % Window the original and lowpass filtered versions
    % of the signal.
    % The lowpass filtered version is used as an envelope
    % which determines noise amplitude.
    psub = p .* windowFunction( windowStart:windowEnd );

    % Build a noise source whose amplitude matches the
    % amplitude of the windowed, lowpass-filtered signal
    psub_lowpass = zeros( NFFT, 1 );
    psub_lowpass( 1:L ) = abs( p_lowpass ) .* windowFunction( windowStart:windowEnd );
    %psub_lowpass( 1:L ) = p_lowpass .* windowFunction( windowStart:windowEnd );
    %psub_lowpass( 1:L ) = abs( p ) .* windowFunction( windowStart:windowEnd );
    psub_lowpass = psub_lowpass .* noise_unscaled;

    [ psub, p_noiseScaled, p_filteredWindow, alpha ] = ...
                                      extend_sub_signal_noise_source( ...
                                           psub, psub_lowpass, NFFT, x, ...
                                           f_blur, blend1, blend2, ...
                                           noiseAmplitude );

    alphas = [ alphas; alpha ];

    p_extended = p_extended + psub( 1:L );

    p_noiseFull = p_noiseFull + p_noiseScaled( 1:L );
    p_filtered = p_filtered + p_filteredWindow( 1:L );

    signalCenter = signalCenter + halfWidth;
    signalStart = signalCenter - halfWidth;
    signalEnd = signalCenter + halfWidth;

    windowStart = windowStart - halfWidth;
    windowEnd = windowEnd - halfWidth;

    windowNum = windowNum + 1;
  end
end
