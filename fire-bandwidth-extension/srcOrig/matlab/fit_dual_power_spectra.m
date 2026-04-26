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
% Fits power spectra of two signals scaled by a kernel function
% The first signal is given by ( Y_L + alpha * Y_N ) and the
% second by Y_S.  The output is a choice of alpha such that the
% integrated powers of these signals, multiplied by the kernel
% f are equal.
%
% Parameters:
%   x: The frequency of each sample
%   Y_L, Y_N, Y_S: As explained above.
%   f: Smoothing function to apply.  This is assumed to have
%      been discretized to have as many entries as x
function alpha = fit_dual_power_spectra( x, Y_L, Y_N, Y_S, f )
  alpha = 0.0;

  L1 = size( Y_L, 1 );
  L2 = size( Y_N, 1 );
  L3 = size( Y_S, 1 );
  L4 = size( f, 1 );

  % Make x a row vector if it isn't already
  if ( size( x, 2 ) ~= 1 )
    x = x';
  end

  if ( L1 ~= L2 || L1 ~= L3 || L1 ~= L4 )
    [ 'L1 = ' num2str( L1 ) '   L2 = ' num2str( L2 ) '    L3 = ' num2str( L3 ) ]
    error( 'Size mismatch' );
  end

  L = L1;

  X1 = x( 1:L-1 );
  X2 = x( 2:L );

  f1 = f( 1:L-1 );
  f2 = f( 2:L );

  % Get the appropriate sub signals for the real part, complex part
  % and squared magnitude of the input signals
  Y_L_r_1 = real( Y_L( 1:L-1 ) );
  Y_L_r_2 = real( Y_L( 2:L ) );
  Y_L_i_1 = imag( Y_L( 1:L-1 ) );
  Y_L_i_2 = imag( Y_L( 2:L ) );

  Y_N_r_1 = real( Y_N( 1:L-1 ) );
  Y_N_r_2 = real( Y_N( 2:L ) );
  Y_N_i_1 = imag( Y_N( 1:L-1 ) );
  Y_N_i_2 = imag( Y_N( 2:L ) );

  Y_S_s_1 = Y_S( 1:L-1 ) .* conj( Y_S( 1:L-1 ) );
  Y_S_s_2 = Y_S( 2:L ) .* conj( Y_S( 2:L ) );

  % Integrands for the 7 integrals we need to take
  z1_1 = Y_L_r_1 .* Y_L_r_1 .* f1;
  z1_2 = Y_L_r_2 .* Y_L_r_2 .* f2;

  z2_1 = Y_N_r_1 .* Y_N_r_1 .* f1;
  z2_2 = Y_N_r_2 .* Y_N_r_2 .* f2;

  z3_1 = 2.0 * ( Y_L_r_1 .* Y_N_r_1 .* f1 );
  z3_2 = 2.0 * ( Y_L_r_2 .* Y_N_r_2 .* f2 );

  z4_1 = Y_L_i_1 .* Y_L_i_1 .* f1;
  z4_2 = Y_L_i_2 .* Y_L_i_2 .* f2;

  z5_1 = Y_N_i_1 .* Y_N_i_1 .* f1;
  z5_2 = Y_N_i_2 .* Y_N_i_2 .* f2;

  z6_1 = 2.0 * ( Y_L_i_1 .* Y_N_i_1 .* f1 );
  z6_2 = 2.0 * ( Y_L_i_2 .* Y_N_i_2 .* f2 );

  zs_1 = Y_S_s_1 .* f1;
  zs_2 = Y_S_s_2 .* f2;

  I1 = 0.5 * ( X2' * z1_1 - X1' * z1_2 + X2' * z1_2 - X1' * z1_1 );
  I2 = 0.5 * ( X2' * z2_1 - X1' * z2_2 + X2' * z2_2 - X1' * z2_1 );
  I3 = 0.5 * ( X2' * z3_1 - X1' * z3_2 + X2' * z3_2 - X1' * z3_1 );
  I4 = 0.5 * ( X2' * z4_1 - X1' * z4_2 + X2' * z4_2 - X1' * z4_1 );
  I5 = 0.5 * ( X2' * z5_1 - X1' * z5_2 + X2' * z5_2 - X1' * z5_1 );
  I6 = 0.5 * ( X2' * z6_1 - X1' * z6_2 + X2' * z6_2 - X1' * z6_1 );

  Is = 0.5 * ( X2' * zs_1 - X1' * zs_2 + X2' * zs_2 - X1' * zs_1 );

  % Quadratic equation coefficients
  a = I2 + I5;
  b = I3 + I6;
  c = I1 + I4 - Is;

  % Solve the quadratic
  disc = b^2 - 4 * a * c;

  if ( disc < 0 )
    error( 'Negative discriminant: complex roots found!' );
  end

  alpha1 = 0;
  alpha2 = 0;

  if ( b > 0 )
    alpha1 = ( -b - sqrt( disc ) ) / ( 2.0 * a );
    alpha2 = c / ( alpha1 * a );
  else
    alpha1 = ( -b + sqrt( disc ) ) / ( 2.0 * a );
    alpha2 = c / ( alpha1 * a );
  end

  if ( alpha1 > 0 && alpha2 > 0 )
    error( 'Two positive alphas found' );
  elseif ( alpha1 < 0 && alpha2 < 0 )
    error( 'Two negative alphas found' );
  else
    alpha = max( alpha1, alpha2 );
  end

  %spectrum1_integral = 0.5 * ( X2' * Y1_1 - X1' * Y1_2 + X2' * Y1_2 - X1' * Y1_1 );

  %spectrum2_integral = 0.5 * ( X2' * Y2_1 - X1' * Y2_2 + X2' * Y2_2 - X1' * Y2_1 );

  %[ 'Got spectrum1 integral' num2str(spectrum1_integral) ]
  %[ 'Got spectrum2 integral' num2str(spectrum2_integral) ]

  %alpha = spectrum1_integral / spectrum2_integral;
end
