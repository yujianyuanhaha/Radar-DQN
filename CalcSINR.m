function SINRdB = CalcSINR(Range, NumBands, CurrentAction, CurrentInt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Transmitter parameters
Ptnew = 100;                     % Transmit power (Watts)
G = 10;                            % Antenna gain (unitless)
lambda = 3*10^8/(2*10^9);           % Wavelength (meters)
sigma = 0.1;                        % Target's radar cross section (square-meters)
Np = 50;                            % Number of coherently integrated pulses (unitless)
BandSize = 20e6;                     % Sub-band width (Hertz)
TBnew = 1e4;                        % Time-Bandwidth product (unitless)

% Receiver parameters
NF = 1;                             % Noise figure (unitless)
Boltzmann = 1.38064852*10^(-23);    % Boltzmann's constant (unitless)
Ts = 295;                           % System temperature (Kelvin)

% Set interference parameters
IntPower = 1*10^(-11);              % Interference power (Watts)

I = sum(CurrentAction.*CurrentInt)*IntPower;
N = Boltzmann*Ts*NF*sum(CurrentAction)*BandSize;
Prnew = Ptnew*G*G*lambda^2*sigma*TBnew*(sum(CurrentAction)/NumBands)*Np/(4*pi)^3/(Range*1000)^4;
SINR = Prnew/(I+N);
SINRdB = 10*log10(SINR);
end

