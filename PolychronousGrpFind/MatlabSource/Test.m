%% Mex Function Include Cell
addpath('..\..\x64\Release_Lib\');

%% Network Setup and Simulation
N = 5;

A = [ 0    2    4    10   5   ;
      6    0    2    3    4   ;
      4    7    0    5    4   ;
      2    7    2    0    4   ;
      2    2    7    6    0   ];

[NEndVect, NStartVect, Delays] = find(A);
Weights = 10*ones(size(Delays));

clear InputStruct;

a = 0.02*ones(N,1);
b = 0.2*ones(N,1);
c = -65*ones(N,1);
d = 8*ones(N,1);

InputStruct.a = single(a);
InputStruct.b = single(b);
InputStruct.c = single(c);
InputStruct.d = single(d);

InputStruct.NStart = int32(NStartVect);
InputStruct.NEnd   = int32(NEndVect);
InputStruct.Weight = single(Weights);
InputStruct.Delay  = single(Delays);

InputStruct.onemsbyTstep  = int32(1);
InputStruct.DelayRange = int32(10);

InputStruct.OutputFile = 'FiveNeuronNetPNGs.mat';
save('../Data/InputData.mat', 'InputStruct');

% OutputVars = PolychronousGrpFind(InputStruct);
% clear functions;
%% Read Data;

load('../Data/FiveNeuronNetPNGs.mat', 'OutputVars');

%% Process Data

ChosenPNGIndex = 18;
ChosenPNG = GetPNG(OutputVars, ChosenPNGIndex);
ChosenRelativePNG = ConvertPNGtoRelative(ChosenPNG, InputStruct.NStart, InputStruct.Delay);

DisplayPNG(ChosenRelativePNG, length(InputStruct.a));