% This file requires that the returned state is retured.

clear InputStruct;

% Getting InputStruct Initialized by returned state
InputStruct = ConvertStatetoInitialCond(StateVarsSparse, (60*8)*4000);

InputStruct.a = single(a);
InputStruct.b = single(b);
InputStruct.c = single(c);
InputStruct.d = single(d);

InputStruct.NStart = int32(NStartVect);
InputStruct.NEnd   = int32(NEndVect);
% Weight Initialization done using retured state
InputStruct.Delay  = single(Delays);

InputStruct.onemsbyTstep          = int32(1);
InputStruct.DelayRange            = int32(RecurrentNetParams.DelayRange);
InputStruct.OutputFile            = 'PNGsin1000NeuronsWOProhib.mat';

save('..\..\PolychronousGrpFind\Data\InputData.mat', 'InputStruct');