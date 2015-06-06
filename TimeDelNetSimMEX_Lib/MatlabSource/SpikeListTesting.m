%%
rng(25);
N = 1000;
E = 0.8;
RecurrentNetParams.NExc = round(N*E);
RecurrentNetParams.NInh = round(N - N*E);

RecurrentNetParams.NSynExctoExc = ceil(100*N/2000);
RecurrentNetParams.NSynExctoInh = ceil(100*N/2000);
RecurrentNetParams.NSynInhtoExc = ceil(1200*N/2000);

RecurrentNetParams.MeanExctoExc = 0.5*2000/N;
RecurrentNetParams.MeanExctoInh = 0.15*2000/N;
RecurrentNetParams.MeanInhtoExc = -0.7*2000/N;

RecurrentNetParams.Var          = 0.2;
RecurrentNetParams.DelayRange   = 20;

[A, Ninh, Weights, Delays] = RecurrentNetwork(RecurrentNetParams);

a = 0.02*ones(N,1);
b = 0.2*ones(N,1);
c = -65*ones(N,1);
d = 8*ones(N,1);

a(Ninh) = 0.1;
b(Ninh) = 0.2;
c(Ninh) = -65;
d(Ninh) = 2;
% Delays = Delays + 10;
[NEndVect, NStartVect] = find(A);

%% Input setup
% Setting up input settings
OutputOptions = { ...
	'V', ...
	'Iin', ...
	'Itot', ...
	'Irand', ...
	'Initial'
	};
% OutputOptions = {'FSF', 'Initial'};
% OutputOptions = {'SpikeList', 'Final', 'Initial'};
% Clearing InputStruct
clear InputStruct;

% Getting Midway state
InputStruct = ConvertStatetoInitialCond(StateVars, 6000*4);
InputStruct.a = single(a);
InputStruct.b = single(b);
InputStruct.c = single(c);
InputStruct.d = single(d);

InputStruct.NStart = int32(NStartVect);
InputStruct.NEnd   = int32(NEndVect);
InputStruct.Weight = single(Weights);
InputStruct.Delay  = single(Delays);

InputStruct.onemsbyTstep          = int32(4);
InputStruct.NoOfms                = int32(3000);
InputStruct.DelayRange            = int32(RecurrentNetParams.DelayRange);
InputStruct.StorageStepSize       = int32(0);
InputStruct.OutputControl         = strjoin(OutputOptions);
InputStruct.StatusDisplayInterval = int32(8000);

% tic;
% try 
% 	[OutputVar2, StateVars2, FinalState2, InitState2] = TimeDelNetSimMEX_Lib(InputStruct);
% catch e
% 	clear functions;
% 	throw(e);
% end
% toc;
InputStruct.OutputFile = 'SimResults1000DebugDetailLong.mat';
save('TimeDelNetSimMEX/TimeDelNetSimMEX_Exe/Data/InputData.mat', 'InputStruct');
clear functions;

%% Data Load
load('TimeDelNetSimMEX/TimeDelNetSimMEX_Exe/Data/SimResults1000DebugSparseLong.mat');

%% Convert Spike to spatio(neuro)-temporal Data.

BegTime = double((6*60 + 30 + 15)*1000*InputStruct.onemsbyTstep);
EndTime = double((6*60 + 30 + 18)*1000*InputStruct.onemsbyTstep);

RelTimes = StateVarsDetailed.Time >= BegTime & StateVarsDetailed.Time < EndTime;
BegTimeIndex = find(RelTimes, 1, 'first');
EndTimeIndex = find(RelTimes, 1, 'last') + InputStruct.onemsbyTstep*InputStruct.DelayRange - 1;

% Calculating Total number of spikes
SpikeSynInds = OutputVarsDetailed.SpikeList.SpikeSynInds;
TimeRchdStartInds = OutputVarsDetailed.SpikeList.TimeRchdStartInds;
TotalLength = double(TimeRchdStartInds(EndTimeIndex + 1) - TimeRchdStartInds(BegTimeIndex));

% Calculating the vector of time instants corresponding to arrival times
% (minus 1)

TimeVect = zeros(TotalLength, 1);

InsertIndex = 1;
Time = StateVarsDetailed.Time(BegTimeIndex);
for i = BegTimeIndex:EndTimeIndex
	NumofElemsCurrTime = double(TimeRchdStartInds(i+1) - TimeRchdStartInds(i));
	TimeVect(InsertIndex:InsertIndex + NumofElemsCurrTime - 1) = Time;
	Time = Time+1;
	InsertIndex = InsertIndex + NumofElemsCurrTime;
end

% Straightening out the Cell Array.
SpikeListVect = SpikeSynInds(TimeRchdStartInds(BegTimeIndex)+1:TimeRchdStartInds(EndTimeIndex+1)) + 1; % +1 for the C++ to matlab 
                                         % indexing convention conversion

% Calculating Synapse parameter vectors
SpikePreSynNeuronVect = InputStruct.NStart(SpikeListVect);
SpikeDelayVect        = round(double(InputStruct.onemsbyTstep)*InputStruct.Delay(SpikeListVect));

% Adjusting TimeVect for Delays
TimeVect = TimeVect - SpikeDelayVect + 1;

% Removing all entries that do not come into the relevant time frame
SpikeListVect         = SpikeListVect(TimeVect >= BegTime & TimeVect <= EndTime);
SpikePreSynNeuronVect = SpikePreSynNeuronVect(TimeVect >= BegTime & TimeVect <= EndTime);
SpikeDelayVect        = SpikeDelayVect(TimeVect >= BegTime & TimeVect <= EndTime);
TimeVect              = TimeVect(TimeVect >= BegTime & TimeVect <= EndTime);

% Creating Sparse Matrix
TimeRange = EndTimeIndex - BegTimeIndex + 1;
SpikeMat = sparse(double(SpikePreSynNeuronVect), double(TimeVect) - BegTime + 1, ones(length(TimeVect), 1), N, double(TimeRange));
figure;
spy(SpikeMat, 1);

%% Random Plotting

