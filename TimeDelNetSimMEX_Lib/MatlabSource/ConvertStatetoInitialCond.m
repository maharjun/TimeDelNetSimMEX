function [ InputStruct ] = ConvertStatetoInitialCond( StateStruct, timeInstant )
% CONVERTSTATETOINITIALCOND Converts structs
%   basically a name conversion Function as such. 

timeDimensionLength = length(StateStruct.Time);
timeIndex = 1;
if timeDimensionLength == 1
	timeIndex = 1;
elseif nargin() == 2 
	timeIndex = find(StateStruct.Time == timeInstant, 1);
	if isempty(timeIndex)
		Ex = MException('NeuralSim:ConvertStatetoInitialCond:InvalidTimeInstant', ...
					'Need to specify a time instant belonging to from StateStruct.Time');
		throw(Ex);
	end

end

InputStruct.V = StateStruct.V(:, timeIndex);
InputStruct.U = StateStruct.U(:, timeIndex);
InputStruct.Iin = StateStruct.Iin(:, timeIndex);
InputStruct.WeightDeriv = StateStruct.WeightDeriv(:, timeIndex);
InputStruct.Weight = StateStruct.Weight(:, timeIndex);

InputStruct.Iext = StateStruct.Iext(:, timeIndex);
InputStruct.IExtGenState = StateStruct.IExtGenState(:, timeIndex);
InputStruct.Time = StateStruct.Time(timeIndex);

InputStruct.CurrentQIndex = StateStruct.CurrentQIndex(timeIndex);
if timeDimensionLength == 1
	InputStruct.SpikeQueue = StateStruct.SpikeQueue;
else
	InputStruct.SpikeQueue = StateStruct.SpikeQueue{timeIndex};
end

InputStruct.LSTNeuron = StateStruct.LSTNeuron(:, timeIndex);
InputStruct.LSTSyn = StateStruct.LSTSyn(:, timeIndex);

end

