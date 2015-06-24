function [ FigureOut ] = DisplayPNG(RelativePNGIn, nNeurons)
%DISPLAYPNG Summary of this function goes here
%   Detailed explanation goes here

VertexSet = [RelativePNGIn.SpikeTimings RelativePNGIn.SpikeNeurons];

RandYShifts = 0.4*rand(nNeurons, 1) - 0.2;

Jittered_Coords = VertexSet + [zeros(size(VertexSet(:,1))) RandYShifts(VertexSet(:,2))];

NInitialNeurons = nnz(RelativePNGIn.IndexVector == 0) - 1;

VertexIndsforSynapses = zeros(size(RelativePNGIn.SpikeSynapses));
VertexIndsforSynapses(RelativePNGIn.IndexVector(1:end-1) + 1) = 1;
VertexIndsforSynapses = cumsum(VertexIndsforSynapses);
VertexIndsforSynapses = VertexIndsforSynapses + NInitialNeurons;

NAdjMat = size(RelativePNGIn.SpikeTimings, 1);

AdjMat = sparse(VertexIndsforSynapses, RelativePNGIn.SpikeSynapses, ...
	            ones(size(RelativePNGIn.SpikeSynapses)), NAdjMat, NAdjMat);

FigureOut = figure;
gplot(AdjMat, Jittered_Coords);
hold on;
plot(Jittered_Coords(:,1), Jittered_Coords(:,2), 'o','markersize', 5);
for i = 1:size(VertexSet,1)
	text(Jittered_Coords(i, 1) + 0.2, Jittered_Coords(i,2),num2str(VertexSet(i,2)));
end

hold off;

end