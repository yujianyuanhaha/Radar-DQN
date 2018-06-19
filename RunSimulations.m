% function RadarMDPSim(SimMethod, TrajFormat, TargetTravelMode, NumRuns, ...
%                      NumEvaluations, EvalMethod, EvalTraj, EnableExport, ...
%                      IncludeTrajPlot, NumBands, NumStatesInMemory, InterferenceBehavior, ...
%                      varargin)

% 1. reduce NumRuns
% 2. EvalTraj

RadarMDPSim('Random', {}, 'Cross-Range', 60, ...
    1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}},'DoNotExport',...
    'Separate', 5, 1, 'CONST',...
    [1 0 0 0 0]);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'INTER', [1 0 0 0 0], 0.9);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'INTER', [1 0 0 0 0], 0.1);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-TRIANGLE', [1 0 0 0 0], 'Up');
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 2, 'FH-TRIANGLE', [1 0 0 0 0], 'Up');
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-SAWTOOTH', [1 0 0 0 0], 'Up');
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-PATTERN', [4 16 8 2 1], 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-PATTERN', [4 16 8 2 1 8 4 2 16 1], 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 2, 'FH-PATTERN', [4 16 8 2 1 8 4 2 16 1], 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 2, 'FH-PATTERN', 2.^(randi(5, 1, 1e4)-1), 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 1000, 1, 'EvalOnNew', {{[-2, 4, 0.2],[0.004, 0.0005]}}, 'DoNotExport', 'Separate', 5, 1, 'DIRECTION-DEPENDENT-CONST', [1 0 0 0 0]);
% RadarMDPSim('Random', {}, 'Cross-Range', 1000, 1, 'EvalOnNew', {{[-2, 4, 0.2],[0.004, 0.0005]}}, 'DoNotExport', 'Separate', 5, 1, 'DIRECTION-DEPENDENT-INTER', [1 0 0 0 0], 0.1);
% RadarMDPSim('Random', {}, 'Cross-Range', 1000, 1, 'EvalOnNew', {{[-2, 4, 0.2],[0.004, 0.0005]}}, 'DoNotExport', 'Separate', 5, 1, 'DIRECTION-DEPENDENT-INTER', [1 0 0 0 0], 0.9);
% 
% RadarMDPSim('Random', {}, 'Down-Range', 10000, 1, 'EvalOnNew', {{[-5, 5.5, 0.2],[0.0025, -0.002]}}, 'DoNotExport', 'Separate', 5, 1, 'INTER', [1 0 0 0 0], 0.1);
% 
