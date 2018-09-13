% function RadarMDPSim(SimMethod,        TrajFormat,   TargetTravelMode,   NumRuns, ...
%                      NumEvaluations,   EvalMethod,   EvalTraj,          EnableExport, ...
%                      IncludeTrajPlot,  NumBands,     NumStatesInMemory, InterferenceBehavior, ...
%                      varargin)

% 1. reduce NumRuns
% 2. EvalTraj

% ==== setup to blend matlab and tensorflow
py.sys.setdlopenflags(int32(10))
py.importlib.import_module('dqn')
py.importlib.import_module('dpg')
py.importlib.import_module('ac')
py.importlib.import_module('actor')
py.importlib.import_module('critic')
py.importlib.import_module('dqnDouble')
py.importlib.import_module('dqnDuel')
py.importlib.import_module('dqnPriReplay')
% tic;

% solver option
% 1. mdp
% 2. dqn - Deep Q Network
% 3. dpg - Deep Policy Gradient
% 4. ac  - Action Critic  (todo)
% 5. dqnDouble 
% 6. dqnDuel
% 7. dqnPriReplay (todo)

solver = "dqnDuel";          % <<<<<<<<<<<<<<<<<<<<<<<
fprintf('solver is %s \n',solver);
RadarMDPSim('Random',   {},           'Cross-Range',                    60000, ...
             1,         'EvalOnNew',  {{[-4, 3.8, 0.2],[0.005, 0]}},    'DoNotExport',...
            'Separate',  5,           1,                                'CONST',...
            solver,       [1 0 0 0 0]);
% 
% RadarMDPSim('Random', {}, 'Down-Range', 10000, 1, 'EvalOnNew', {{[-5, 5.5, 0.2],[0.0025, -0.002]}}, 'DoNotExport', 'Separate', 5, 1, 'INTER',solver, [1 0 0 0 0], 0.1);
% 
% RadarMDPSim(   'Random',          {},                  'Cross-Range',        60000, ...
%                       1, 'EvalOnNew',  {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', ...
%              'Separate',           5,                               1,       'INTER', ...
%                solver,    [1 0 0 0 0],         0.9);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'INTER', solver,[1 0 0 0 0], 0.1);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-TRIANGLE', solver,[1 0 0 0 0], 'Up');
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 2, 'FH-TRIANGLE', solver,[1 0 0 0 0], 'Up');
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-SAWTOOTH', solver,[1 0 0 0 0], 'Up');
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-PATTERN',solver, [4 16 8 2 1], 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 1, 'FH-PATTERN', solver,[4 16 8 2 1 8 4 2 16 1], 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 2, 'FH-PATTERN',solver,[4 16 8 2 1 8 4 2 16 1], 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 60000, 1, 'EvalOnNew', {{[-4, 3.8, 0.2],[0.005, 0]}}, 'DoNotExport', 'Separate', 5, 2, 'FH-PATTERN', solver,2.^(randi(5, 1, 1e4)-1), 0);
% RadarMDPSim('Random', {}, 'Cross-Range', 1000, 1, 'EvalOnNew', {{[-2, 4, 0.2],[0.004, 0.0005]}}, 'DoNotExport', 'Separate', 5, 1, 'DIRECTION-DEPENDENT-CONST', solver,[1 0 0 0 0]);
% RadarMDPSim('Random', {}, 'Cross-Range', 1000, 1, 'EvalOnNew', {{[-2, 4, 0.2],[0.004, 0.0005]}}, 'DoNotExport', 'Separate', 5, 1, 'DIRECTION-DEPENDENT-INTER',solver, [1 0 0 0 0], 0.1);
% RadarMDPSim('Random', {}, 'Cross-Range', 1000, 1, 'EvalOnNew', {{[-2, 4, 0.2],[0.004, 0.0005]}}, 'DoNotExport', 'Separate', 5, 1, 'DIRECTION-DEPENDENT-INTER', solver,[1 0 0 0 0], 0.9);
% % 

