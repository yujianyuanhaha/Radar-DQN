function RadarMDPSim(   SimMethod,       TrajFormat, TargetTravelMode,  NumRuns, ...
    NumEvaluations,  EvalMethod, EvalTraj,          EnableExport, ...
    IncludeTrajPlot, NumBands,   NumStatesInMemory, InterferenceBehavior, ...
    solver, varargin)

tic;


% Potentially add number of bands, as a parameter so that is adjustable
% Potentially add InitialState as a parameter to TRIANGLE, SAWTOOTH, and BURSTY
% Potentially add NumRepeats as a parameter to PATTERN and PSEUDO
% Initialize intermittent using rand and if statement

% Add toolboxes to MATLAB's search path
addpath(genpath('./mdp-toolbox'));      % MDP-Toolbox
addpath(genpath('./line2arrow-pkg'));   % Line2Arrow
addpath(genpath('./subaxis'));          % Subaxis
addpath(genpath('./export-fig'));       % Export-Fig

% Shuffle random number generator
rng('shuffle');

% Set figure text interpreter to Tex, and set default font to LM Roman 10
set(0, 'DefaultTextInterpreter', 'Tex');
set(0, 'DefaultLegendInterpreter', 'Tex');
set(0, 'DefaultAxesTickLabelInterpreter', 'Tex');
set(0, 'DefaultTextFontname', 'L M Roman10');
set(0, 'DefaultAxesFontname', 'L M Roman10');

% Set miscellaneous parameters
RadarBehavior  = 'OPTI';         % 'CONS', 'OPTI','RAND' % constant action, optimal (ML), random
DiscountFactor = 0.9;

% Set simulation step time, number of time steps, and time vector
TimeSteps    = 150/1.0;                                     % seconds
TimeInterval = 10;                                          % seconds
t            = 0: TimeInterval: (TimeSteps) * TimeInterval; % J - t -> s

% Transmitter parameters
Ptnew    = 100;                           % Transmit power (Watts)
G        = 10;                            % Antenna gain (unitless)
lambda   = 3*10^8/(2*10^9);               % Wavelength (meters)
sigma    = 0.1;                           % Target's radar cross section (square-meters)
Np       = 50;                            % Number of coherently integrated pulses (unitless)
BandSize = 20e6;                          % Sub-band width (Hertz)
TBnew    = 1e4;                           % Time-Bandwidth product (unitless)

% Receiver parameters
NF        = 1;                            % Noise figure (unitless)
Boltzmann = 1.38064852*10^(-23);          % Boltzmann's constant (unitless)
Ts        = 295;                          % System temperature (Kelvin)

% Set interference parameters
IntPower = 1*10^(-11);              % Interference power (Watts)

% Initialize target positions in horizontal plane, target height, target
% positions in 3-dimensional space, target velocities, SINRs, interference
% states, and actions
TargetPositionsXY = [...
    -4, 5.0; -3, 5.0; -2, 5.0; -1, 5.0; 0, 5.0; 1, 5.0; 2, 5.0; 3, 5.0; 4, 5.0; ...
    -4, 4.5; -3, 4.5; -2, 4.5; -1, 4.5; 0, 4.5; 1, 4.5; 2, 4.5; 3, 4.5; 4, 4.5; ...
    -4, 4.0; -3, 4.0; -2, 4.0; -1, 4.0; 0, 4.0; 1, 4.0; 2, 4.0; 3, 4.0; 4, 4.0; ...
    -3, 3.5; -2, 3.5; -1, 3.5; 0, 3.5; 1, 3.5; 2, 3.5; 3, 3.5; ...
    -3, 3.0; -2, 3.0; -1, 3.0; 0, 3.0; 1, 3.0; 2, 3.0; 3, 3.0; ...
    -3, 2.5; -2, 2.5; -1, 2.5; 0, 2.5; 1, 2.5; 2, 2.5; 3, 2.5; ...
    -1, 2.0; 0, 2.0; 1, 2.0; ...
    -1, 1.5; 0, 1.5; 1, 1.5; ...
    -1, 1.0; 0, 1.0; 1, 1.0 ...
    ];

TargetHeight    = 200; % m
TargetPositions = [TargetPositionsXY, (TargetHeight/1000)*ones(size(TargetPositionsXY,1),1)]; % km

switch TargetTravelMode
    case 'Cross-Range'
        TargetVelocities = 1.0*[0.005, 0; 0.005, 0.0002; 0.005, -0.0002;  ...
            -0.005, 0.0002; -0.005, -0.0002; -0.005, 0]; % km/s
    case 'Down-Range'
        TargetVelocities = 1.0*[0, 0.005; 0.0002, 0.005; -0.0002, 0.005;  ...
            0.0002, -0.005; -0.0002, -0.005; 0, -0.005]; % km/s
end

SINRs   = [11]';
Actions = SelectOnlyContiguousBands(de2bi([1:2^(NumBands)-1], NumBands, 'left-msb'));

interference_cellsXY = [2, 5.0; 2, 4.5; 2, 4.0; 2, 3.5; 2, 3.0; 2, 2.5; ...
    3, 5.0; 3, 4.5; 3, 4.0; 3, 3.5; 3, 3.0; 3, 2.5];
interference_cells   = [interference_cellsXY, (TargetHeight/1000)*ones(size(interference_cellsXY,1),1)];
states_with_interference = zeros(size(interference_cells,1),1);

for stateIndex = 1:size(interference_cells,1)
    tmp = sum(abs(TargetPositions - repmat(interference_cells(stateIndex,:), size(TargetPositions, 1) ,1))');
    [~, min_indx_intf_pos_cells] = min(tmp);
    states_with_interference(stateIndex) = min_indx_intf_pos_cells;
end

% Initialize interferer
switch InterferenceBehavior
    case 'CONST' % Constant interferer
        CurrentInt = varargin{1};
    case 'INTER' % Intermittent interferer
        CurrentInt = varargin{1};
        IntermProb = varargin{2};
    case 'AVOID' % Avoiding interferer
        CurrentInt = varargin{1};
    case 'FH-TRIANGLE' % Triangle frequency sweep interferer
        CurrentInt     = varargin{1};
        NextSweepState = varargin{2};
    case 'FH-SAWTOOTH' % Sawtooth frequency sweep interferer
        CurrentInt     = varargin{1};
        NextSweepState = varargin{2};
    case 'FH-PATTERN' % Frequency hopping pattern interferer
        fhPattern        = varargin{1};
        NumIterations    = varargin{2};
        NextPatternIndex = 1;
        CurrentInt       = de2bi(fhPattern(1), NumBands, 'left-msb');
    case 'FH-PSEUDO' % Pseudorandom frequency hopping interferer
        PseudoLength   = varargin{1};
        NumIterations  = varargin{2};
        pseudosequence = randi([1 (2^(NumBands)-1)], 1, PseudoLength);
        if NumIterations == 0
            fhPattern = pseudosequence;
        elseif NumIterations > 0
            fhPattern = [repmat(pseudosequence, 1, NumIterations), zeros(1, 150)];
        end
        
        CurrentInt       = de2bi(fhPattern(1), NumBands, 'left-msb');
        NextPatternIndex = 1;
    case 'BURSTY' % Bursty interferer
        % CurrentInt = de2bi(randi(2^(NumBands)-1), NumBands, 'left-msb');
        CurrentInt   = de2bi(2.^(randi([1 NumBands])-1), NumBands, 'left-msb');
        bandDuration = ceil(exprnd(5, 1, 1));
        elapsedTime  = 0;
    case 'JAMMER' % Jamming interferer
        
    case 'DIRECTION-DEPENDENT-CONST' % Direction-dependent constant interferer
        IntfMask   = varargin{1};
        CurrentInt = zeros(size(IntfMask));
    case 'DIRECTION-DEPENDENT-INTER' % Direction-dependent intermittent interferer
        IntfMask   = varargin{1};
        IntermProb = varargin{2};
        CurrentInt = zeros(size(IntfMask));
end
CurrentInt   = repmat(CurrentInt, [1, 1, NumStatesInMemory]);
OriginalIntf = CurrentInt;

% Initialize interference states
IntfStatesSingle = de2bi([0:2^(NumBands)-1], NumBands, 'left-msb');
IntfStatesMat    = repmat(IntfStatesSingle, [1, 1, NumStatesInMemory]);

% Set the initial action, which is to use all bands
CurrentAction       = Actions(end,:);
CurrentActionNumber = size(Actions, 1);

% Create variables representing the number of velocities, SINRs,
% interference states, positions, number of actions and number of states
NumVelocities = size(TargetVelocities,1);
NumSINRs      = size(SINRs, 1);
% NumInt = size(InterferenceState,1);
NumInt     = size(IntfStatesSingle,1)^NumStatesInMemory;
NumPos     = size(TargetPositions,1);
NumActions = size(Actions, 1);
NumStates  = NumVelocities*NumSINRs*NumInt*NumPos;

% Initialize action, training and state histogram vectors
ActionHist          = zeros(NumActions,1);
TrainingHist        = zeros(NumActions,1);
StateHist           = zeros(NumStates, 1);
PositionXYHistogram = [TargetPositionsXY, zeros(size(TargetPositionsXY,1), 1)];

% Initialize policy, define reward matrix and reward count
policy = NumActions*ones(NumStates,1);          % set all ones
R_sparse = cell(1, NumActions);
for k = 1:NumActions
    R_sparse{k} = sparse(NumStates, NumStates);
end
RewardCount_sparse    = R_sparse;
R_sparse_unnormalized = R_sparse;

% Define transition probability matrix, as a 1xA cell of SxS sparse
% matricies, where A is the number of actions and S is the number of states
P_sparse = cell(1, NumActions);
for k = 1:NumActions
    P_sparse{k} = sparse(NumStates, NumStates);
end
P_hist_sparse = P_sparse;
% J - P_sparse

switch SimMethod
    case {'Random', 'Single'}
        NumTrainingRuns = NumRuns;
    case 'Uniform'
        NumTrainingRuns = NumRuns * size(TargetPositions, 1) * size(TargetVelocities, 1);     % double for position & velocities
        UniformPositionsAndVelocities = combvec(TargetPositions', TargetVelocities')';
end

% Define SINR, Range, Reward and Bandwidth history
SINR_hist  = zeros(NumTrainingRuns, length(t));
Range_hist = zeros(NumTrainingRuns, length(t));
R_hist     = zeros(NumTrainingRuns, length(t));
Bandwidth  = zeros(1, length(t));

% Initialize cell of training data
% TrainingData{:} is the data from one training run
% TrainingData{:}{1} is the target's initial position data
% TrainingData{:}{2} is the target's initial velocity data
% TrainingData{:}{3} is the target's position data versus time
% TrainingData{:}{4} is the interference states data versus time
TrainingData = cell(1, NumTrainingRuns);
for cellIndex = 1:numel(TrainingData)
    TrainingData{cellIndex} = cell(1, 4);
end

if solver == "dqn"
    n_features = 4;   % constant
    dqn_ = py.dqn.dqn(0, int32( NumBands*(NumBands+1)/2 ),  int32(n_features) ) ;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  OFFLINE TRAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  NumTrainingRuns <- t i.e. 60000
% Run a simulation of each training run
for kk = 1:(NumTrainingRuns)
    % Select a random position and random velocity
    switch SimMethod
        case 'Random' % Set random position and velocity
            position = TargetPositions(ceil(rand*size(TargetPositions,1)),:) + [0.1*randn(1,2), 0];
            velocity = TargetVelocities(ceil(rand*size(TargetVelocities,1)),:) + 0.0005*randn(1,2);
        case 'Uniform' % For each combination of position and velocity, make position and velocity random
            unifMatIndex     = ceil(kk/NumRuns);
            selectedposition = UniformPositionsAndVelocities(unifMatIndex, 1:3);
            selectedvelocity = UniformPositionsAndVelocities(unifMatIndex, 4:5);
            position         = selectedposition + [0.1*randn(1,2), 0];
            velocity         = selectedvelocity + 0.0005*randn(1,2);
        case 'Single' % Use only the one trajectory
            position = TrajFormat{1};
            velocity = TrajFormat{2};
    end
    
    % Initialize cumulative reward and target's position vector
    CumulativeReward = 0;
    positionvector = zeros(length(t), 3);
    
    % Initialize next pattern index if using pattern or pseudorandom mode
    switch InterferenceBehavior
        case 'FH-PATTERN'
            NextPatternIndex = 1;
        case 'FH-PSEUDO'
            NextPatternIndex = 0;
    end
    
    % Store the target's initial position and velocity
    TrainingData{kk}{1} = position;
    TrainingData{kk}{2} = velocity;
    
    % Calculate range, interference and noise power, received power, and
    % SINR in both linear and dB units
    Range  = norm(position);
    I      = sum(CurrentAction.*CurrentInt(:,:,1)) * IntPower;
    N      = Boltzmann * Ts * NF * sum(CurrentAction) * BandSize;
    Prnew  = Ptnew * G * G * lambda ^ 2 * sigma * TBnew * (sum(CurrentAction)/NumBands) * Np/(4*pi)^3/(Range*1000)^4;
    SINR   = Prnew / (I+N);
    SINRdB = 10 * log10(SINR);
    
    
    
    
    
    
    
    
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OFFLINE TRAIN CONTINUE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inner loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run through each time step of a single simulation run
    
    for i = 0: length(t)-1
        % Determine the current state
        [State, StateNumber] = MapState(position, TargetPositions, velocity, ...
            TargetVelocities, SINRdB, SINRs, CurrentInt, IntfStatesMat);
%         fprintf('StateNum = %d \n',StateNumber);
        
        observation = State; % for dqn
        
        % Select action based on the radar's behavior
        switch RadarBehavior
            case 'OPTI' % Optimal case; perform exploration by selecting a random action on each time instant
                CurrentActionNumber = ceil(rand*NumActions);
            case 'CONS' % Constant case; select the same action each time
                CurrentActionNumber = CurrentActionNumber;
            case 'RAND' % Random case; select a random action each time
                CurrentActionNumber = ceil(rand*NumActions);
        end
        
        % Determine the action from CurrentActionNumber and record the amount of bandwidth the radar will occupy
        CurrentAction = Actions(CurrentActionNumber,:);
        Bandwidth(i+1) = BandSize*sum(CurrentAction);
        
        CurrentInt = circshift(CurrentInt, 1, 3);
        OldInt = CurrentInt(:,:,((NumStatesInMemory >= 2) + 1));
        % Update interference based on interference behavior
        switch InterferenceBehavior
            case 'CONST' % Constant interferer
                NewInt = UpdateInterference(InterferenceBehavior, OldInt);
            case 'AVOID' % Avoiding/evading interferer
                NewInt = UpdateInterference(InterferenceBehavior, OriginalIntf(:,:,1), CurrentAction, IntfStatesSingle);
            case 'INTER' % Intermittent interferer
                NewInt = UpdateInterference(InterferenceBehavior, OriginalIntf(:,:,1), IntermProb);
            case {'FH-TRIANGLE', 'FH-SAWTOOTH'} % Triangle sweep or sawtooth sweep frequency hopper
                [NewInt, NextSweepState] = UpdateInterference(InterferenceBehavior, OldInt, NextSweepState);
            case {'FH-PATTERN', 'FH-PSEUDO'} % User-defined or pseudorandom sequence frequency hopper
                [NewInt, NextPatternIndex] = UpdateInterference(InterferenceBehavior, NextPatternIndex, fhPattern, NumBands);
            case 'BURSTY' % Bursty interferer
                [NewInt, bandDuration, elapsedTime] = UpdateInterference(InterferenceBehavior, OldInt, bandDuration, elapsedTime, NumBands);
            case 'JAMMER' % Jamming interferer
                [NewInt] = UpdateInterference(InterferenceBehavior, CurrentAction);
            case 'DIRECTION-DEPENDENT-CONST' % Direction-dependent constant interferer
                [NewInt] = UpdateInterference(InterferenceBehavior, State(1), states_with_interference, IntfMask);
            case 'DIRECTION-DEPENDENT-INTER' % Direction-dependent intermittent interferer
                [NewInt] = UpdateInterference(InterferenceBehavior, OriginalIntf(:,:,1), IntermProb,...
                    State(1), states_with_interference, IntfMask);
        end
        CurrentInt(:,:,1) = NewInt;
        
        % Update position, store it, and compute target range
        position               = [position(1)+velocity(1)*TimeInterval, position(2)+velocity(2)*TimeInterval, position(3)];
        positionvector(i+1, :) = position;
        Range                  = norm(position);
        
        % Update SINR
        I      = sum(CurrentAction.*CurrentInt(:,:,1))*IntPower;
        N      = Boltzmann*Ts*NF*sum(CurrentAction)*BandSize;
        Prnew  = Ptnew*G*G*lambda^2*sigma*TBnew*(sum(CurrentAction)/NumBands)*Np/(4*pi)^3/(Range*1000)^4;
        SINR   = Prnew/(I+N);
        SINRdB = 10*log10(SINR);
        

        
        % CurrentActionNumber -> Bandwidth
        % ========================================================================
        % =================== UPDATE
        [State, StateNumber]   = MapState(position, TargetPositions, velocity, ...
            TargetVelocities, SINRdB, SINRs, CurrentInt, IntfStatesMat);

        CurrentReward = CalculateReward(SINRdB, CurrentAction, NumBands);

        
        
        
        if solver == "mdp"
            
            % Store SINR and range of target
            SINR_hist(kk,i+1)  = SINRdB;
            Range_hist(kk,i+1) = Range;
            
            % Store the current state as the old state and determine the new
            % state and statenumber and increment the state history for the
            % newly determined state
            OldStateNumber         = StateNumber;
                     
            % State = [position_number, velocity_number, interference_number, sinr_number]
            StateHist(StateNumber) = StateHist(StateNumber) + 1;

            % Increment the position and training histogram
            PositionXYHistogram(State(1), 3)  = PositionXYHistogram(State(1), 3) + 1;
            TrainingHist(CurrentActionNumber) = TrainingHist(CurrentActionNumber) + 1;

            % Increment history of state transitions
            P_hist_sparse{CurrentActionNumber}(OldStateNumber,StateNumber) = ...
                P_hist_sparse{CurrentActionNumber}(OldStateNumber,StateNumber) + 1;
            % ################ Update P matrix ###############################
            % ####### P_hist_sparse <- StateNumber <- MapState( .. CurrentInt) <- NewInt <- UpdateInterference(.. CurrentInt.. )

            RewardCount_sparse{CurrentActionNumber}(OldStateNumber, StateNumber) = ...
                RewardCount_sparse{CurrentActionNumber}(OldStateNumber, StateNumber) + 1;
            % ################ Update R matrix ###############################
            R_sparse_unnormalized{CurrentActionNumber}(OldStateNumber, StateNumber) = ...
                R_sparse_unnormalized{CurrentActionNumber}(OldStateNumber, StateNumber) + CurrentReward;
            CumulativeReward = CumulativeReward + CurrentReward;
            R_hist(kk,i+1)   = CumulativeReward;
            
        else
            % ===============  DQN OFFLINE: 1. store (s, a, r, s_) into
            % memory  2. learn ========================================
            observation_ = State ;
            dqn_.store_transition(int32(observation), int32(CurrentActionNumber-1),...
                int32(CurrentReward), int32(observation_));                          
            if i > ceil(length(t)/4)
                if mod(i,5) == 0
                    dqn_.learn();
                end
            end
        end
        
        
        % Store the interference history
        % TrainingData{kk}{4}(i+1) = bi2de(CurrentInt(:,:,1), 'left-msb');
        
        %if (abs(TargetPositions(State(1),1) - position(1)) > 1.5) || (abs(TargetPositions(State(1),2) - position(2)) > 1.5)
        %   % Fill SINR_hist, Range_hist, R_hist, Bandwidth with NaNs so
        %   % the plot shows zero when the target goes out of bounds, and
        %   % trim the remaining zeros from the end of the position vector
        %   SINR_hist(kk, (i+1):end) = nan(1, size(SINR_hist, 2)-i);
        %   Range_hist(kk, (i+1):end) = nan(1, size(SINR_hist, 2)-i);
        %   R_hist(kk, (i+1):end) = nan(1, size(SINR_hist, 2)-i);
        %   Bandwidth(1, (i+1):end) = nan(1, size(SINR_hist, 2)-i);
        %   positionvector = positionvector(1:(i+1),:);
        %
        %   break;
        %end
    end
    
    % show process bar
    if kk == floor(NumTrainingRuns/10)
        disp('Progress 10% ');
        toc;
    elseif kk == floor(NumTrainingRuns*2/10)
        disp('Progress 20% ');
        toc;
    elseif kk == floor(NumTrainingRuns*3/10)
        disp('Progress 30% ');
        toc;
    elseif kk == floor(NumTrainingRuns*4/10)
        disp('Progress 40% ');
        toc;
    elseif kk == floor(NumTrainingRuns*5/10)
        disp('Progress 50% ');
        toc;
    elseif kk == floor(NumTrainingRuns*6/10)
        disp('Progress 60% ');
        toc;
    elseif kk == floor(NumTrainingRuns*7/10)
        disp('Progress 70% ');
        toc;       
    elseif kk == floor(NumTrainingRuns*8/10)
        disp('Progress 80% ');
        toc;
    elseif kk == floor(NumTrainingRuns*9/10)
        disp('Progress 90% ');
        toc;    
    elseif kk == floor(NumTrainingRuns*10/10)
        disp('Progress 100% ');
        toc;       
    end
   
    % Store target position history versus time
    TrainingData{kk}{3} = positionvector;
end
disp("-------------- OFFLINE ends ------------------");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OFFLINE TRAIN ENDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






if solver == "mdp"        
    P_sparse_test = cell(1, NumActions);
    R_sparse_test = cell(1, NumActions);
    for k = 1:NumActions
        P_sparse_rowsums = sum(P_hist_sparse{k}, 2);
        P_sparse_test{k} = bsxfun(@times, P_hist_sparse{k}, spfun(@(x) 1./x, P_sparse_rowsums));
        % ##################### P_sparse_test <- P_hist_sparse
        R_sparse_test{k} = bsxfun(@times, R_sparse_unnormalized{k}, spfun(@(x) 1./x, RewardCount_sparse{k}));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~, policy] = mdp_policy_iteration(P_sparse_test, R_sparse_test, DiscountFactor);
    % ################## Get policy by P, R ###################################
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % J in Chris [~,obj.policy] = mdp_policy_iteration(obj.avgStateTrans,obj.rewardTrans,obj.discountFactor);
    % 1. policy
    % 2. P_sparse_test
    % 3. R_sparse_test
    % 4. DiscountFactor
    % NOTICE, offline train
end

% Get a timestamp and create the foldername where results will be stored
FolderDateTimeStr = datestr(now, 'yyyy-mmm-dd-HHMMSS');
switch InterferenceBehavior
    case 'CONST'
        foldername = sprintf('%s-%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'CONST', sprintf('%d', OriginalIntf), NumStatesInMemory);
    case 'INTER'
        foldername = sprintf('%s-%s-%s-P0%2G-%dMEMSTATE-Results', FolderDateTimeStr, 'INTER', ...
            sprintf('%d', OriginalIntf), IntermProb*100, NumStatesInMemory);
    case 'AVOID'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'AVOID', NumStatesInMemory);
    case 'FH-TRIANGLE'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'FH-TRIANGLE', NumStatesInMemory);
    case 'FH-SAWTOOTH'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'FH-SAWTOOTH', NumStatesInMemory);
    case 'FH-PATTERN'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'FH-PATTERN', NumStatesInMemory);
    case 'FH-PSEUDO'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'FH-PSEUDO', NumStatesInMemory);
    case 'BURSTY'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'BURSTY', NumStatesInMemory);
    case 'JAMMER'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'JAMMER', NumStatesInMemory);
    case 'DIRECTION-DEPENDENT-CONST'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'DIRECTION-CONST', NumStatesInMemory);
    case 'DIRECTION-DEPENDENT-INTER'
        foldername = sprintf('%s-%s-%dMEMSTATE-Results', FolderDateTimeStr, 'DIRECTION-INTER', NumStatesInMemory);
end

% Set folder path based on operating system, and then create the directory
if ismac || isunix
    folderstr = sprintf('./%s/', foldername);
elseif ispc
    folderstr = sprintf('.\%s\', foldername);
end
[SUCESS, MSG, MDGID] = mkdir(folderstr);

switch InterferenceBehavior
    case 'CONST' % Constant interferer
        CurrentInt = varargin{1};
    case 'INTER' % Intermittent interferer
        CurrentInt = varargin{1};
        IntermProb = varargin{2};
    case 'AVOID' % Avoiding interferer
        CurrentInt = varargin{1};
    case 'FH-TRIANGLE' % Triangle frequency sweep interferer
        CurrentInt = varargin{1};
        NextSweepState = varargin{2};
    case 'FH-SAWTOOTH' % Sawtooth frequency sweep interferer
        CurrentInt = varargin{1};
        NextSweepState = varargin{2};
    case 'FH-PATTERN' % Frequency hopping pattern interferer
        fhPattern = varargin{1};
        NumIterations = varargin{2};
        NextPatternIndex = 1;
        CurrentInt = de2bi(fhPattern(1), NumBands, 'left-msb');
    case 'FH-PSEUDO' % Pseudorandom frequency hopping interferer
        PseudoLength = varargin{1};
        NumIterations = varargin{2};
        pseudosequence = randi([1 (2^(NumBands)-1)], 1, PseudoLength);
        if NumIterations == 0
            fhPattern = pseudosequence;
        elseif NumIterations > 0
            fhPattern = [repmat(pseudosequence, 1, NumIterations), zeros(1, 150)];
        end
        
        CurrentInt = de2bi(fhPattern(1), NumBands, 'left-msb');
        NextPatternIndex = 1;
    case 'BURSTY' % Bursty interferer
        % CurrentInt = de2bi(randi(2^(NumBands)-1), NumBands, 'left-msb');
        CurrentInt = de2bi(2.^(randi([1 NumBands])-1), NumBands, 'left-msb');
        bandDuration = ceil(exprnd(5, 1, 1));
        elapsedTime = 0;
    case 'JAMMER' % Jamming interferer
        
    case 'DIRECTION-DEPENDENT-CONST' % Direction-dependent constant interferer
        IntfMask = varargin{1};
        CurrentInt = zeros(size(IntfMask));
    case 'DIRECTION-DEPENDENT-INTER' % Direction-dependent intermittent interferer
        IntfMask = varargin{1};
        IntermProb = varargin{2};
        CurrentInt = zeros(size(IntfMask));
end
CurrentInt = repmat(CurrentInt, [1, 1, NumStatesInMemory]);
OriginalIntf = CurrentInt;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ONLINE EVAL WRAPPER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For loop for evaluating the policy
for evalIndx = 1:NumEvaluations
    % J - NumEvaluations always 1
    
    CumulativeReward = 0;
    
    switch EvalMethod
        case 'EvalOnTrained'
            lengthsOfPositions = zeros(1, length(TrainingData));
            for j = 1:length(TrainingData)
                lengthsOfPositions(j) = size(TrainingData{j}{3}, 1);
            end
            [~, J] = find(lengthsOfPositions > 0.9*TimeSteps); pos = randi(length(J));
            randRun = J(pos);
            position = TrainingData{randRun}{1} + [0.1*randn(1,2), 0];
            velocity = TrainingData{randRun}{2} + 0.0005*randn(1,2);
        case 'EvalOnNew'
            position = EvalTraj{evalIndx}{1};
            velocity = EvalTraj{evalIndx}{2};
    end
    
    Range = norm(position);
    I = sum(CurrentAction.*CurrentInt(:,:,1))*IntPower;
    N = Boltzmann*Ts*NF*sum(CurrentAction)*BandSize;
    Prnew = Ptnew*G*G*lambda^2*sigma*TBnew*(sum(CurrentAction)/NumBands)*Np/(4*pi)^3/(Range*1000)^4;
    SINR = Prnew/(I+N);
    SINRdB = 10*log10(SINR);
    
    %TimeStepsEval = 150;
    %t = 0:TimeInterval:(TimeStepsEval-1)*TimeInterval;
    
    SINR_hist_eval = zeros(1, length(t));
    Range_hist_eval = zeros(1, length(t));
    R_hist_eval = zeros(1, length(t));
    Bandwidth_eval = zeros(1, length(t));
    action_history_eval = zeros(1, length(t));
    intf_history_eval = zeros(1, length(t));
    position_eval = zeros(length(t), 3);
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% ONLINE EVAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    for i=0:length(t)-1
        
        % Determine state and state number
        [State, StateNumber] = MapState(position, TargetPositions, velocity, TargetVelocities, SINRdB, SINRs, CurrentInt, IntfStatesMat);
        
        % Pick the current action and current action number according to the policy and record the amount of bandwidth
        if solver == "mdp"
            CurrentActionNumber    = policy(StateNumber);
        else
            % ================ ONE EVAL: DQN make action ======================
            % NOTICE dqn_.choose_action return vaule start from 0 while
            % matlab start from I.
            dqn_CurrentActionNumber = dqn_.choose_action(int32(observation));
            CurrentActionNumber = dqn_CurrentActionNumber + 1;            
        end
        % ################## Get action by policy ###################################
        % J - getAction by policy
        % J track CurrentActionNumber
        % ############# policy -> CurrentActionNumber -> CurrentAction ->
        % "CurrentReward " -> UpdateInterference() -> NewInt -> CurrentInt -> MapState() ->
        CurrentAction          = Actions(CurrentActionNumber,:);
        Bandwidth_eval(i+1)    = BandSize*sum(CurrentAction);
        % J - Bandwidth_eval for plot
        
        intf_history_eval(i+1) = bin2dec(num2str(CurrentInt(:,:,1)));
        
        CurrentInt             = circshift(CurrentInt, 1, 3);
        OldInt                 = CurrentInt(:,:,((NumStatesInMemory >= 2) + 1));
        switch InterferenceBehavior
            case 'CONST' % Constant interferer
                NewInt = UpdateInterference(InterferenceBehavior, OldInt);
            case 'AVOID' % Avoiding/evading interferer
                NewInt = UpdateInterference(InterferenceBehavior, OriginalIntf(:,:,1), CurrentAction, IntfStatesSingle);
                % J - UpdateInterference, various input, track 'AVIOD'
                % J - equal likely new state s_, "s, a, r, s_"
            case 'INTER' % Intermittent interferer
                NewInt = UpdateInterference(InterferenceBehavior, OriginalIntf(:,:,1), IntermProb);
            case {'FH-TRIANGLE', 'FH-SAWTOOTH'} % Triangle sweep or sawtooth sweep frequency hopper
                [NewInt, NextSweepState] = UpdateInterference(InterferenceBehavior, OldInt, NextSweepState);
            case {'FH-PATTERN', 'FH-PSEUDO'} % User-defined or pseudorandom sequence frequency hopper
                [NewInt, NextPatternIndex] = UpdateInterference(InterferenceBehavior, NextPatternIndex, fhPattern, NumBands);
            case 'BURSTY' % Bursty interferer
                [NewInt, bandDuration, elapsedTime] = UpdateInterference(InterferenceBehavior, OldInt, ...
                    bandDuration, elapsedTime, NumBands);
            case 'JAMMER' % Jamming interferer
                [NewInt] = UpdateInterference(InterferenceBehavior, CurrentAction);
            case 'DIRECTION-DEPENDENT-CONST' % Direction-dependent constant interferer
                [NewInt] = UpdateInterference(InterferenceBehavior, State(1), states_with_interference, IntfMask);
            case 'DIRECTION-DEPENDENT-INTER' % Direction-dependent intermittent interferer
                [NewInt] = UpdateInterference(InterferenceBehavior, OriginalIntf(:,:,1), IntermProb, ...
                    State(1), states_with_interference, IntfMask);
        end
        CurrentInt(:,:,1) = NewInt;
        
        % Update position, store it, and compute target range
        position              = [position(1)+velocity(1)*TimeInterval, ...
            position(2)+velocity(2)*TimeInterval, position(3)];
        Range                 = norm(position);
        position_eval(i+1, :) = position;
        
        % Update SINR
        I      = sum(CurrentAction.*CurrentInt(:,:,1))*IntPower;
        N      = Boltzmann*Ts*NF*sum(CurrentAction)*BandSize;
        Prnew  = Ptnew*G*G*lambda^2*sigma*TBnew*(sum(CurrentAction)/NumBands)*Np/(4*pi)^3/(Range*1000)^4;
        SINR   = Prnew/(I+N);
        SINRdB = 10*log10(SINR);
        
        SINR_hist_eval(1,i+1)  = SINRdB;
        Range_hist_eval(1,i+1) = Range;   % all variable ending with _eval is for further plot
        
        [State, StateNumber]            = MapState(position, TargetPositions, velocity, TargetVelocities,...
            SINRdB, SINRs, CurrentInt, IntfStatesMat);
        
        StateHist(StateNumber)          = StateHist(StateNumber) + 1;
        
        ActionHist(CurrentActionNumber) = ActionHist(CurrentActionNumber) + 1;
        action_history_eval(i+1)        = bin2dec(num2str(Actions(CurrentActionNumber, :)));
        
        
        CurrentReward                   = CalculateReward(SINRdB, CurrentAction, NumBands);
        % ################## Get reward by action ###################################
        % J - get reward
        CumulativeReward                = CumulativeReward + CurrentReward;
        R_hist_eval(1,i+1)              = CumulativeReward;
        
        % If x position or y position is more than 25 units (km) away from
        % the nearest state, then stoop; otherwise continue
        %if (abs(TargetPositions(State(1),1) - position(1)) > 3) || (abs(TargetPositions(State(1),2) - position(2)) > 3)
        %    SINR_hist_eval = SINR_hist_eval(1:(i+1),:);
        %    Range_hist_eval = Range_hist_eval(1:(i+1),:);
        %    R_hist_eval = R_hist_eval(1:(i+1),:);
        %    Bandwidth_eval = Bandwidth_eval(1:(i+1),:);
        %    action_history_eval = action_history_eval(1:(i+1),:);
        %    intf_history_eval = intf_history_eval(1:(i+1),:);
        %
        %    position_eval = position_eval(1:(i+1),:);
        %    break;
        %end
        
        
        
        % ############# update DQN solver #################        
        if solver == "dqn"
            observation_ = State;
            dqn_.store_transition(int32(observation), int32(dqn_CurrentActionNumber),...
                int32(CurrentReward), int32(observation_))            
            dqn_.learn()
        end        
    end

    disp("-------------- ONLINE ends ------------------");
    toc;
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%  PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DateTimeStr = datestr(now, 'yyyy-mmm-dd-HHMMSS');
    switch InterferenceBehavior
        case 'CONST'
            figstr = sprintf('%s-%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'CONST', sprintf('%d', OriginalIntf), NumStatesInMemory, evalIndx);
            figtitle = sprintf('Constant Interferer');
        case 'INTER'
            figstr = sprintf('%s-%s-%s-P0%2G-%dMEMSTATE-N%d', DateTimeStr, 'INTER', sprintf('%d', OriginalIntf), IntermProb*100, NumStatesInMemory, evalIndx);
            figtitle = sprintf('Intermittent Interferer');
        case 'AVOID'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'AVOID',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Avoiding/Evading Interferer');
        case 'FH-TRIANGLE'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'FH-TRIANGLE',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Frequency Hopping Interferer, Triangular Frequency Sweep');
        case 'FH-SAWTOOTH'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'FH-SAWTOOTH',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Frequency Hopping Interferer, Sawtooth Frequency Sweep');
        case 'FH-PATTERN'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'FH-PATTERN',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Frequency Hopping Interferer, Pattern');
        case 'FH-PSEUDO'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'FH-PSEUDO',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Frequency Hopping Interferer, Pseudorandom');
        case 'BURSTY'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'BURSTY',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Bursty Interferer');
        case 'JAMMER'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'JAMMER',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Jammer Interferer');
        case 'DIRECTION-DEPENDENT-CONST'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'DIRECTION-CONST',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Direction-Dependent Constant Interferer');
        case 'DIRECTION-DEPENDENT-INTER'
            figstr = sprintf('%s-%s-%dMEMSTATE-N%d', DateTimeStr, 'DIRECTION-INTER',  NumStatesInMemory, evalIndx);
            figtitle = sprintf('Direction-Dependent Intermittent Interferer');
    end
    
    f = figure;
    markerstep = 8;
    switch IncludeTrajPlot
        case 'Include'
            subaxis(2,2,1,1, 'MarginLeft', 0.08);
            [hAX, hLine1, hLine2] = plotyy([t', t'], [R_hist_eval(1,:)'/100, Bandwidth_eval'/1e6], [t', t'], [SINR_hist_eval(1,:)', Range_hist_eval(1,:)']);
            hold(hAX(1), 'on'); hold(hAX(2), 'on');
            p1 = plot(hAX(1), t(1:markerstep:end), R_hist_eval(1:markerstep:end)/100, '*', 'Color', hLine1(1).Color);
            p2 = plot(hAX(1), t(1:markerstep:end), Bandwidth_eval(1:markerstep:end)/1e6, 'd', 'Color', hLine1(2).Color);
            p3 = plot(hAX(2), t(1:markerstep:end), SINR_hist_eval(1:markerstep:end), '^', 'Color', hLine2(1).Color);
            p4 = plot(hAX(2), t(1:markerstep:end), Range_hist_eval(1:markerstep:end), 'o', 'Color', hLine2(2).Color);
            hold(hAX(1), 'off'); hold(hAX(2), 'off');
            legend([p1, p2, p3, p4], 'Rewards (x100)','Bandwidth (MHz)','SINR (dB)','Range (km)', 'Location', 'SouthOutside', 'Orientation', 'Horizontal');
            ylabel(hAX(1), {sprintf('\\color[rgb]{%s}Rewards (x100)', sprintf('%f ', hLine1(1).Color)), ...
                sprintf('\\color[rgb]{%s}Bandwidth (MHz)', sprintf('%f ', hLine1(2).Color))});
            ylabel(hAX(2), {sprintf('{\\color[rgb]{%s}SINR (dB)}', sprintf('%f ', hLine2(1).Color)), ...
                sprintf('{\\color[rgb]{%s}Range (km)}', sprintf('%f ', hLine2(2).Color))});
            xlabel('Time (sec)');
            title('History of Rewards and State Variables');
            axis(hAX(1), 'fill'); axis(hAX(2), 'fill');
            hLine1(1).LineWidth = 1.5; hLine1(2).LineWidth = 1.5;
            hLine2(1).LineWidth = 1.5; hLine2(2).LineWidth = 1.5;
            box(hAX(1), 'off'); box(hAX(2), 'on');
            box on;
            switch TargetTravelMode
                case 'Cross-Range'
                    ylim(hAX(1), [-20, 120]);
                    ylim(hAX(2), [-8, 20]);
                case 'Down-Range'
                    ylim(hAX(1), [-20, 120]);
                    ylim(hAX(2), [-4, 24]);
                    %                     hAX(2).YTick = floor([-8:(24+8)/7:24]);
                    %                     hAX(2).YTickLabels = cellfun(@num2str, num2cell(floor([-8:(24+8)/7:24])), 'UniformOutput', 0);
            end
            %ylim(hAX(1), [-20, 120]);
            %ylim(hAX(2), [-8, 20]);
            labelrange = [-20:20:120]; set(hAX(1), 'YTick', labelrange, 'YTickLabels', arrayfun(@num2str, labelrange, 'UniformOutput', false));
            labelrange = [-8:4:20]; set(hAX(2), 'YTick', labelrange, 'YTickLabels', arrayfun(@num2str, labelrange, 'UniformOutput', false));
            set(hAX(1), 'Layer', 'top'); set(hAX(2), 'Layer', 'top');
            
            subaxis(2,2,1,2, 'MarginLeft', 0.08, 'MarginBottom', 0.05);
            hold on;
            p5 = plot(t, action_history_eval, 'LineWidth', 1.5);
            p6 = plot(t, intf_history_eval, 'LineWidth', 1.5);
            p7 = plot(t(1:markerstep:end), action_history_eval(1:markerstep:end), 'x', 'LineWidth', 1.5, 'Color', p5.Color);
            p8 = plot(t(1:markerstep:end), intf_history_eval(1:markerstep:end), 's', 'LineWidth', 1.5, 'Colo', p6.Color);
            hold off;
            xlabel('Time (sec)');
            ylabel('Action and Interference States');
            legend([p7, p8], 'Action', 'Interference', 'Location', 'SouthOutside', 'Orientation', 'Horizontal');
            title('History of Actions and Interference');
            ylim([0 2^(NumBands)]);
            box on;
            set(gca, 'Layer', 'top');
            
            subaxis(2,2,2,1,1,2, 'PaddingLeft', 0.05, 'MarginBottom', 0.05, 'MarginRight', 0.08);
            tgtpos = position_eval;
            hold on;
            h = plot(tgtpos(:,1), tgtpos(:,2), 'LineWidth', 1.5);
            plot(TargetPositions(:,1), TargetPositions(:,2),'o', 'LineWidth', 1.5)
            plot(0, 0, 's', 'MarkerSize', 8, 'LineWidth', 1.5)
            hold off;
            line2arrow(h);
            xlabel('Cross-Range (km)');
            ylabel('Down-Range (km)');
            legend('Target Trajectory', 'Position States', 'Radar', 'Location', 'SouthOutside', 'Orientation', 'Horizontal');
            title('Target Trajectory and Position States');
            axis([-6 6 0 6])
            box on;
            set(gca, 'Layer', 'top');
            
            annotation(f, 'textbox', [0 0.875 1 0.1], 'String', figtitle, ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
            
        case 'Separate'
            subaxis(2,1,1, 'MarginLeft', 0.08);
            [hAX, hLine1, hLine2] = plotyy([t', t'], [R_hist_eval(1,:)'/100, Bandwidth_eval'/1e6], [t', t'], [SINR_hist_eval(1,:)', Range_hist_eval(1,:)']);
            hold(hAX(1), 'on'); hold(hAX(2), 'on');
            p1 = plot(hAX(1), t(1:markerstep:end), R_hist_eval(1:markerstep:end)/100, '*', 'Color', hLine1(1).Color);
            p2 = plot(hAX(1), t(1:markerstep:end), Bandwidth_eval(1:markerstep:end)/1e6, 'd', 'Color', hLine1(2).Color);
            p3 = plot(hAX(2), t(1:markerstep:end), SINR_hist_eval(1:markerstep:end), '^', 'Color', hLine2(1).Color);
            p4 = plot(hAX(2), t(1:markerstep:end), Range_hist_eval(1:markerstep:end), 'o', 'Color', hLine2(2).Color);
            hold(hAX(1), 'off'); hold(hAX(2), 'off');
            legend([p1, p2, p3, p4], 'Rewards (x100)','Bandwidth (MHz)','SINR (dB)','Range (km)', 'Location', 'SouthOutside', 'Orientation', 'Horizontal');
            ylabel(hAX(1), {sprintf('\\color[rgb]{%s}Rewards (x100)', sprintf('%f ', hLine1(1).Color)), ...
                sprintf('\\color[rgb]{%s}Bandwidth (MHz)', sprintf('%f ', hLine1(2).Color))});
            ylabel(hAX(2), {sprintf('{\\color[rgb]{%s}SINR (dB)}', sprintf('%f ', hLine2(1).Color)), ...
                sprintf('{\\color[rgb]{%s}Range (km)}', sprintf('%f ', hLine2(2).Color))});
            xlabel('Time (sec)');
            title('History of Rewards and State Variables');
            axis(hAX(1), 'fill'); axis(hAX(2), 'fill');
            hLine1(1).LineWidth = 1.5; hLine1(2).LineWidth = 1.5;
            hLine2(1).LineWidth = 1.5; hLine2(2).LineWidth = 1.5;
            box(hAX(1), 'off'); box(hAX(2), 'on');
            box on;
            switch TargetTravelMode
                case 'Cross-Range'
                    ylim(hAX(1), [-20, 120]);
                    ylim(hAX(2), [-8, 20]);
                case 'Down-Range'
                    ylim(hAX(1), [-20, 120]);
                    ylim(hAX(2), [-4, 24]);
                    %                     hAX(2).YTick = floor([-8:(24+8)/7:24]);
                    %                     hAX(2).YTickLabels = cellfun(@num2str, num2cell(floor([-8:(24+8)/7:24])), 'UniformOutput', 0);
            end
            %ylim(hAX(1), [-20, 120]);
            %ylim(hAX(2), [-8, 20]);
            labelrange = [-20:20:120]; set(hAX(1), 'YTick', labelrange, 'YTickLabels', arrayfun(@num2str, labelrange, 'UniformOutput', false));
            labelrange = [-8:4:20]; set(hAX(2), 'YTick', labelrange, 'YTickLabels', arrayfun(@num2str, labelrange, 'UniformOutput', false));
            set(hAX(1), 'Layer', 'top'); set(hAX(2), 'Layer', 'top');
            
            subaxis(2,1,2, 'MarginLeft', 0.08, 'MarginBottom', 0.05);
            hold on;
            p5 = plot(t, action_history_eval, 'LineWidth', 1.5);
            p6 = plot(t, intf_history_eval, 'LineWidth', 1.5);
            p7 = plot(t(1:markerstep:end), action_history_eval(1:markerstep:end), 'x', 'LineWidth', 1.5, 'Color', p5.Color);
            p8 = plot(t(1:markerstep:end), intf_history_eval(1:markerstep:end), 's', 'LineWidth', 1.5, 'Colo', p6.Color);
            hold off;
            xlabel('Time (sec)');
            ylabel({'Action and'; 'Interference States'});
            legend([p7, p8], 'Action', 'Interference', 'Location', 'SouthOutside', 'Orientation', 'Horizontal');
            title('History of Actions and Interference');
            ylim([0 2^(NumBands)]);
            box on;
            set(gca, 'Layer', 'top');
            
            ftraj = figure;
            tgtpos = position_eval;
            hold on;
            h = plot(tgtpos(:,1), tgtpos(:,2), 'LineWidth', 1.5);
            plot(TargetPositions(:,1), TargetPositions(:,2),'o', 'LineWidth', 1.5)
            plot(0, 0, 's', 'MarkerSize', 8, 'LineWidth', 1.5)
            hold off;
            line2arrow(h);
            xlabel('Cross-Range (km)');
            ylabel('Down-Range (km)');
            legend('Target Trajectory', 'Position States', 'Radar', 'Location', 'SouthOutside', 'Orientation', 'Horizontal');
            title('Target Trajectory and Position States');
            axis([-6 6 0 6])
            box on;
            set(gca, 'Layer', 'top');
            
            annotation(f, 'textbox', [0 0.875 1 0.1], 'String', figtitle, ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    end
    
    
    % [ax, h3] = suplabel(figtitle, 't');
    % set(h3, 'FontSize', 12);
    
    %     annotation('textbox', [0 0.875 1 0.1], 'String', figtitle, ...
    %         'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold')
    
    %figure('units', 'normalized', 'outerposition', [0 0 1 1]);
    set(f, 'units', 'normalized', 'outerposition', [0 0 1 1]);
    line2arrow(h);
    
    figfilename = sprintf('%s%s', folderstr, figstr);
    savefig(f, figfilename);
    if exist('ftraj', 'var')
        savefig(ftraj, sprintf('%s%s-TRAJ', folderstr, figstr));
    end
    
    pdfname = sprintf('%s.pdf', figfilename);
    % export_fig figfilename -pdf -transparent
    switch EnableExport
        case 'Export'
            export_fig(figfilename, '-pdf', '-transparent' , '-c 25 25 25 25');
            saveas(f, figfilename, 'svg');
            [cmdstat, cmdres] = system(sprintf('inkscape -f %s.svg -A %s --export-area-drawing', figfilename, pdfname));
            [cmdstat, cmdres] = system(sprintf('pdfcrop %s %s', pdfname, pdfname));
        case 'DoNotExport'
            
    end
    % Do not close figure after completion
    % close(f);
    if exist('ftraj', 'var')
        close(ftraj);
    end
    
    %!pdfcrop pdfname pdfname
    %[sys_stat, sys_result] = system(sprintf('pdfcrop --margins 100 --noclip %s %s ', pdfname, pdfname));
end

% Evaluate policy with a different instance it hasn't seen before

% Create function for evaluating policy
end
