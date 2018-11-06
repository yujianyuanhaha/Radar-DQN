function varargout = UpdateInterference(InterferenceBehavior, varargin)
% function [CurrentInt, varargout] = UpdateInterference(OldInt, InterferenceBehavior, Action, InterferenceState, IntProb, varargin)
    
% Determine interference based on its behavior 
switch InterferenceBehavior
    % Constant interferer % [CurrentInt] = UpdateInterference(InterferenceBehavior, OldInt)
    case 'CONST'
        % Set variables from input arguments
        OldInt = varargin{1};
        
        % Set current interference equal to old interference
        CurrentInt = OldInt;
        
        % Set output arguments
        varargout{1} = CurrentInt;
        
    % Avoiding interferer % [CurrentInt] = UpdateInterference(InterferenceBehavior, OldInt, Action, InterferenceState)
    case 'AVOID'
        % Set variables from input arguments
        OldInt = varargin{1};
        Action = varargin{2};
        InterferenceState = varargin{3};
        
        % J -  NewInt = UpdateInterference(InterferenceBehavior, ...
        % OriginalIntf(:,:,1), CurrentAction, IntfStatesSingle);
        
        % Determine how the interference will avoid the radar
        % TODO: Add more comments here after analyzing code
        % TODO: Change the default location the interferer uses when all
        % bands are occupied by the radar (change [1 0 0 0] to something
        % user defined)
        num_bands = sum(OldInt);
        tx_bands = sum(InterferenceState');
        ind = find(tx_bands<=num_bands & tx_bands>0);
        [tmp,ind2] = sort(tx_bands(ind));
        chosen = 0;
        i=length(ind2);
        while(~chosen && i>0);
            ind3 = find(~Action);
            tmp = InterferenceState(ind(ind2(i)),:);
            tmp2 = tmp(:,ind3);
            if tx_bands(ind(ind2(i))) == sum(tmp2);
                chosen = 1;
                CurrentInt = InterferenceState(ind(ind2(i)),:);
            end
            i = i-1;
        end
        if ~chosen
            % CurrentInt = [1 0 0 0]; % Delete this line
            CurrentInt = [1, zeros(1, numel(OldInt)-1)];
        end
        
        % Set output arguments
        varargout{1} = CurrentInt;
        % J - return CurrentInt <- InterferenceState <- Action
        
    % Intermittent interferer % [CurrentInt] = UpdateInterference(InterferenceBehavior, OldInt, IntProb)
    case 'INTER'
        % Set variables from input arguments
        OldInt = varargin{1};
        IntProb = varargin{2};
        tmp = rand;
        if nargin < 3
            IntProb = 0.5;
        end
        if tmp < IntProb
            CurrentInt = OldInt;
        else
            % CurrentInt = [0 0 0 0]; % Delete this line
            CurrentInt = zeros(1, numel(OldInt));
        end
        
        % Set output arguments
        varargout{1} = CurrentInt;
        
    % Frequency hopping linear sweep
    % Frequency hopper, triangular frequency sweep % [CurrentInt, NextSweepState] = UpdateInterference(InterferenceBehavior, OldInt, SweepState)
    case 'FH-TRIANGLE'
        % Set variables from input arguments
        OldInt = varargin{1};
        SweepState = varargin{2};
        
        [~, bandIndex] = find(OldInt);
        % SweepState = varargin{1}; % No longer needed
        switch SweepState
            case 'Up'
                % Is the occupied band at the upper limit?
                if bandIndex ~= numel(OldInt)
                    % If not, then jump to the next highest band
                    CurrentInt = circshift(OldInt, 1, 2);
                    NextSweepState = 'Up';
                    
                    % varargout{1} = SweepState; % No longer needed
                elseif bandIndex == numel(OldInt)
                    % If at the upper limit, then jump to next lowest band
                    % and change sweep state to 'Down'
                    CurrentInt = circshift(OldInt, -1, 2);
                    NextSweepState = 'Down';
                    
                    % varargout{1} = NextSweepState; % No longer needed
                end
                
            case 'Down'                
                % Is the occupied band at the lower limit?
                if bandIndex ~= 1
                    % If not, then jump to the next lowest band
                    CurrentInt = circshift(OldInt, -1, 2);
                    NextSweepState = 'Down';
                    
                    % varargout{1} = SweepState; % No longer needed
                elseif bandIndex == 1
                    % If at the lower limit, then jump to next highest band
                    % and change sweep state to 'Up'
                    CurrentInt = circshift(OldInt, 1, 2);
                    NextSweepState = 'Up';
                    
                    % varargout{1} = NextSweepState; % No longer needed
                end
        end
        
        % Set output arguments
        varargout{1} = CurrentInt;
        varargout{2} = NextSweepState;
        
    % Frequency hopping linear sweep, wraparound    
    % Frequency hopper, sawtooth frequency sweep % [CurrentInt, NextSweepState] = UpdateInterference(InterferenceBehavior, OldInt, SweepState)
    case 'FH-SAWTOOTH'
        % Set variables from input arguments
        OldInt = varargin{1};
        SweepState = varargin{2};
        
        % SweepState = varargin{1}; % No longer neded
        switch SweepState
            case 'Up'
                CurrentInt = circshift(OldInt, 1, 2);
            case 'Down'
                CurrentInt = circshift(OldInt, -1, 2);
        end
        NextSweepState = SweepState;
        
        % Set output arguments
        varargout{1} = CurrentInt;
        varargout{2} = NextSweepState;
        % varargout{1} = SweepState; % No longer needed
        
    % Frequency hopping, following user-defined pattern
    % Frequency hopper, following a user-defined pattern or pseudorandom
    % sequence % [CurrentInt, nextIndex] = UpdateInterference(InterferenceBehavior, lastIndex, pattern, NumBands)
    case {'FH-PATTERN', 'FH-PSEUDO'}
        % The numbers in pattern describe user-defined positions of where
        % the interference should be; i.e. 4 = [0 1 0 0 ], 9 = [1 0 0 1],
        % etc; If the end of the pattern is reached, reset the index

        % Get index from last iteration
%         lastIndex = varargin{1}; % No longer needed
%         
%         pattern = varargin{2}; % No longer needed
        
        % Set variables from input arguments
        lastIndex = varargin{1};
        pattern = varargin{2};
        NumBands = varargin{3};
        
        % If the last index is equal to number of elements in patter, reset
        % it to 1 (to prevent out-of-bound error), otherwise, increment
        % index by 1
        % if lastIndex == numel(pattern)
        %     nextIndex = 1;    
        % elseif lastIndex < numel(pattern)
        %     nextIndex = lastIndex + 1;
        % end

        % Determine next index; increment by one, or reset if nextIndex
        % reaches the number of elements in pattern (to prevent
        % out-of-bounds error), and output the next index
        nextIndex = mod(lastIndex, numel(pattern))+1;
        % varargout{1} = nextIndex; % No longer needed
        
        % Set the interference
        CurrentInt = de2bi(pattern(nextIndex), NumBands, 'left-msb');
        
        % Set output arguments
        varargout{1} = CurrentInt;
        varargout{2} = nextIndex;
                
    % Bursty interferer, stays in bans for a random length of time given by
    % an exponential distribution, changes selected bands randomly %
    % [CurrentInt, BandDuration, ElapsedTimeInBand] =
    % UpdateInterference(InterferenceBehavior, OldInt, BandDuration,
    % ElapsedTimeInBand, NumBands)
    case 'BURSTY'
        % Set variables from input arguments
        OldInt = varargin{1};
        BandDuration = varargin{2};
        ElapsedTimeInBand = varargin{3};
        NumBands = varargin{4};
        
        if ElapsedTimeInBand >= BandDuration
            ElapsedTimeInBand = 0;
            
            % CurrentInt = de2bi(randi(2^(NumBands)-1), NumBands, 'left-msb');
            CurrentInt = de2bi(2.^(randi([1 NumBands])-1), NumBands, 'left-msb');
            BandDuration = ceil(exprnd(5, 1, 1));
            
            % varargout{1} = de2bi(randi(15), 4, 'left-msb'); % No longer
            % needed
            % varargout{2} = ceil(exprnd(10, 1, 1)); % No longer needed
        elseif ElapsedTimeInBand < BandDuration
            ElapsedTimeInBand = ElapsedTimeInBand + 1;
            
            CurrentInt = OldInt;
            % varargout{1} = CurrentInt; % No longer needed
            % varargout{2} = BandDuration; % No longer needed
        end
        
        % Set output arguments
        varargout{1} = CurrentInt;
        varargout{2} = BandDuration;
        varargout{3} = ElapsedTimeInBand;
        
    case 'JAMMER'
        Action = varargin{1};
        CurrentInt = Action;
        varargout{1} = CurrentInt;
        
    case 'DIRECTION-DEPENDENT-CONST'
        CurrentPosState = varargin{1};
        IntfPosStates = varargin{2};
        IntfMask = varargin{3};
        
        if ismember(CurrentPosState, IntfPosStates)
            CurrentInt = IntfMask;
        else
            CurrentInt = zeros(size(IntfMask));
        end
        
        varargout{1} = CurrentInt;

        %Not needed anymore
%         numpos = size(TargetPositions,1);
%         tmp = sum(abs(TargetPositionsXY - repmat(interference_cells(5,:), numpos,1))');
%         [a,b]=min(tmp);
%         position_number = b;

        % Get the state numbers for the position cells that may have
        % interference
        % If present state number == state that has interference, set
        % interference = 1, otherwise interference = 0
        
    case 'DIRECTION-DEPENDENT-INTER'        
        OldInt = varargin{1};
        IntProb = varargin{2};
        CurrentPosState = varargin{3};
        IntfPosStates = varargin{4};
        IntfMask = varargin{5};
        
        tmp = rand;
        if (tmp < IntProb) && ismember(CurrentPosState, IntfPosStates)
            CurrentInt = IntfMask;
        else
            % CurrentInt = [0 0 0 0]; % Delete this line
            CurrentInt = zeros(size(IntfMask));
        end
        
        varargout{1} = CurrentInt;
        
end
end

        