% This script runs a MDP-based cognitive radar simulation
clear all
%close all

RadarBehavior = 'OPTI'; % 'CONS', 'OPTI','RAND'
                        % constant action, optimal (ML), random
TimeSteps = 150;   % seconds
TimeInterval = 10; % seconds
Pt = 1000;  %Watts
G = 500;
lambda = 3*10^8/(2*10^9); % meters
sigma = 0.1;    % RCS
PRF = 2000;
T_dwell = 20;   % pulses
PulseWidth =1e-5;   % seconds
BandSize = 1e8; %Hz
% receiver characteristics
NF = 1; % Noise Figure
Boltzmann = 1.38064852* 10^(-23);
Ts = 295; % Kelvin

DiscountFactor = 0.9;

NumTrainingRuns = 500;

InterferenceBehavior = 'CONST'; % 'CONST', 'AVOID', 'INTER'

TB = 1/PulseWidth/PRF*T_dwell; % time bandwidth product

InitialPos = [-70, 150];
velocity = [0.050,-0.100];
CurrentInt = [1, 0, 0, 0];

IntPower = 1*10^(-10); % watts

State = [0, 0, 0, 0];

% first element represents target position in x, y
% second element represents target velocity
% third element represents interference conditions
% fourth element represents SINR

TargetPositions = [-70, 150; -50, 150; -30, 150; ...
    -10, 150; 10, 150; 30, 150; 50, 150; 70, 150;...
    -50, 130;-30, 130; -10, 130; 10, 130; 30, 130; 50, 130;...
    -30, 110; -10, 110; 10, 110; 30, 110; ...
    -30, 90; -10, 90; 10, 90; 30, 90; ...
    -10, 70; 10,70; -10, 50; 10,50; -10, 30; 10,30;...
    -10, 10; 10,10; ];  % km
%TargetVelocities = [0, -0.100; 0.050, -0.100; 0.100, 0; -0.050,-0.100]; % km
TargetVelocities = [0.050, -0.100];
SINRs = [0, 2, 5, 8, 11, 14, 17, 20]';
%InterferenceState = [ 0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1;...
%   0 0 1 1; 1 1 0 0 ; 1 1 1 0;];
InterferenceState = [ 0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0];

Actions = [ 1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1;...
    0 0 1 1; 1 1 0 0; 0 1 1 0; 0 1 1 1;...
    1 1 1 0; 1 1 1 1 ];

CurrentAction = Actions(10,:);
CurrentActionNumber = 10;

NumVelocities = size(TargetVelocities,1);
NumSINRs = size(SINRs, 1);
NumInt = size(InterferenceState,1);
NumPos = size(TargetPositions,1);
NumStates = NumVelocities*NumSINRs*NumInt*NumPos;
NumActions = size(Actions, 1);

ActionHist = zeros(NumActions,1);
TrainingHist = zeros(NumActions,1);
StateHist = zeros(NumStates, 1);

policy = 10*ones(NumStates,1);

Mdp_path = 'C:\Users\MikeandAndrea\Google Drive\My Documents\Matlab Library\MDP\fsroot\MDPtoolbox'
addpath(Mdp_path)


% The estimated state transition matrix.
P = zeros(NumStates, NumStates, NumActions);
P_hist = P;
for i=1:NumActions
    P_hist(:,1,i) = ones(NumStates,1,1);
end
R = zeros(NumStates, NumActions);
RewardCount = R;

t = 0:TimeInterval:(TimeSteps-1)*TimeInterval;


Range = norm(InitialPos);

Pr = Pt*G*G*lambda^2*sigma*TB/(4*pi)^3/(Range*100)^4;
I = sum(CurrentAction.*CurrentInt)*IntPower;
N = Boltzmann*Ts*NF*sum(CurrentAction)*BandSize;

SINR = Pr/(I+N);
SINRdB = 10*log10(SINR);

[State, StateNumber] = MapState(InitialPos, TargetPositions, ...
    velocity, TargetVelocities, SINRdB, SINRs, ...
    CurrentInt, InterferenceState);

position = InitialPos;

%fignum = figure
%plot(TargetPositions(:,1), TargetPositions(:,2),'o')
%hold on

SINR_hist = zeros(NumTrainingRuns+1, length(t));
Range_hist = zeros(NumTrainingRuns+1, length(t));
R_hist = zeros(NumTrainingRuns+1, length(t));
Bandwidth = zeros(1, length(t));

for kk = 1:(NumTrainingRuns+1)
    
    CumulativeReward = 0;
    position = InitialPos;
    
    for i=0:length(t)-1
        kk
        i
        
        % update range and position
        position = [position(1)+velocity(1)*TimeInterval, position(2)+velocity(2)*TimeInterval];
        Range = norm(position);
        
        switch RadarBehavior 
            case 'OPTI'
                if kk<=NumTrainingRuns
                    CurrentActionNumber = ceil(rand*NumActions);
                else
                    CurrentActionNumber = policy(StateNumber);
                end
            case 'CONS'
                CurrentActionNumber = CurrentActionNumber;
            case 'RAND'
                CurrentActionNumber = ceil(rand*NumActions);
        end
        CurrentAction = Actions(CurrentActionNumber,:);
        Bandwidth(i+1) = 1e8*sum(CurrentAction);
        
        % update Interference
        CurrentInt = UpdateInterference(CurrentInt, ...
            InterferenceBehavior, CurrentAction, ...
            InterferenceState, 0.25);
        
        
        % update SINR
        Pr = Pt*G*G*lambda^2*sigma*TB/(4*pi)^3/(Range*100)^4;
        I = sum(CurrentAction.*CurrentInt)*IntPower;
        N = Boltzmann*Ts*NF*sum(CurrentAction)*BandSize;
        
        SINR = Pr/(I+N);
        SINRdB = 10*log10(SINR);
        SINR_hist(kk,i+1) = SINRdB;
        Range_hist(kk,i+1) = Range;
        
        OldState = State;
        OldStateNumber = StateNumber;
        
        [State, StateNumber] = MapState(position, TargetPositions, ...
            velocity, TargetVelocities, SINRdB, SINRs, ...
            CurrentInt, InterferenceState);
        
        StateHist(StateNumber) = StateHist(StateNumber) + 1;
        
        if kk <= NumTrainingRuns 
            TrainingHist(CurrentActionNumber) = TrainingHist(CurrentActionNumber) + 1;
        else
            ActionHist(CurrentActionNumber) = ActionHist(CurrentActionNumber) + 1;
        end
        
        P_hist(OldStateNumber,StateNumber, CurrentActionNumber) = ...
            P_hist(OldStateNumber, StateNumber, CurrentActionNumber) +1;
        %for ii=1:NumStates
        %    for k=1:NumActions
        %       P(ii,:,k) = P_hist(ii, :, k)/sum(P_hist(ii, :, k));
                P(OldStateNumber,:,CurrentActionNumber) = P_hist(OldStateNumber, :, CurrentActionNumber)/sum(P_hist(OldStateNumber, :, CurrentActionNumber));
        %    end
        %end
        
        CurrentReward = CalculateReward(State, CurrentAction );
        
        
        RewardCount(StateNumber, CurrentActionNumber) = RewardCount(StateNumber, CurrentActionNumber) + 1;
        R(StateNumber, CurrentActionNumber) = ...
            R(StateNumber, CurrentActionNumber)*(RewardCount(StateNumber, CurrentActionNumber)-1)/RewardCount(StateNumber, CurrentActionNumber) ...
            + CurrentReward/RewardCount(StateNumber, CurrentActionNumber);
        CumulativeReward = CumulativeReward + CurrentReward;
        
        R_hist(kk,i+1) = CumulativeReward;
        
        % determine the new policy
        if kk>NumTrainingRuns && (sum(RadarBehavior=='OPTI'))
            [V,policy]=mdp_policy_iteration(P,R,DiscountFactor);
        end
        
        %figure(fignum)
        %plot(position(1), position(2), 'x')
        %plot(TargetPositions(State(1),1), TargetPositions(State(1),2),'ro')
        
    end
    
    if kk > NumTrainingRuns
        figure
        plot(t, SINR_hist(kk,:));
        hold on
        plot(t, Range_hist(kk,:),'k')
        plot(t, R_hist(kk,:), 'r')
        plot(t,Bandwidth/1e6,'g')
        legend('SINR(dB)','Range (km)','Rewards')
    end
end