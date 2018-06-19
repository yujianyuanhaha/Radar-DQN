function [State, StateNumber] = MapState( position,         TargetPositions, velocity, ...
                                          TargetVelocities, sinr,            SINRs, ...
                                          CurrentInt,       InterferenceState)
        
% function [State, StateNumber] = MapState(position, TargetPositions, ...
%            velocity, TargetVelocities, sinr, SINRs, ...
%           CurrentInt, InterferenceState)

numpos          = size(TargetPositions,1);
tmp             = sum(abs(TargetPositions - repmat(position, numpos,1))');
[a,b]           = min(tmp);
position_number = b;

numvel          = size(TargetVelocities,1);
tmp             = sum(abs(TargetVelocities - repmat(velocity, numvel,1))');
[a,b]           = min(tmp);
velocity_number = b;

numIntfCombos   = size(InterferenceState,1);
numStatesInMem  = size(InterferenceState,3);
numint          = numIntfCombos ^numStatesInMem;
% tmp = sum(abs(InterferenceState - repmat(CurrentInt, numint,1))');
tmp             = sum(permute(abs(InterferenceState - repmat(CurrentInt, numIntfCombos, 1)), [2, 1, 3]));
[a,b]           = min(tmp);
% interference_number = b;
c               = transpose(b(:));
d               = c - [ones(1, numStatesInMem-1), 0];
e               = [numStatesInMem-1:-1:0];
f               = numIntfCombos.^e;
g               = sum(d.*f);
interference_number = g;
% interference_number = (b(:,:,1)-1)*size(InterferenceState,1) + b(:,:,2);


numsinr     = size(SINRs,1);
tmp         = (abs(SINRs - repmat(sinr, numsinr,1)));
[a,b]       = min(tmp);
sinr_number = b;

State = [position_number, velocity_number, interference_number, sinr_number];
StateNumber = (position_number-1)*(numvel*numint*numsinr) + ...
              (velocity_number-1)*numint*numsinr + ...
              (interference_number-1)*numsinr + sinr_number;