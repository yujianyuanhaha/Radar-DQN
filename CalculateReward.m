function Reward = CalculateReward(SINR, CurrentAction, NumBands)

Reward = 0;

% switch State(4)
%     case 1
%         Reward = Reward - 25;
%     case 2
%         Reward = Reward + 1;
%     case 3
%         Reward = Reward + 2;
%     case 4
%         Reward = Reward + 3;
%     case 5
%         Reward = Reward + 4;
%     case 6
%         Reward = Reward + 6;
%     case 7
%         Reward = Reward + 8;
%     case 8
%         Reward = Reward + 10;
% end
NegSINRPenalty = -10*(NumBands - 1) +5;

if SINR <= 0
    % Reward = Reward - 35; % Delete this line
    Reward = Reward + NegSINRPenalty;
elseif (SINR > 0) && (SINR < 2)
    Reward = Reward + 1;
elseif (SINR >= 2) && (SINR < 5)
    Reward = Reward + 2;
elseif (SINR >= 5) && (SINR < 8)
    Reward = Reward + 3;
elseif (SINR >= 8) && (SINR < 11)
    Reward = Reward + 4;
elseif (SINR >= 11) && (SINR < 14)
    Reward = Reward + 5;    
elseif (SINR >= 14) && (SINR < 17)
    Reward = Reward + 6;
elseif (SINR >= 17) && (SINR < 20)
    Reward = Reward + 8;    
elseif SINR >= 20
    Reward = Reward + 10;    
end

tmp = sum(CurrentAction);

% switch tmp
%     case 1
%         Reward = Reward;
%     case 2
%         Reward = Reward + 10;
%     case 3 
%         Reward = Reward + 20;
%     case 4 
%         Reward = Reward + 30;
% end
Reward = Reward + 10*(tmp-1);
end