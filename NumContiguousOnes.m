function [NumOnes] = NumContiguousOnes(Sequence)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

NumOnes = max(diff([0 (find(~(Sequence > 0))) numel(Sequence) + 1]) - 1);
end