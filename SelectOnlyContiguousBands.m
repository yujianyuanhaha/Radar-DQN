function [AllContiguousBands] = SelectOnlyContiguousBands(BandSequences)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

NumContigBands = arrayfun(@(n) NumContiguousOnes(BandSequences(n, :)), 1:size(BandSequences, 1))';
RowSum = sum(BandSequences, 2);
IndxContiguousSequences = (NumContigBands == RowSum);
AllContiguousBands = BandSequences(IndxContiguousSequences, :);
% ind = 6; x(ind,:), max(diff([0 (find(~(x(ind,:) > 0))) numel(x(ind,:)) + 1]) - 1), sum(x(ind,:))
end