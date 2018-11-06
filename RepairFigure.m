function RepairFigure(FigureName)
% Repair figures
% This will  fix the plots for the 2018 Radar Conference paper, to put
% bandwidth on the scale of 20-100 MHz (from 100-500 MHz). Also normalizes
% the rewards by 100, and changes the y-label and legend to reflect that.

H = openfig(FigureName);
axesObjs = get(H, 'Children');
dataObjs = get(axesObjs, 'Children');

oldBW_ydata = dataObjs{5}(1).YData;
oldRewards_ydata = dataObjs{5}(2).YData;

newBW_ydata = oldBW_ydata/5;
newRewards_ydata = oldRewards_ydata*10/100;

dataObjs{5}(1).YData = newBW_ydata;
dataObjs{5}(2).YData = newRewards_ydata;

J = findall(H, 'type', 'axes');
old_ylabel = J(3).YLabel.String;
new_ylabel = old_ylabel;
new_ylabel{1} = strrep(old_ylabel{1}, 'Rewards (x10)', 'Rewards (x100)');
J(3).YLabel.String = new_ylabel;
legend(J(3), 'Rewards (x100)','Bandwidth (MHz)','SINR (dB)','Range (km)', 'Location', 'SouthOutside', 'Orientation', 'Horizontal');
axis(J(3), 'auto');
J(3).YTick = [0:60:120];
J(3).YTickLabel = arrayfun(@num2str, J(3).YTick, 'UniformOutput', false);
J(3).YLim = [0 120];

savefig(FigureName);
end