function ExportToPDF(FigureName)

f = openfig(FigureName);
set(f, 'WindowStyle', 'Normal', 'Units', 'Inches', 'Position', [0 0 8 6], 'PaperPositionMode', 'Manual', ...
    'PaperUnits', 'Inches', 'PaperPosition', [0 0 8 6]);

trajPlotHandle = findall(gcf, 'Type', 'Line', 'DisplayName', 'Target Trajectory');
if numel(trajPlotHandle) > 0
    line2arrow(trajPlotHandle);
end

savefig(FigureName);
figfilename = strrep(FigureName, '.fig', '');
pdfname = sprintf('%s.pdf', figfilename);
saveas(f, figfilename, 'svg');
system(sprintf('inkscape -f %s.svg -A %s --export-area-drawing', figfilename, pdfname));
system(sprintf('pdfcrop %s %s', pdfname, pdfname));

close(f);
end

