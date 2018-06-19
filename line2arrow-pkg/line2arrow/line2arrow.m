function varargout = line2arrow(h, varargin);
%LINE2ARROW Convert line to arrow
%
% line2arrow(h);
% line2arrow(h, param1, val1, ...)
% ha = line2arrow(...)
%
% This function adds annotation arrows to the end of a line in a 2D plot.
%
% Arrows inherit line properties from the line they are attached to, unless
% otherwise specified by the user.  You can also resync the line and arrow
% by clicking on the line.  Resyncing matches the arrow color, line width,
% and line style of the arrows to the lines, and repositions the arrows
% (because annotation arrows use figure units rather than axis units, the
% match between the arrow location and line location will be thrown off
% during some resizing of figures/axes, for example with manual aspect
% ratios).       
%
% Passing a line handle that is already associated with arrows provides the
% same syncing action as clicking on the line, but can also be used to
% manually change some properties.  See example for details.
%
% Input arguments:
%
%   h:          handle(s) of line(s)
%
% Optional arguments (passed as parameter/value pairs):
%
%   Color:      color of arrow [inherited from the line object]
%
%   HeadLength: length of arrowhead, in pixels [10]
%
%   HeadWidth:  width of arrowhead, in pixels [10]
%
%   HeadStyle:  shape of arrowhead.  See Annotation Arrow Properties in
%               Matlab documentation for list. ['vback2'] 
%
%   LineStyle:  style of arrow line [inherited from the line object]
%
%   LineWidth:  width of arrow line, in pixels [inherited from the line
%               object]
%
% Output arguments:
%
%   ha:         handle(s) to annotation objects.  Can also be accessed by
%               getappdata(h, 'arrow').
%
% Example:
%
%   x = -pi:.1:pi;
%   y = sin(x);
%   h = plot(x,y);
%   line2arrow(h);
%   
%   % To demonstrate syncing
%
%   axis equal; % Now click on line
%
%   % More syncing
%
%   set(h, 'color', 'red', 'linewidth', 2); % Should sync automatically
%
%   % Manual syncing
%
%   set(gca, 'dataaspectratio', [2 1 1]);
%   line2arrow(h);
%
%   % Manual resetting
%
%   line2arrow(h, 'color', 'r', 'headwidth', 5); % Note that headwidth will
%                                                % be inherited on future
%                                                % syncs but color will
%                                                % return to matching line

% Copyright 2011-2014 Kelly Kearney

% Check input

if ~all(ishandle(h(:))) || ~all(strcmp(get(h(:), 'type'), 'line'))
    error('Input h must be array of line handles');
end

% Apply arrow and add callback function

for ih = 1:numel(h)
    addarrow(h(ih), [], varargin);
    addlistener(h(ih), 'Color',     'PostSet', @addarrow);
    addlistener(h(ih), 'LineStyle', 'PostSet', @addarrow);
    addlistener(h(ih), 'LineWidth', 'PostSet', @addarrow);

    set(h(ih), 'buttondownfcn', @addarrow)
end

if nargout == 1
    if verLessThan('matlab', '8.4.0')
        ha = zeros(size(h));
    else
        ha = gobjects(size(h));
    end
    for ih = 1:numel(h)
    	ha(ih) = getappdata(h(ih), 'arrow');
    end
    varargout{1} = ha;
end

%-------------------------
% Apply arrow to a line
%-------------------------

function addarrow(h, ev, varargin)

% Find the line object (I know this way of setting up the if/then is silly,
% but with the 2014b changes I'm losing track of what's what, so I'd rather
% be explicit)

if verLessThan('matlab', '8.4.0')
    if ishandle(h) && isempty(ev) && strcmp(get(h,'type'), 'line')
        % initial call or button-down, line handle is already h
    elseif strcmp(class(h), 'schema.prop') %
        h = ancestor(get(ev, 'AffectedObject'), 'line');
    end    
else
    if ishandle(h) && strcmp(h.Type, 'line')
        % initial call and button-down
    else
        h = ev.AffectedObject;
    end
end

npv = length(varargin);

% Normalize figure

hfig = ancestor(h, 'figure');

figunit = get(hfig, 'units');
set(hfig, 'units', 'normalized');

% Create/update arrow

x = get(h, 'xdata');
y = get(h, 'ydata');

[xa, ya] = axescoord2figurecoord(x(end-1:end),y(end-1:end));

arrowexists = isappdata(h, 'arrow');
if arrowexists
    harrow = getappdata(h, 'arrow');
end

% Default options

if arrowexists
    Opt.HeadLength = get(harrow, 'headlength');
    Opt.HeadWidth  = get(harrow, 'headwidth');
    Opt.HeadStyle  = get(harrow, 'headstyle');
    Opt.Color      = get(h, 'color');
    Opt.LineStyle  = get(h, 'linestyle');
    Opt.LineWidth  = get(h, 'linewidth');
else
    Opt.HeadLength = 10;
    Opt.HeadWidth  = 10;
    Opt.HeadStyle  = 'vback2';
    Opt.Color      = get(h, 'color');
    Opt.LineStyle  = get(h, 'linestyle');
    Opt.LineWidth  = get(h, 'linewidth');
end

% Override with user options

if npv > 0
    Opt = parsepv(Opt, varargin{:});
end


if arrowexists
    
    set(harrow, 'x', xa, 'y', ya, ...
             'color', Opt.Color, ...
             'headlength', Opt.HeadLength, ...
             'headwidth', Opt.HeadWidth, ...
             'headstyle', Opt.HeadStyle, ...
             'LineStyle', Opt.LineStyle, ...
             'LineWidth', Opt.LineWidth);
    
else

    harrow = annotation('arrow', xa, ya, ...
             'color', Opt.Color, ...
             'headlength', Opt.HeadLength, ...
             'headwidth', Opt.HeadWidth, ...
             'headstyle', Opt.HeadStyle, ...
             'LineStyle', Opt.LineStyle, ...
             'LineWidth', Opt.LineWidth);
    setappdata(h, 'arrow', harrow);
end

% Return figure units to their starting value

set(hfig, 'units', figunit);














        



