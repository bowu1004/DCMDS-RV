function [ stress ] = Stress_VisEvaluate_BW( X,Y,plot_signal )
%This main function 
% is to calculate the Kruskal Stress parameter, which is used to evaluate
% the effect of projection/mapping.
% 
% Detailed explaination is explicited in Kruskal's publication.
% Parameters:
%   ---- X=[np,ndim],points before mapping
%   ---- Y=[np,ndim],points after mapping


%
% Create a dissimilarity vector, which is distance in HIGH-dim space
%
% === X=[np,ndim]
dissimilarities = pdist(X);

% ============ Here goes the MDS techniques ===========================

%
% Calculate the distance vector, which is distance in LOW-dim space
%
% === Y=[np,ndim]
distances = pdist(Y);
%
% fit the disparities from Distances and distances in hd and 2d space
%
disparities2 = lsqisotonic_BW(dissimilarities, distances);
%
% sort it ascendingly
%
[val_dissim,ord]=sort(dissimilarities);
distance_2=distances(ord);
disparities_2=disparities2(ord);
if plot_signal
    figure,plot(val_dissim,distance_2,'bo',val_dissim,disparities_2,'r.');
    xlabel('Dissimilarities'); ylabel('Distances/Disparities')
    legend({'Distances' 'Disparities'},'Location','NW');
    title('J.B. Kruskal Stress');
end;

% for i=1:length(dissimilarities)
for i=length(dissimilarities)
    a=0;b=0;
    for j=1:i
        a=a+(distance_2(j)-disparities_2(j))^2;
        b=b+distance_2(j)^2;       
    end;
    c=sqrt(a/b);
%     disp(['stess ' num2str(i) ' is: ' num2str(c)]);   
end;
%
% usually the last stress factor is the smallest
%

% disp(['stess is: ' num2str(c)]); 
stress=c;




% =========================================================================
function yhat = lsqisotonic_BW(x,y,w)
%LSQISOTONIC Isotonic least squares. -BW
%   YHAT = LSQISOTONIC(X,Y) returns a vector of values that minimize the
%   sum of squares (Y - YHAT).^2 under the monotonicity constraint that
%   X(I) > X(J) => YHAT(I) >= YHAT(J), i.e., the values in YHAT are
%   monotonically non-decreasing with respect to X (sometimes referred
%   to as "weak monotonicity").  LSQISOTONIC uses the "pool adjacent
%   violators" algorithm.
%
%   If X(I) == X(J), then YHAT(I) may be <, ==, or > YHAT(J) (sometimes
%   referred to as the "primary approach").  If ties do occur in X, a plot
%   of YHAT vs. X may appear to be non-monotonic at those points.  In fact,
%   the above monotonicity constraint is not violated, and a reordering
%   within each group of ties, by ascending YHAT, will produce the desired
%   appearance in the plot.
%
%   YHAT = LSQISOTONIC(X,Y,W) performs weighted isotonic regression using
%   the non-negative weights in W.

%   Copyright 2003-2006 The MathWorks, Inc.


%   References:
%      [1] Kruskal, J.B. (1964) "Nonmetric multidimensional scaling: a
%          numerical method", Psychometrika 29:115-129.
%      [2] Cox, R.F. and Cox, M.A.A. (1994) Multidimensional Scaling,
%          Chapman&Hall.

n = numel(x);
if nargin<3
    yclass = superiorfloat(x,y);
else
    yclass = superiorfloat(x,y,w);
end

% Sort points ascending in x, break ties with y.
[xyord,ord] = sortrows([x(:) y(:)]); iord(ord) = 1:n;
xyord = double(xyord);

% Initialize fitted values to the given values.
yhat = xyord(:,2);

block = 1:n;
if (nargin == 3) && ~isempty(w)
    w = double(w(:)); w = w(ord); % reorder w as a column

    % Merge zero-weight points with preceding pos-weighted point (or
    % with the following pos-weighted point if at start).
    posWgts = (w > 0);
    if any(~posWgts)
        idx = cumsum(posWgts); idx(idx == 0) = 1;
        w = w(posWgts);
        yhat = yhat(posWgts);
        block = idx(block);
    end

else
    w = ones(size(yhat));
end

while true
    % If all blocks are monotonic, then we're done.
    diffs = diff(yhat);
    if all(diffs >= 0), break; end

    % Otherwise, merge blocks of non-increasing fitted values, and set the
    % fitted value within each block equal to a constant, the weighted mean
    % of values in that block.
    idx = cumsum([1; (diffs>0)]);
    sumyhat = accumarray(idx,w.*yhat);
    w = accumarray(idx,w);
    yhat = sumyhat ./ w;
    block = idx(block);
end

% Broadcast merged blocks out to original points, and put back in the
% original order and shape.
yhat = yhat(block);
yhat = reshape(yhat(iord), size(y));
if isequal(yclass,'single')
    yhat = single(yhat);
end
% =========================================================================

