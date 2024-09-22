function index  = KrigingSelection(PopObj,MSE,RP,V)
    NP = size(PopObj,1);
    NV = size(RP,1);
    sigma = sqrt(MSE);
    DisPop = CalDis(PopObj,RP,V);
    [~,associate] = min(DisPop,[],2);
    % Select the next population
    Next = zeros(1,NV);
    for i = unique(associate)'
        current  = find(associate==i);
        PopObj_temp = PopObj - repmat(RP(i,:),NP,1);
        dis       =  sum(PopObj_temp(current,:) .* V,2);
        mse       =  sqrt(sum((sigma(current,:) .* V).^2,2));
        de =  dis + 2 * mse;
        % Select the one with the minimum projection distance
        [~,best] = min(de);
        Next(i)  = current(best);
    end   
    % Population for next generation
    index = Next(Next~=0);
end
% Calculate the distance from each candidate solution to each reference line
function Dis = CalDis(PopObj,RP,V)
    NP = size(PopObj,1);
    NV = size(RP,1);
    Dis = zeros(NP,NV);
    for i = 1:NP
        de = sqrt(sum((repmat(PopObj(i,:),NV,1) - RP).^2,2));
        cosine = sum((repmat(PopObj(i,:),NV,1) - RP).* repmat(V,NV,1),2) ./ de;
        Dis(i,:) = de .* sqrt(1 - cosine.^2);
    end
end