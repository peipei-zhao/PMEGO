classdef PMEGO < ALGORITHM
% <multi> <real/integer> <expensive>
% PMEGO
% r ---     --- Set of preferred point
% alpha ---   0.1 --- The absolute size of the preferred PF region
% gmax  --- 20 --- Number of generations before updating Kriging models
% mu    ---  5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Peipei Zhao

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [r,alpha,gmax,mu] = Algorithm.ParameterSet(zeros(1,Problem.M)+0.2,0.1,20,5);            
           %% Initialization
            [R0,~]  = UniformPoint(Problem.N,Problem.M);
            NI     = 100;
            Plhs   = UniformPoint(NI,Problem.D,'Latin');
            TS     = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1) .* Plhs + repmat(Problem.lower,NI,1));
            lob    = 1e-5.*ones(1,Problem.D);
            upb    = 100.*ones(1,Problem.D);
            THETA  = 5.*ones(Problem.M,Problem.D);
            Kmodel = cell(1,Problem.M);
           %% Optimization
            PopDec = TS.decs;
            while Algorithm.NotTerminated(TS)             
                %% Train a GP model for each objective
                TSDec = TS.decs;
                TSObj = TS.objs;
                for i = 1 : Problem.M
                    [mS, mY]   = dsmerge(TSDec, TSObj(:,i));
                    dmodel     = dacefit(mS,mY,'regpoly0','corrgauss',THETA(i,:),lob,upb);
                    Kmodel{i}  = dmodel;
                    THETA(i,:) = dmodel.theta;
                end
                %% Generate a series of new reference points derived from the orignal reference point
                Zmin = min(TSObj,[],1);
                Zmax = max(TSObj,[],1);
                R = R0.*repmat(Zmax - Zmin ,size(R0,1),1);    
                RP = [];
                for i = 1 : size(r,1)
                    RP = [RP;r(i,:)+ alpha * R];
                end
                %% Estimate the unit direction vector
                V = (Zmax - Zmin) ./ sqrt(sum((Zmax - Zmin).^2,2));
                %% Model-based optimization
                g = 1;
                while g <= gmax
                    drawnow();
                    OffDec = OperatorGA(Problem,PopDec);
                    PopDec = [PopDec;OffDec];
                    N  = size(PopDec,1);
                    PopObj = zeros(N,Problem.M);
                    MSE    = zeros(N,Problem.M);
                    for i = 1: N
                        for j = 1 : Problem.M
                            [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),Kmodel{j});
                        end
                    end
                    index  = KrigingSelection(PopObj,MSE,RP,V);
                    PopDec = PopDec(index,:);
                    g = g + 1;
                end              
                %% Surrogate management
                N = size(PopDec,1);
                M = Problem.M;
                PopObj = zeros(N,M);
                MSE    = zeros(N,M);
                for i = 1: N
                     for j = 1 : M
                            [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),Kmodel{j});
                     end
                end          
                %% Multipoint infill criteria
                % Monte Carlo calculation
                num_simluation_point = 3000;
                rand_sample = zeros(num_simluation_point,M,size(PopObj,1));         
                for i = 1:N
                    rand_sample(:,:,i) = mvnrnd(PopObj(i,:),diag(MSE(i,:)),num_simluation_point); 
                end
                NCluster  = min(mu,size(RP,1));
                [IDX,~]   = kmeans(RP,NCluster);    
                Popreal = zeros(NCluster,size(PopDec,2));
                ASF_EI = zeros(NCluster,size(PopDec,1));
                for i = 1: NCluster
                    group = find(IDX == i);
                    vp = RP(group(randi(length(group))),:);
                    minASF = min(max((TSObj - repmat(vp,size(TSObj,1),1))./V,[],2));
                    gg = ones(num_simluation_point,M,N).* vp;
                    VV = ones(num_simluation_point,M,N).* V;
                    ASF_EI(i,:) = mean(max(repmat(minASF,num_simluation_point,N) - squeeze(max((rand_sample - gg) ./ VV,[],2)),0));
                    [~,maxind] = max(ASF_EI(i,:));                   
                    Popreal(i,:) = PopDec(maxind,:);
                end  
                % If no better infill sample can be found in one iteration, then randomly find one to improve efficiency
                TS_ASF = max((TSObj - repmat(r(randi(size(r,1)),:),size(TSObj,1),1))./V,[],2);  
                [~,index] = sort(TS_ASF);
                Popreal = unique(Popreal,'rows');
                if sum(sum(ASF_EI)) == 0 || (size(Popreal,1) == 1 && ismember(Popreal,TSDec,'rows'))
                        Popreal =  OperatorGAhalf(Problem,[TSDec(index(1),:);TSDec(index(1),:)]);
                end
                % Delete duplicated solutions  
                for i = 1 : size(Popreal,1)
                    TSDec    = [TS.decs;Popreal(i,:)];
                    [~,index] = unique(TSDec,'rows');
                    if length(index) == size(TSDec,1)
                        PopNew = Problem.Evaluation(Popreal(i,:));
                        TS     = [TS,PopNew];
                    end
                end
                %% The next population 
                Parent = TSDec;
                PopDec =  [Parent(TournamentSelection(2,size(Parent,1),TS_ASF),:);PopDec];               
                PopDec = unique(PopDec,'rows');
            end
        end
    end
end