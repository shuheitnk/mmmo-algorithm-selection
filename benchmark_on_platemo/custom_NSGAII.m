classdef custom_NSGAII < MyALGORITHM
% <2002> <multi> <real/integer/label/binary/permutation> <constrained/none>
% Nondominated sorting genetic algorithm II

%------------------------------- Reference --------------------------------
% K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A fast and elitist
% multiobjective genetic algorithm: NSGA-II. IEEE Transactions on
% Evolutionary Computation, 2002, 6(2): 182-197.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)

            thisFilePath = fileparts(mfilename('fullpath'));
            targetPath = fullfile(thisFilePath, '..', 'PlatEMO', 'PlatEMO', 'Algorithms', 'Multi-objective optimization', 'NSGA-II');
            addpath(targetPath, '-begin');

            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);

                % Customized the implementation to allow retrieval of final-generation individuals.
                % Since NSGA-II does not use an archive, the final population is recovered instead.
                Algorithm.FinalPop = Population; % <-- Added
            end
        end
    end
end