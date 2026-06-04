classdef custom_MMOEADC < MyALGORITHM

    methods
        function main(Algorithm,Problem)

            thisFilePath = fileparts(mfilename('fullpath'));
            targetPath = fullfile(thisFilePath, '..', 'ComparativeStudyofMMOP', ...
                'PleaEMOCodes', 'MMOEADC');
            addpath(targetPath, '-begin');

            delta=5;
        
            Population = Problem.Initialization();
            gen=1;
            while Algorithm.NotTerminated(Population)
                CrowdDis=Crowding(Population.decs);
                MatingPool=TournamentSelection(2,Problem.N,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Union=[Population Offspring];
                Population=Environmental_Selection(Union,Problem.N,delta);
                gen=gen+1;

                % Customized the implementation to allow retrieval of final-generation individuals.
                % Since MMOEA/DC does not use an archive, the final population is recovered instead.
                Algorithm.FinalPop = Population; % <-- Added
            end
        end
    end
end
