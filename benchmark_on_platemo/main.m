
thisFilePath = fileparts(mfilename('fullpath'));
cd(thisFilePath);

addpath(genpath(fullfile(thisFilePath, '..', 'PlatEMO')));
rehash;

PYTHON_PATH = '..\.venv\Scripts\python.exe';
PYTHON_PATH_ABS = canonical_path(PYTHON_PATH);

pe = pyenv;
if pe.Status == "NotLoaded"
    pe = pyenv('Version', PYTHON_PATH_ABS, 'ExecutionMode', 'InProcess');
else
    current_abs = canonical_path(char(pe.Executable));
    if ~strcmpi(current_abs, PYTHON_PATH_ABS)
        error(['Python environment mismatch:\n' ...
               '  Current   : %s\n' ...
               '  Requested : %s\n' ...
               'Restart MATLAB to switch.'], current_abs, PYTHON_PATH_ABS);
    end
end
fprintf('CurrentPython: %s\n', char(pe.Executable));


function p = canonical_path(p)
    info = dir(p);
    if ~isempty(info)
        p = fullfile(info(1).folder, info(1).name);
    else
        if ~startsWith(p, ["\\", "/"]) && isempty(regexp(p, '^[A-Za-z]:', 'once'))
            p = fullfile(pwd, p);
        end
    end
end

PROJECT_ROOT = fullfile(thisFilePath, '..');

if count(py.sys.path, PROJECT_ROOT) == 0
    insert(py.sys.path, int32(0), PROJECT_ROOT);
end

construct_msg = reload_py('x_msg.construct_msg_landscape');
sampling_py   = reload_py('x_msg.sampling');
make_mo_msg   = reload_py('x_msg.make_multi_objective_msg');
extract_feat  = reload_py('x_msg.extract_features');
torch_py      = py.importlib.import_module('torch');

BASE_SEED     = 42;
D_class       = [2, 5, 10];
num_obj = 2;
eps = 0.1;
pf_shape = "convex";  % Pareto front shape: "convex" or "concave"

for d = 1:length(D_class)
    D = D_class(d);
    fprintf('=== Starting experiments for D=%d ===\n', D);
    NUM_GAUSSIANS = 50 * D;
    DEVICE        = 'cpu';
    NUM_TRIALS    = 31;
    NUM_SAMPLES   = 500 * D;

    types = ["max_max_max", "max_max_min", "max_min_max", "max_min_min", ...
            "min_max_max", "min_max_min", "min_min_max", "min_min_min"];

    for i = 1:length(types)
        fprintf('Type %d: %s\n', i, types(i));

        type = types(i); 

        CKPT_PATH = char(fullfile( ...
                            PROJECT_ROOT, ...
                            'res_rq2', ...
                            'msg_ela', ...
                            "results_" + type + "_" + num2str(D) + "d.pt"));
        CSV_FILE  = "platemo_benchmark_results_" + type + "_" + pf_shape + "_" + num2str(D) + "d.csv";
        FEATURE_LIST = py.list({'optima_feature','fdc_feature', ...
                                'disp_feature','r2_feature'});
        P_LIST       = py.list({0.1});

        means = sampling_py.sobol_sampling( ...
            py.int(D), py.int(NUM_GAUSSIANS), ...
            pyargs('device', py.str(DEVICE), 'seed', py.int(BASE_SEED)));
        means = means.to(py.str(DEVICE));

        ckpt          = torch_py.load(CKPT_PATH);
        theta_history = double(ckpt{'theta_history'}.detach().cpu().numpy());

        completed = containers.Map('KeyType', 'char', 'ValueType', 'logical');
        if isfile(CSV_FILE)
            T = readtable(CSV_FILE);
            bad = any(ismissing(T(:, {'idx','seed'})), 2);
            if any(bad)
                warning('Dropping %d corrupted row(s) from resume map', sum(bad));
                T(bad, :) = [];
            end
            for r = 1:height(T)
                k = sprintf('%d_%d', T.idx(r), T.seed(r));
                completed(k) = true;
            end
            fprintf('Resume mode: %d rows already in %s\n', height(T), CSV_FILE);
        else
            fprintf('Fresh run: %s not found\n', CSV_FILE);
        end

        make_key = @(i, s) sprintf('%d_%d', i, s);

        for idx = 1:size(theta_history, 1)

            all_done = true;
            for t = 1:NUM_TRIALS
                if ~isKey(completed, make_key(idx, BASE_SEED + t))
                    all_done = false; break;
                end
            end
            if all_done
                fprintf('Skip idx=%d (all %d trials done)\n', idx, NUM_TRIALS);
                continue;
            end

            theta = py.torch.tensor(theta_history(idx, :), pyargs( ...
                'dtype', py.torch.float32, 'device', py.str(DEVICE)));
            msg   = construct_msg.MSGLandscape(means, theta);
            msg.eval();

            features = extract_feat.compute_features( ...
                py.int(NUM_SAMPLES), theta, means, P_LIST, DEVICE, FEATURE_LIST);
            num_local_optima = double(features{'num_local_optima'}.cpu().numpy());
            fdc              = double(features{'fdc'}.cpu().numpy());
            dispersion       = double(features{'disp_10pct'}.cpu().numpy());

            mo_msg = make_mo_msg.make_multi_objective_msg(pyargs( ...
                'm', py.int(num_obj), 'dim_msg', py.int(D), ...
                'function_g', msg, 'pf_shape', py.str(pf_shape)));

            multi_msg = @(x) msg_wrapper_batch(x, mo_msg, DEVICE);
            PRO       = MyProblem(mo_msg, DEVICE, D+(num_obj-1));

            optima     = msg.find_optima_exact(py.float(0.0));
            local_opt  = optima(1);
            global_opt = optima(3);

            X_local  = sample_pareto_set(local_opt,  num_obj, round(1000^(1/(num_obj-1))));
            X_global = sample_pareto_set(global_opt, num_obj, round(1000^(1/(num_obj-1))));
            F_local  = multi_msg(X_local);
            F_global = multi_msg(X_global);

            [pop_filtered, ~] = get_epsilon_local_optima( ...
                X_local, F_local, F_global, eps);
            n_eps_local = size(pop_filtered, 1);

            for trial = 1:NUM_TRIALS
                seed = BASE_SEED + trial;
                if isKey(completed, make_key(idx, seed))
                    continue;
                end

                IdealPop.best.objs = F_global;  
                IdealHV = HV(IdealPop, F_global);

                rng(seed);
                py.random.seed(py.int(seed));
                py.numpy.random.seed(py.int(seed));
                py.torch.manual_seed(py.int(seed));

                scores_IGDX = zeros(1, 4);
                scores_Global_IGDX = zeros(1, 4);
                scores_HV   = zeros(1, 4);
                HREA     = custom_HREA();
                stime = tic;
                HREA.Solve(PRO);
                time = toc(stime);
                FinalPop_HREA = HREA.FinalPop;
                scores_IGDX(1)    = IGDX(FinalPop_HREA, pop_filtered);
                scores_Global_IGDX(1) = IGDX(FinalPop_HREA, X_global);
                scores_HV(1)      = HV(FinalPop_HREA, F_global)/IdealHV;
                pop_size = length(FinalPop_HREA);
                fprintf("pop_size=%d\n", pop_size);
                fprintf('HREA trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f, Time=%.2f\n', trial, NUM_TRIALS, scores_IGDX(1), scores_Global_IGDX(1), scores_HV(1), time);

                rng(seed);
                py.random.seed(py.int(seed));
                py.numpy.random.seed(py.int(seed));
                py.torch.manual_seed(py.int(seed));

                MMEAWI    = custom_MMEAWI();
                stime = tic;
                MMEAWI.Solve(PRO);
                time = toc(stime);
                FinalPop_MMEAWI = MMEAWI.FinalPop;
                scores_IGDX(2)    = IGDX(FinalPop_MMEAWI, pop_filtered);
                scores_Global_IGDX(2) = IGDX(FinalPop_MMEAWI, X_global);
                scores_HV(2)      = HV(FinalPop_MMEAWI, F_global)/IdealHV;
                final_pop_size = length(FinalPop_MMEAWI);
                fprintf("final_pop_size=%d\n", final_pop_size);
                fprintf('MMEAWI trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f, Time=%.2f\n', trial, NUM_TRIALS, scores_IGDX(2), scores_Global_IGDX(2), scores_HV(2), time);

                rng(seed);
                py.random.seed(py.int(seed));
                py.numpy.random.seed(py.int(seed));
                py.torch.manual_seed(py.int(seed));

                MMOEADC     = custom_MMOEADC();
                stime = tic;
                MMOEADC.Solve(PRO);
                time = toc(stime);
                FinalPop_MMOEADC = MMOEADC.FinalPop;
                scores_IGDX(3)    = IGDX(FinalPop_MMOEADC, pop_filtered);
                scores_Global_IGDX(3) = IGDX(FinalPop_MMOEADC, X_global);
                scores_HV(3)      = HV(FinalPop_MMOEADC, F_global)/IdealHV;
                mmoeadc_pop_size = length(FinalPop_MMOEADC);
                fprintf("mmoeadc_pop_size=%d\n", mmoeadc_pop_size);
                fprintf('MMOEADC trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f, Time=%.2f\n', trial, NUM_TRIALS, scores_IGDX(3), scores_Global_IGDX(3), scores_HV(3), time);

                rng(seed);
                py.random.seed(py.int(seed));
                py.numpy.random.seed(py.int(seed));
                py.torch.manual_seed(py.int(seed));

                NSGAII     = custom_NSGAII();
                stime = tic;
                NSGAII.Solve(PRO);
                time = toc(stime);
                FinalPop_NSGAII = NSGAII.FinalPop;
                scores_IGDX(4)    = IGDX(FinalPop_NSGAII, pop_filtered);
                scores_Global_IGDX(4) = IGDX(FinalPop_NSGAII, X_global);
                scores_HV(4)      = HV(FinalPop_NSGAII, F_global)/IdealHV;
                nsgaii_pop_size = length(FinalPop_NSGAII);
                fprintf("nsgaii_pop_size=%d\n", nsgaii_pop_size);
                fprintf('NSGAII trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f, Time=%.2f\n', trial, NUM_TRIALS, scores_IGDX(4), scores_Global_IGDX(4), scores_HV(4), time);


                fprintf('D=%d, idx=%d, trial=%d/%d finished\n', D, idx, trial, NUM_TRIALS);

                append_row(CSV_FILE, idx, seed, n_eps_local, ...
                        num_local_optima, fdc, dispersion, scores_IGDX, scores_Global_IGDX, scores_HV);

                completed(make_key(idx, seed)) = true;
            end
        end
    end
end


function m = reload_py(name)
    m = py.importlib.import_module(name);
    py.importlib.reload(m);
end

function append_row(csv_file, idx, seed, n_eps, n_local, fdc, disp10, scores_IGDX, scores_Global_IGDX, scores_HV)
    if ~isfile(csv_file)
        header = {'idx','seed','num_epsilon_local_optima', ...
                  'num_local_optima','fdc','disp_10pct', ...
                  'IGDX_HREA','IGDX_MMEAWI','IGDX_MMOEADC','IGDX_NSGAII', ...
                  'IGDX_Global_HREA','IGDX_Global_MMEAWI','IGDX_Global_MMOEADC','IGDX_Global_NSGAII', ...
                  'HV_HREA','HV_MMEAWI','HV_MMOEADC','HV_NSGAII'};
        writecell(header, csv_file);
    end
    row = [{idx}, {seed}, {n_eps}, {n_local}, {fdc}, {disp10}, num2cell(scores_IGDX), num2cell(scores_Global_IGDX), num2cell(scores_HV)];
    writecell(row, csv_file, 'WriteMode', 'append');
end