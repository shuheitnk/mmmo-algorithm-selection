function score = run_single_eta(eta, seed, PRO, pop_filtered)

    rng(seed);
    py.random.seed(seed);

    ALG = custom_HREA('save', 0);
    ALG.p   = 1.0;
    ALG.eta = eta;
    ALG.eps = 0.1;

    ALG.Solve(PRO);

    ArchiveDecs = vertcat(ALG.Archive.dec);
    score = IGDX(ArchiveDecs, pop_filtered);
end