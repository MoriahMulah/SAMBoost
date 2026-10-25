# SAMBoost

## Abstract

We propose SAMBoost (Spatially Aware Mesh Boosting), a gradient boosting method that
combines tree-based covariate modelling with a stochastic partial differential
equation (SPDE) latent spatial field through a single penalised objective. At each
boosting iteration, the predictor is decomposed into separate covariate and spatial
components. The method is motivated by applications in which the spatial effect is not
a nuisance to be removed but a quantity of interest that must be recovered alongside
the covariates. In insurance, for example, practitioners need both flexible nonlinear
covariate effects and an interpretable geographic risk surface that can be reported on
its own. Existing approaches often involve trade-offs, offering flexible covariate
modelling but no explicit spatial field, rigorous spatial inference with a fixed
additive covariate structure, or a combination of both at computational costs that
limit practicality for large datasets. SAMBoost addresses these goals within the
boosting loop, and the underlying mesh produces a sparse precision structure whose
cost is largely independent of sample size. On simulations and on an open benchmark,
SAMBoost matches or improves on the predictive accuracy of joint Gaussian process and
tree boosting approaches. It is also faster and recovers the spatial field and
covariate effects more accurately. When covariate effects include interactions, it
outperforms additive Bayesian spatial models. Applied to a flood insurance portfolio
with 4.9 million records, SAMBoost trains on the full training set without subsampling
and runs about sixteen times faster than a Bayesian reference, with a consistent
spatial risk surface.

SAMBoost is a spatial gradient boosting method. Gradient-boosted trees model the
effect of the covariates, and a Gaussian Markov random field built on a triangular
mesh (SPDE approach) models the spatial effect.

This repository contains the code and results of the accompanying paper: the
simulation study, the house-price application and the fit-time scaling study. The
insurance application of the paper uses confidential data and is not included.

## Repository layout

    samboost/          the model
    utils/             mesh construction, variogram, cross-validation
    simulations/       data generators for the simulation study
    experiments/       one runner per method and simulation (SAMBoost, GPBoost, XGBoost, INLA)
    run_pipeline/      drivers of the simulation study, paper tables and figures
    real_data/house/   house-price application (Lucas County, open data)
    validation/        fit-time scaling study
    outputs/           results, tables and figures

## The model

- `samboost/model.py`: classification (Bernoulli likelihood, logit link)
- `samboost/model_continuous.py`: regression (squared error, custom objectives allowed)
- `samboost/model_extended.py`: extended version. One class for both likelihoods, plus
  an XGBoost tree backend, exposure weights, early stopping and intercept
  recalibration (used for the insurance application). With default settings it gives
  the same results as the two models above.

The model classes contain more options than the method described in the paper:
multi-resolution spatial fields (`n_levels > 1`), additional step-size modes for the
spatial update (`w_mode`), an iterative solver and other experimental settings. They
come from earlier tests and ongoing work. The scripts set the arguments explicitly.

## Simulation study

- Sim 1 (`generate_sim1_wave.py`): classification, smooth spatial field, no confounding.
- Sim 2 (`generate_sim2_hotspot_river.py`): regression, a local hotspot and a river
  covariate that is spatially structured (confounding), with interactions between
  covariates.
- Ablation (`generate_sim2_ablation_simpleF.py`): Sim 2 with an additive covariate
  effect, to show that INLA fails only when covariates interact.

Each seed generates one dataset on which all methods are fitted; reported values are
means over 30 seeds. Hyperparameters are tuned on seed 1 and reused for seeds 2-30.

To reproduce the results (from the repository root):

1. Python methods (SAMBoost, GPBoost, XGBoost): set `RUN_SIM1`, `RUN_SIM2` and/or
   `RUN_SIM2_ABLATION` to `True` at the top of `run_pipeline/run_multi_seed.py`, then

       python run_pipeline/run_multi_seed.py

2. INLA (R and R-INLA): set the same switches and `PYTHON_EXE` (used to generate the
   data) at the top of `run_pipeline/run_multi_seed_inla.R`, then

       Rscript run_pipeline/run_multi_seed_inla.R

   The INLA runners use the temporary folder `LOCAL_TMP` (`C:/inla_tmp`); change it if
   needed.
3. Tables and figures:

       python run_pipeline/build_paper.py    # --tables or --figures for one only

Results are written to `outputs/results/multi_seed/<sim>/run_<timestamp>/`.
`build_paper.py` reads the run folders listed at the top of
`run_pipeline/paper_helpers.py`; after new runs, update those paths.

The published run folders contain the 30-seed metrics, so `--tables` works directly.
The per-seed outputs (simulated data, fitted fields, predictions) are not published:
`--figures` needs steps 1 and 2 first, and without them the leak / peak columns of two
tables, which are not used in the paper, show "--".

## House-price application

Lucas County house sales (`spData::house`, 25,357 sales from 1993 to 1998) with the
expanding-window protocol of Sigrist (2022): for each test year t in {1996, 1997, 1998},
the models are trained on the sales before year t, the year t - 1 serving for validation.
The year of sale is also used as a covariate. The GPBoost settings start from Sigrist
(2022); with the help of Fabio Sigrist, some changes were made to them that improve on
the results of that paper.

To reproduce the results (from the repository root):

    # 1. Build the dataset (R)
    Rscript real_data/house/prepare_data.R

    # 2. Fit the Python methods with the selected hyperparameters
    python real_data/house/run_python_methods.py --method samboost --variant ex
    python real_data/house/run_python_methods.py --method gpboost  --variant ex  --threads 6
    python real_data/house/run_python_methods.py --method xgboost  --variant geo --threads 6

    # 3. Fit INLA, one test year per R process
    #    bash:       for t in 1996 1997 1998; do HOUSE_INCLUDE_SYEAR=1 INLA_FOLDS=$t Rscript real_data/house/run_inla.R; done
    #    PowerShell: foreach ($t in 1996,1997,1998) { $env:HOUSE_INCLUDE_SYEAR=1; $env:INLA_FOLDS=$t; Rscript real_data/house/run_inla.R }

    # 4. Tables and figures
    python real_data/house/tables_figures.py

`--variant ex`: the trees do not see the coordinates (for the spatial methods they enter
only the spatial component); `--variant geo`: the coordinates are also tree features.
Add `--grid` to the `run_python_methods.py` calls to redo the hyperparameter search
instead of using the selected values (about 1 to 2 hours per test year for the spatial
methods). Results are written to `outputs/results/house_price/final_syear/`.
The mesh-resolution figure is drawn from the stored tables in `final_syear/plots/`;
`run_mesh_study.py --mode reselect` (or `--mode fixed`) recomputes that study (slow).

INLA's mesh and prior are derived from a variogram on a random subsample of 300 points,
with no fixed seed, so reruns give slightly different INLA results. The INLA 1998 result
files come from such a rerun (RMSE 0.274; the paper reports 0.281 from the original run).
Rebuilding the house table from the stored files therefore gives an INLA mean of 0.323
instead of 0.326; no ranking changes.

## Fit-time scaling study

Fit time as a function of the number of training points (500 to 64,000), everything
else fixed:

    python validation/scaling/run_scaling.py           # Python methods (--smoke: quick test)
    Rscript validation/scaling/bench_inla.R            # INLA, on the datasets saved by run_scaling.py
    python validation/scaling/plot_scaling.py          # log-log figure and table
    python validation/scaling/plot_scaling_linear.py   # figure of the paper (linear axes)

## Requirements

Python 3.12 with the packages in `requirements.txt`.
R 4.4.3 with INLA, dplyr and jsonlite (simulations), sf and spData (house data).
The code was run on Windows 11.

## Citation

If you use this code, please cite the paper (full reference to be added):

    Moriah, M., Vermet, F., Ailliot, P., Naveau, P. and Legrand, J. (2026).
    SAMBoost: Spatially Aware Mesh Boosting.

## License

This code is released under the Apache License 2.0 (see `LICENSE`). It was developed
during a CIFRE PhD at Addactis and the Laboratoire de Mathematiques de Bretagne
Atlantique (LMBA).
