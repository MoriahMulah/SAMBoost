r"""
build_paper.py

Single driver that regenerates every paper figure and table from the 30-seed
results, using the helpers in paper_helpers.py. Run from anywhere:

    python run_pipeline/build_paper.py            # everything (figures + tables)
    python run_pipeline/build_paper.py --figures  # figures only
    python run_pipeline/build_paper.py --tables   # tables only

Outputs -> outputs/paper_figures/

Figures:
  fig_sim1_domain, fig_sim1_field_repseed, fig_sim1_field_mean
  fig_sim2_domain, fig_sim2_field_repseed, fig_sim2_field_mean
  fig_ablation_Fcollapse_015                (F recovery, simple vs complex F)
Tables:
  table_agg_sim1_wave, table_agg_sim2_hotspot_river            (30-seed mean +/- sd)
  table_sim2_final_bis_augmented                               (adds leak / peak)
  table_ablation_Fcomplexity_015                               (simple vs complex F)
"""
import argparse
import paper_helpers as P


def build_figures():
    print("=" * 60 + "\n  Paper figures\n" + "=" * 60)
    for cfg, tag in [(P.SIM1, "sim1"), (P.SIM2, "sim2")]:
        med = P.rep_seed(cfg)
        P.domain_figure(cfg, f"fig_{tag}_domain.png")
        P.field_figure(cfg, cfg["rep_methods"], med, f"fig_{tag}_field_repseed.png")
        P.field_figure_mean(cfg, cfg["rep_methods"], f"fig_{tag}_field_mean.png")
    # ablation figure: F-recovery, rows = simple/complex F, cols = SAMBoost/INLA
    P.ablation_figure(P.SIM2, P.ABLATION, "fig_ablation_Fcollapse_015.png")


def build_tables():
    print("=" * 60 + "\n  Paper tables\n" + "=" * 60)
    P.aggregate_table(P.SIM1)
    P.aggregate_table(P.SIM2)
    P.augmented_table(P.SIM2, "table_sim2_final_bis_augmented")
    P.ablation_table(P.SIM2, P.ABLATION, "table_ablation_Fcomplexity_015")


def main():
    ap = argparse.ArgumentParser(description="Build the paper's figures and tables.")
    ap.add_argument("--figures", action="store_true", help="figures only")
    ap.add_argument("--tables", action="store_true", help="tables only")
    a = ap.parse_args()
    do_figs = a.figures or not a.tables
    do_tabs = a.tables or not a.figures
    if do_figs:
        build_figures()
    if do_tabs:
        build_tables()
    print(f"\nDone -> {P.OUT}")


if __name__ == "__main__":
    main()
