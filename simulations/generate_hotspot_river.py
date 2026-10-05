"""
generate_hotspot_river.py
=========================
Spatial pieces shared by the Sim 2 and ablation generators and by the
analysis scripts:

  hotspot_river_effect(locs)   the true spatial field of Sim 2 (Mexican hat)
      b(s) = A * [exp(-d^2 / (2 sigma_in^2)) - S * exp(-d^2 / (2 sigma_out^2))]
      d = distance to HOTSPOT_CENTER = (0.5, 0.2), A = 2.5,
      sigma_in = 0.15, sigma_out = 0.35, S = 0.35

  river_indicator(locs)        1 inside a corridor around the line
      RIVER_P1 = (0, 0.70) -> RIVER_P2 = (1, 0.90), 0 outside.
      Default half-width 0.10, used as the corridor mask of the "leak" metric.
      The Sim 2 data use a half-width of 0.15 (river_indicator_wide in
      generate_sim2_hotspot_river.py).
"""

import numpy as np

HOTSPOT_CENTER = np.array([0.50, 0.20])
HOTSPOT_AMP    = 2.5
SIGMA_INNER    = 0.15
SIGMA_OUTER    = 0.35
RING_STRENGTH  = 0.35

RIVER_P1        = np.array([0.0, 0.70])
RIVER_P2        = np.array([1.0, 0.90])
RIVER_HALFWIDTH = 0.10          # 1 if perpendicular distance to the line < this


def hotspot_river_effect(locs, center=HOTSPOT_CENTER, amplitude=HOTSPOT_AMP,
                         sigma_in=SIGMA_INNER, sigma_out=SIGMA_OUTER,
                         ring_strength=RING_STRENGTH):
    """b(s) = A*[exp(-d^2/2 sig_in^2) - S*exp(-d^2/2 sig_out^2)], one hotspot."""
    sq = ((locs - center) ** 2).sum(axis=1)
    peak = np.exp(-sq / (2 * sigma_in ** 2))
    ring = np.exp(-sq / (2 * sigma_out ** 2))
    return amplitude * (peak - ring_strength * ring)


def river_indicator(locs, p1=RIVER_P1, p2=RIVER_P2, halfwidth=RIVER_HALFWIDTH):
    """Binary corridor: 1 inside the band (perpendicular distance < halfwidth) else 0."""
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    point_vec = locs - p1
    cross = point_vec[:, 0] * line_vec[1] - point_vec[:, 1] * line_vec[0]
    d = np.abs(cross) / line_len
    return (d < halfwidth).astype(float)
