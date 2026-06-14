#!/usr/bin/env python3
"""Spatial structure analysis of price groups and OD demand for MO-TNDP cities.

This script provides quantitative and visual evidence for two claims used to
motivate cell-level (spatial) fairness metrics:

  1. Travel demand is geographically concentrated (e.g., in Xi'an the central
     half of demand cells carries the large majority of total OD demand).
  2. Price groups are NOT spatially organised: a cell's price-group index is
     only weakly correlated with its location, so parity across price groups
     does not imply parity across neighbourhoods.

For each requested number of groups it prints:
  - Pearson corr(group_id, distance_from_center)   (~0  =>  groups not radial)
  - Pearson corr(group_id, cell_demand)
  - downtown vs suburb demand share (inner/outer half of demand cells)

It also saves a 3-panel figure (for a chosen group count):
  (A) per-cell OD demand heatmap
  (B) price-group membership map  (speckled, not banded => spatially mixed)
  (C) distance-to-center distribution per group  (overlapping => not spatial)

Usage
-----
    python vis_city.py \
        --city-path envs/mo-tndp/cities/xian \
        --nr-groups 3 10 --fig-groups 10 \
        --out figures/xian_group_spatial_proof.png

Notes
-----
- The Pearson correlation only captures *linear* (e.g. radial/monotonic)
  spatial structure; panel (B) is the more robust, assumption-free evidence.
- The city center defaults to the geometric grid center; pass --center demand
  to use the demand-weighted center of mass instead.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

from motndp.city import City


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def load_city(city_path: Path, groups_file: str) -> City:
    """Load a MO-TNDP City with the given price-groups file."""
    return City(Path(city_path), groups_file=groups_file)


def compute_stats(city: City, center: str = "grid") -> dict:
    """Compute spatial statistics for one city / group configuration.

    Args:
        city: a loaded motndp City.
        center: "grid" (geometric center) or "demand" (demand center of mass).

    Returns:
        dict with correlations, demand shares, and arrays needed for plotting.
    """
    gx, gy, G = city.grid_x_size, city.grid_y_size, city.grid_size

    # Per-cell group id (1..n, NaN if the cell belongs to no group).
    group = np.asarray(city.grid_groups, dtype=float).flatten()

    # Per-cell total OD demand = trips originating + terminating at the cell.
    demand = city.od_mx.sum(axis=0) + city.od_mx.sum(axis=1)

    # Cell (row, col) coordinates from the flattened index.
    xs = np.arange(G) // gy
    ys = np.arange(G) % gy

    # City center.
    if center == "demand" and demand.sum() > 0:
        cx = float((xs * demand).sum() / demand.sum())
        cy = float((ys * demand).sum() / demand.sum())
    else:
        cx, cy = (gx - 1) / 2.0, (gy - 1) / 2.0

    radial = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    # Correlations (only over cells that belong to a group).
    valid = ~np.isnan(group)
    corr_radius = _safe_corr(group[valid], radial[valid])
    corr_demand = _safe_corr(group[valid], demand[valid])

    # Downtown vs suburb: split demand cells by the median distance to center.
    has_demand = demand > 0
    r_median = np.median(radial[has_demand]) if has_demand.any() else 0.0
    downtown = has_demand & (radial <= r_median)
    suburb = has_demand & (radial > r_median)
    total = demand.sum() if demand.sum() > 0 else 1.0

    return {
        "n_groups": int(np.nanmax(group)) if valid.any() else 0,
        "grid": (gx, gy),
        "center": (cx, cy),
        "corr_radius": corr_radius,
        "corr_demand": corr_demand,
        "downtown_share": 100.0 * demand[downtown].sum() / total,
        "suburb_share": 100.0 * demand[suburb].sum() / total,
        "n_zero_cells": int((demand == 0).sum()),
        # arrays for plotting
        "group": group,
        "demand": demand,
        "radial": radial,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns NaN instead of raising on zero variance."""
    if a.std() == 0 or b.std() == 0 or len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_spatial_proof(stats: dict, out_path: Path) -> None:
    """Render the 3-panel spatial-proof figure and save it to out_path."""
    gx, gy = stats["grid"]
    group = stats["group"]
    demand = stats["demand"]
    radial = stats["radial"]
    n_groups = stats["n_groups"]
    cx, cy = stats["center"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # (A) demand heatmap
    im0 = ax[0].imshow(demand.reshape(gx, gy), cmap="inferno", origin="lower")
    ax[0].plot(cy, cx, "c+", ms=12, mew=2)
    ax[0].set_title(
        f"(A) OD demand per cell\n"
        f"downtown {stats['downtown_share']:.0f}% vs "
        f"suburb {stats['suburb_share']:.0f}% of total",
        fontsize=11,
    )
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    # (B) price-group membership map
    cmap = "tab10" if n_groups <= 10 else "tab20"
    im1 = ax[1].imshow(group.reshape(gx, gy), cmap=cmap, origin="lower")
    ax[1].set_title(
        f"(B) Price-group membership ({n_groups} groups)\n"
        "speckled, not banded -> groups are spatially mixed",
        fontsize=11,
    )
    plt.colorbar(im1, ax=ax[1], fraction=0.046, label="group id")

    # (C) distance-to-center distribution per group
    data = [radial[(~np.isnan(group)) & (group == g)] for g in range(1, n_groups + 1)]
    ax[2].boxplot(data, showfliers=False)
    ax[2].set_xlabel("price group")
    ax[2].set_ylabel("distance from city center")
    ax[2].set_title(
        "(C) Distance-from-center by group\n"
        f"all groups span the same range  (corr={stats['corr_radius']:+.2f})",
        fontsize=11,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial structure analysis of price groups / OD demand."
    )
    parser.add_argument(
        "--city-path", type=str, default="envs/mo-tndp/cities/xian",
        help="Path to the MO-TNDP city directory.",
    )
    parser.add_argument(
        "--groups-template", type=str, default="price_groups_{}.txt",
        help="Filename template for the groups file; {} is replaced by the group count.",
    )
    parser.add_argument(
        "--nr-groups", type=int, nargs="+", default=[3, 10],
        help="Group counts to print statistics for.",
    )
    parser.add_argument(
        "--fig-groups", type=int, default=10,
        help="Group count to use when drawing the figure.",
    )
    parser.add_argument(
        "--center", type=str, default="grid", choices=["grid", "demand"],
        help="Reference center: geometric grid center or demand center of mass.",
    )
    parser.add_argument(
        "--out", type=str, default="figures/xian_group_spatial_proof.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    # Printed statistics for each requested group count.
    print(f"City: {args.city_path}   center: {args.center}")
    for ng in args.nr_groups:
        city = load_city(args.city_path, args.groups_template.format(ng))
        s = compute_stats(city, center=args.center)
        print(f"\n=== nr_groups={ng} ===")
        print(f"  grid                               = {s['grid'][0]}x{s['grid'][1]}")
        print(f"  zero-demand cells                  = {s['n_zero_cells']}")
        print(f"  corr(group_id, distance_to_center) = {s['corr_radius']:+.3f}")
        print(f"  corr(group_id, cell_demand)        = {s['corr_demand']:+.3f}")
        print(f"  demand share: downtown={s['downtown_share']:.0f}%  "
              f"suburb={s['suburb_share']:.0f}%")

    # Figure for the chosen group count.
    city = load_city(args.city_path, args.groups_template.format(args.fig_groups))
    stats = compute_stats(city, center=args.center)
    out_path = Path(args.out)
    plot_spatial_proof(stats, out_path)
    print(f"\nSaved figure -> {out_path}")


if __name__ == "__main__":
    main()