"""
ILP theoretical bounds for MO-TNDP metrics (Gurobi only).

Computes optimal bounds for: per-group Utopia, Efficiency, Max-Min, Gini, Sen
Welfare, Nash Welfare (+ geometric-mean Nash), and cell-level Demand Coverage.
Supports relaxed (no path) and exact (sequence-based path connectivity) models.

Exact path uses the Sequence Selection formulation:
  y_{i,t} in {0,1} : cell i is the t-th station of the line
  adjacency        : if y_{i,t}=1 then some neighbour j has y_{j,t+1}=1

On large instances the exact-path MIP usually hits the time limit; in that case
the printed result is the best INCUMBENT (achievable) plus the DUAL BOUND
(a valid upper bound the optimum cannot exceed) and the MIP gap.

Usage:
    # 1) Utopia first (per-group upper bounds)
    python bound.py --env xian --groups_file price_groups_10.txt \
        --nr_stations 20 --starting_loc_x 9 --starting_loc_y 19 \
        --exact_path --time_limit 1200 --utopia_only

    # 2) Then the metric bounds (skip utopia, already done)
    python bound.py --env xian --groups_file price_groups_10.txt \
        --nr_stations 20 --starting_loc_x 9 --starting_loc_y 19 \
        --exact_path --time_limit 1200 --no_utopia \
        --metrics efficiency maxmin demand_coverage nash
"""

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False


_STATUS = {
    1: "Loaded", 2: "Optimal", 3: "Infeasible", 4: "Inf/Unbounded",
    5: "Unbounded", 9: "TimeLimit", 11: "Interrupted", 13: "Suboptimal",
}


def status_name(code):
    return _STATUS.get(code, f"Status{code}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_city(env_name, groups_file):
    from motndp.city import City
    return City(Path(f"envs/mo-tndp/cities/{env_name}"), groups_file=groups_file)


def build_adjacency(city):
    """4-connectivity adjacency from the grid structure."""
    adj = {i: set() for i in range(city.grid_size)}
    for i in range(city.grid_size):
        x, y = i // city.grid_y_size, i % city.grid_y_size
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < city.grid_x_size and 0 <= ny < city.grid_y_size:
                adj[i].add(nx * city.grid_y_size + ny)
    return adj


def get_nonzero_od_pairs(od_matrix, grid_size):
    """Upper-triangle non-zero OD pairs."""
    pairs = []
    for i in range(grid_size):
        for j in range(i + 1, grid_size):
            if od_matrix[i, j] > 0 or od_matrix[j, i] > 0:
                pairs.append((i, j))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Gurobi solver
# ═══════════════════════════════════════════════════════════════════════════════

class GurobiSolver:
    """Build and solve the MO-TNDP ILP bounds with Gurobi."""

    def __init__(self, city, adj, nr_stations, starting_loc, exact_path=False,
                 time_limit=600):
        if not HAS_GUROBI:
            raise RuntimeError("Gurobi not available. Install with: pip install gurobipy")
        self.city = city
        self.adj = adj
        self.K = nr_stations
        self.G_size = city.grid_size
        self.s = starting_loc
        self.exact_path = exact_path
        self.time_limit = time_limit
        self.n_groups = len(city.groups)

        self.group_pairs = [get_nonzero_od_pairs(city.group_od_mx[g], self.G_size)
                            for g in range(self.n_groups)]
        all_pairs = set()
        for pairs in self.group_pairs:
            all_pairs.update(pairs)
        self.all_pairs = sorted(all_pairs)

    # ---- model construction --------------------------------------------------
    def _build_base(self, name):
        m = gp.Model(name)
        m.Params.TimeLimit = self.time_limit
        m.Params.OutputFlag = 0

        x = m.addVars(self.G_size, vtype=GRB.BINARY, name="x")
        z = m.addVars(self.all_pairs, vtype=GRB.BINARY, name="z")

        m.addConstr(gp.quicksum(x[i] for i in range(self.G_size)) == self.K)
        m.addConstr(x[self.s] == 1)

        for (i, j) in self.all_pairs:                     # McCormick: z = x_i AND x_j
            m.addConstr(z[i, j] <= x[i])
            m.addConstr(z[i, j] <= x[j])
            m.addConstr(z[i, j] >= x[i] + x[j] - 1)

        if self.exact_path:                               # sequence-based connectivity
            y = m.addVars(self.G_size, self.K, vtype=GRB.BINARY, name="y")
            m.addConstr(y[self.s, 0] == 1)
            for t in range(self.K):
                m.addConstr(gp.quicksum(y[i, t] for i in range(self.G_size)) == 1)
            for i in range(self.G_size):
                m.addConstr(gp.quicksum(y[i, t] for t in range(self.K)) <= 1)
            for t in range(self.K - 1):
                for i in range(self.G_size):
                    m.addConstr(gp.quicksum(y[j, t + 1] for j in self.adj[i]) >= y[i, t])
            for i in range(self.G_size):
                m.addConstr(x[i] == gp.quicksum(y[i, t] for t in range(self.K)))

        return m, x, z

    def _R(self, m, z, g):
        """Group return R_g as a Gurobi linear expression."""
        od_g = self.city.group_od_mx[g]
        total_g = self.city.group_od_sum[g]
        return gp.quicksum((od_g[i, j] + od_g[j, i]) * z[i, j]
                           for (i, j) in self.group_pairs[g]) / total_g

    def _extract(self, x, z):
        returns = []
        for g in range(self.n_groups):
            od_g = self.city.group_od_mx[g]
            total_g = self.city.group_od_sum[g]
            val = sum((od_g[i, j] + od_g[j, i])
                      for (i, j) in self.group_pairs[g] if z[i, j].X > 0.5) / total_g
            returns.append(val)
        cells = sorted(i for i in range(self.G_size) if x[i].X > 0.5)
        return returns, cells

    # ---- generic optimize + report (handles time-limit incumbents) -----------
    def _run(self, m, x, z, name, maximize, extra=None):
        t0 = time.time()
        m.optimize()
        elapsed = time.time() - t0
        res = {"metric": name, "direction": "max" if maximize else "min",
               "status": status_name(m.Status), "value": None, "bound": None,
               "gap": None, "returns": [], "cells": [], "time": elapsed}
        if m.SolCount > 0:                                # have an incumbent
            returns, cells = self._extract(x, z)
            res.update(value=m.ObjVal, bound=m.ObjBound, gap=m.MIPGap,
                       returns=returns, cells=cells)
            if m.Status == GRB.OPTIMAL:
                res["status"] = "Optimal"
        if extra:
            res.update(extra(res))
        return res

    # ---- metric objectives ---------------------------------------------------
    def solve_per_group(self, g, maximize=True):
        m, x, z = self._build_base(f"R{g}")
        m.setObjective(self._R(m, z, g), GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
        return self._run(m, x, z, f"R_group{g}", maximize)

    def solve_efficiency(self, maximize=True):
        m, x, z = self._build_base("Efficiency")
        m.setObjective(gp.quicksum(self._R(m, z, g) for g in range(self.n_groups)),
                       GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
        return self._run(m, x, z, "Efficiency", maximize)

    def solve_maxmin(self):
        m, x, z = self._build_base("MaxMin")
        t = m.addVar(lb=0, name="t_floor")
        for g in range(self.n_groups):
            m.addConstr(t <= self._R(m, z, g))
        m.setObjective(t, GRB.MAXIMIZE)
        return self._run(m, x, z, "Max-Min", True)

    def solve_demand_coverage(self):
        """Max demand-weighted spatial reach: fraction of total cell demand located
        in served cells. Upper bound uses served_i = x_i (tight for a connected line)."""
        m, x, z = self._build_base("DemandCoverage")
        D = self.city.od_mx.sum(axis=0) + self.city.od_mx.sum(axis=1)
        total = float(D.sum())
        m.setObjective(gp.quicksum(float(D[i]) * x[i] for i in range(self.G_size)) / total,
                       GRB.MAXIMIZE)
        return self._run(m, x, z, "Demand Coverage", True)

    def solve_sen_welfare(self, maximize=True):
        m, x, z = self._build_base("SenWelfare")
        R = {g: self._R(m, z, g) for g in range(self.n_groups)}
        diff = []
        for g in range(self.n_groups):
            for h in range(g + 1, self.n_groups):
                d = m.addVar(lb=0, name=f"d_{g}_{h}")
                m.addConstr(d >= R[g] - R[h])
                m.addConstr(d >= R[h] - R[g])
                diff.append(d)
        n = self.n_groups
        m.setObjective(gp.quicksum(R[g] for g in range(n)) - (1.0 / n) * gp.quicksum(diff),
                       GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
        return self._run(m, x, z, "Sen Welfare", maximize)

    def solve_gini(self, minimize=True):
        m, x, z = self._build_base("Gini")
        R = {g: self._R(m, z, g) for g in range(self.n_groups)}
        diff = []
        for g in range(self.n_groups):
            for h in range(g + 1, self.n_groups):
                d = m.addVar(lb=0, name=f"d_{g}_{h}")
                m.addConstr(d >= R[g] - R[h])
                m.addConstr(d >= R[h] - R[g])
                diff.append(d)
        # avoid the all-zero solution
        m.addConstr(gp.quicksum(R[g] for g in range(self.n_groups)) >= 1e-4)
        m.setObjective(gp.quicksum(diff), GRB.MINIMIZE if minimize else GRB.MAXIMIZE)
        res = self._run(m, x, z, "Gini", minimize)
        res["actual_gini"] = self._gini(res["returns"]) if res["returns"] else None
        return res

    def solve_nash_welfare(self):
        """Nash welfare via log transform: max sum_g log(R_g) (== max product)."""
        m, x, z = self._build_base("Nash")
        log_R = []
        for g in range(self.n_groups):
            Rg = m.addVar(lb=1e-8, name=f"R_{g}")
            m.addConstr(Rg == self._R(m, z, g) + 1e-8)
            lg = m.addVar(lb=-GRB.INFINITY, name=f"logR_{g}")
            m.addGenConstrLog(Rg, lg)
            log_R.append(lg)
        m.setObjective(gp.quicksum(log_R), GRB.MAXIMIZE)
        m.Params.NonConvex = 2

        def _extra(res):
            rr = res.get("returns")
            if rr:
                prod = float(np.prod(rr))
                geom = float(np.exp(np.mean(np.log(np.clip(np.asarray(rr, float), 1e-12, None)))))
                return {"product": prod, "geom_mean": geom}
            return {}

        return self._run(m, x, z, "Nash Welfare", True, extra=_extra)

    @staticmethod
    def _gini(returns):
        r = np.asarray(returns, dtype=float)
        n = len(r)
        if r.sum() == 0:
            return 0.0
        diffs = sum(abs(r[i] - r[j]) for i in range(n) for j in range(i + 1, n))
        return diffs / (n * r.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(r):
    print(f"  Status: {r['status']}")
    if r.get('value') is not None:
        print(f"  Incumbent (achievable): {r['value']:.8f}")
        if r.get('bound') is not None:
            print(f"  Dual bound (valid UB) : {r['bound']:.8f}   gap={r.get('gap', 0):.2%}")
    if r.get('actual_gini') is not None:
        print(f"  Actual Gini: {r['actual_gini']:.8f}")
    if r.get('product') is not None:
        print(f"  Nash product: {r['product']:.8f}   geom-mean: {r['geom_mean']:.8f}")
    if r.get('returns'):
        print(f"  Per-group returns: {[f'{v:.6f}' for v in r['returns']]}")
    if r.get('cells'):
        print(f"  Selected cells ({len(r['cells'])}): {r['cells']}")
    if r.get('time') is not None:
        print(f"  Time: {r['time']:.1f}s")


def main():
    p = argparse.ArgumentParser(description="Gurobi ILP bounds for MO-TNDP metrics")
    p.add_argument("--env", type=str, default="dilemma_5x5")
    p.add_argument("--groups_file", type=str, default="groups.txt")
    p.add_argument("--nr_stations", type=int, default=5)
    p.add_argument("--starting_loc_x", type=int, default=2)
    p.add_argument("--starting_loc_y", type=int, default=2)
    p.add_argument("--exact_path", action="store_true",
                   help="add sequence-based path connectivity constraints")
    p.add_argument("--time_limit", type=int, default=600, help="per-solve time limit (s)")
    p.add_argument("--utopia_only", action="store_true", help="only per-group bounds")
    p.add_argument("--no_utopia", action="store_true", help="skip per-group bounds")
    p.add_argument("--metrics", type=str, nargs="+",
                   default=["efficiency", "maxmin", "gini", "sen", "nash", "demand_coverage"])
    args = p.parse_args()

    if not HAS_GUROBI:
        print("ERROR: gurobipy not available / no license. Install gurobipy and set a license.")
        return

    city = load_city(args.env, args.groups_file)
    adj = build_adjacency(city)
    start = args.starting_loc_x * city.grid_y_size + args.starting_loc_y

    print("=" * 60)
    print("MO-TNDP Theoretical Bounds (Gurobi)")
    print("=" * 60)
    print(f"Env: {args.env} ({city.grid_x_size}x{city.grid_y_size} = {city.grid_size} cells)")
    print(f"Groups: {len(city.groups)}, Stations: {args.nr_stations}, "
          f"Start: ({args.starting_loc_x},{args.starting_loc_y})=cell {start}")
    print(f"Path: {'EXACT (sequence)' if args.exact_path else 'RELAXED'}, "
          f"time_limit={args.time_limit}s")
    print("=" * 60)

    solver = GurobiSolver(city, adj, args.nr_stations, start, args.exact_path, args.time_limit)
    results = {}

    if not args.no_utopia:
        print("\n" + "-" * 60 + "\nPer-Group Upper Bounds (Utopia Point)\n" + "-" * 60)
        utopia = []
        for g in range(len(city.groups)):
            print(f"\n  > Group {int(city.groups[g])} UB:")
            r = solver.solve_per_group(g, maximize=True)
            print_result(r)
            utopia.append(r.get("value"))
        results["utopia"] = utopia

    if not args.utopia_only:
        for metric in args.metrics:
            print("\n" + "-" * 60)
            if metric == "efficiency":
                print("Efficiency = sum_g R_g\n  > Upper Bound:")
                results["efficiency_ub"] = solver.solve_efficiency(True)
                print_result(results["efficiency_ub"])
            elif metric == "maxmin":
                print("Max-Min = max min_g R_g\n  > Upper Bound:")
                results["maxmin_ub"] = solver.solve_maxmin()
                print_result(results["maxmin_ub"])
            elif metric == "demand_coverage":
                print("Demand Coverage = served demand / total demand\n  > Upper Bound:")
                results["demand_coverage_ub"] = solver.solve_demand_coverage()
                print_result(results["demand_coverage_ub"])
            elif metric == "gini":
                print("Gini\n  > Lower Bound (most equal):")
                results["gini_lb"] = solver.solve_gini(minimize=True)
                print_result(results["gini_lb"])
            elif metric == "sen":
                print("Sen Welfare = E*(1-Gini)\n  > Upper Bound:")
                results["sen_ub"] = solver.solve_sen_welfare(True)
                print_result(results["sen_ub"])
            elif metric == "nash":
                print("Nash Welfare = prod R_g (+ geometric mean)\n  > Upper Bound:")
                results["nash_ub"] = solver.solve_nash_welfare()
                print_result(results["nash_ub"])

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print(f"Path: {'EXACT' if args.exact_path else 'RELAXED'}")
    if "utopia" in results:
        print(f"Utopia: {[f'{v:.6f}' if v else 'N/A' for v in results['utopia']]}")
    for key in ["efficiency_ub", "maxmin_ub", "sen_ub", "demand_coverage_ub"]:
        if key in results and results[key].get("value") is not None:
            r = results[key]
            print(f"{key}: incumbent={r['value']:.6f}  bound={r['bound']:.6f}  gap={r['gap']:.2%}")
    if "gini_lb" in results and results["gini_lb"].get("actual_gini") is not None:
        print(f"gini_lb: {results['gini_lb']['actual_gini']:.6f}")
    if "nash_ub" in results and results["nash_ub"].get("value") is not None:
        r = results["nash_ub"]
        print(f"nash_ub: log={r['value']:.4f}  product={r.get('product'):.8f}  "
              f"geom={r.get('geom_mean'):.6f}")


if __name__ == "__main__":
    main()
