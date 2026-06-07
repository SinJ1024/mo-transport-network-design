"""
ILP-based theoretical upper/lower bounds for MO-TNDP metrics.

Computes bounds for: Efficiency, Max-Min Fairness, Gini, Sen Welfare, Nash Welfare.
Supports both relaxed (no path) and exact (sequence-based path) formulations.

Path connectivity uses the Sequence Selection formulation:
  y_{i,t} ∈ {0,1} = cell i is the t-th station in the path
  Adjacency: if y_{i,t}=1, then some neighbor j has y_{j,t+1}=1

Usage:
    # Relaxed bounds (fast, valid UB)
    python compute_upper_bounds.py --env dilemma_5x5 --groups_file groups.txt \
        --nr_stations 5 --starting_loc_x 2 --starting_loc_y 2

    # Exact path bounds (slower, tight)
    python compute_upper_bounds.py --env dilemma_5x5 --groups_file groups.txt \
        --nr_stations 5 --starting_loc_x 2 --starting_loc_y 2 --exact_path

    # Xi'an (needs Gurobi license for speed)
    python compute_upper_bounds.py --env xian --groups_file price_groups_3.txt \
        --nr_stations 20 --starting_loc_x 14 --starting_loc_y 14 --solver gurobi
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

import pulp


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_city(env_name, groups_file):
    from motndp.city import City
    return City(Path(f"envs/mo-tndp/cities/{env_name}"), groups_file=groups_file)


def build_adjacency(city):
    """4-connectivity adjacency from grid structure."""
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
# PuLP-based Solver (open source, works without license)
# ═══════════════════════════════════════════════════════════════════════════════

class PuLPSolver:
    """Build and solve MO-TNDP ILP with PuLP + CBC."""

    def __init__(self, city, adj, nr_stations, starting_loc, exact_path=False,
                 time_limit=600):
        self.city = city
        self.adj = adj
        self.K = nr_stations
        self.G = city.grid_size
        self.s = starting_loc
        self.exact_path = exact_path
        self.time_limit = time_limit
        self.n_groups = len(city.groups)

        # Precompute per-group non-zero OD pairs
        self.group_pairs = []
        for g in range(self.n_groups):
            self.group_pairs.append(
                get_nonzero_od_pairs(city.group_od_mx[g], self.G)
            )
        # Union of all OD pairs (for shared z variables)
        all_pairs_set = set()
        for pairs in self.group_pairs:
            all_pairs_set.update(pairs)
        self.all_pairs = sorted(all_pairs_set)

    def _build_base(self, name, sense=pulp.LpMaximize):
        """Build base ILP with selection + OD + (optional) path constraints."""
        prob = pulp.LpProblem(name, sense)

        # Cell selection
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(self.G)}

        # OD satisfaction
        z = {(i, j): pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
             for (i, j) in self.all_pairs}

        # Budget + start
        prob += pulp.lpSum(x[i] for i in range(self.G)) == self.K
        prob += x[self.s] == 1

        # McCormick linearization
        for (i, j) in self.all_pairs:
            prob += z[i, j] <= x[i]
            prob += z[i, j] <= x[j]
            prob += z[i, j] >= x[i] + x[j] - 1

        # Path connectivity via sequence selection
        if self.exact_path:
            y = {}
            for i in range(self.G):
                for t in range(self.K):
                    y[i, t] = pulp.LpVariable(f"y_{i}_{t}", cat="Binary")

            # Start at s
            prob += y[self.s, 0] == 1

            # One cell per position
            for t in range(self.K):
                prob += pulp.lpSum(y[i, t] for i in range(self.G)) == 1

            # Each cell at most once
            for i in range(self.G):
                prob += pulp.lpSum(y[i, t] for t in range(self.K)) <= 1

            # Adjacency: next station must be neighbor of current
            for t in range(self.K - 1):
                for i in range(self.G):
                    prob += (pulp.lpSum(y[j, t + 1] for j in self.adj[i])
                             >= y[i, t])

            # Link y to x
            for i in range(self.G):
                prob += x[i] == pulp.lpSum(y[i, t] for t in range(self.K))

        return prob, x, z

    def _group_return(self, z, g):
        """Linear expression for R_g = Σ od_g[i,j]*z_{ij} / total_g."""
        od_g = self.city.group_od_mx[g]
        total_g = self.city.group_od_sum[g]
        return pulp.lpSum(
            (od_g[i, j] + od_g[j, i]) * z[i, j]
            for (i, j) in self.group_pairs[g]
        ) / total_g

    def _solve(self, prob):
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))
        return pulp.LpStatus[prob.status], pulp.value(prob.objective)

    def _extract_returns(self, z):
        """Compute per-group returns from solved z variables."""
        returns = []
        for g in range(self.n_groups):
            od_g = self.city.group_od_mx[g]
            total_g = self.city.group_od_sum[g]
            val = sum(
                (od_g[i, j] + od_g[j, i])
                for (i, j) in self.group_pairs[g]
                if z[i, j].varValue and z[i, j].varValue > 0.5
            ) / total_g
            returns.append(val)
        return returns

    def _extract_cells(self, x):
        return sorted(i for i in range(self.G)
                       if x[i].varValue and x[i].varValue > 0.5)

    # ── Metric-specific solvers ──────────────────────────────────────────

    def solve_efficiency(self, maximize=True):
        """Efficiency = Σ_g R_g"""
        sense = pulp.LpMaximize if maximize else pulp.LpMinimize
        prob, x, z = self._build_base(f"Eff_{'max' if maximize else 'min'}", sense)
        prob += pulp.lpSum(self._group_return(z, g) for g in range(self.n_groups))
        return self._run(prob, x, z, "Efficiency", maximize)

    def solve_maxmin(self):
        """Max-Min Fairness: max min_g R_g"""
        prob, x, z = self._build_base("MaxMin", pulp.LpMaximize)
        t = pulp.LpVariable("t_floor", lowBound=0)
        for g in range(self.n_groups):
            prob += t <= self._group_return(z, g)
        prob += t
        return self._run(prob, x, z, "Max-Min", True)

    def solve_gini(self, minimize=True):
        """Gini index via absolute-difference linearization.

        Gini = Σ_{g<h} |R_g - R_h| / (n_groups * Σ_g R_g)

        Since the denominator depends on the solution, we use Charnes-Cooper:
        For minimizing Gini, we minimize the numerator while constraining
        total efficiency ≥ some threshold. For a simpler bound, we just
        minimize the absolute differences (numerator), which is valid when
        comparing solutions with similar efficiency.

        Exact approach: min numerator / denominator as fractional program.
        Practical approach: min numerator (= minimize absolute differences).
        """
        sense = pulp.LpMinimize if minimize else pulp.LpMaximize
        prob, x, z = self._build_base(f"Gini_{'min' if minimize else 'max'}", sense)

        R = {}
        for g in range(self.n_groups):
            R[g] = self._group_return(z, g)

        # Require nonzero efficiency (avoid trivial all-zero solution)
        eff_expr = pulp.lpSum(R[g] for g in range(self.n_groups))
        prob += eff_expr >= 1e-4

        # For max Gini: also cap efficiency to avoid unbounded numerator
        if not minimize:
            prob += eff_expr <= 10.0

        # |R_g - R_h| linearization
        # upBound=1.0 because R_g ∈ [0,1] so |R_g-R_h| ≤ 1
        d = {}
        obj_terms = []
        for g in range(self.n_groups):
            for h in range(g + 1, self.n_groups):
                d[g, h] = pulp.LpVariable(f"d_{g}_{h}", lowBound=0, upBound=1.0)
                prob += d[g, h] >= R[g] - R[h]
                prob += d[g, h] >= R[h] - R[g]
                obj_terms.append(d[g, h])

        prob += pulp.lpSum(obj_terms)
        status, obj_val = self._solve(prob)
        returns = self._extract_returns(z)
        cells = self._extract_cells(x)

        # Compute actual Gini from returns
        actual_gini = self._compute_gini(returns)

        return {
            "metric": "Gini",
            "direction": "min" if minimize else "max",
            "status": status,
            "abs_diff_sum": obj_val,
            "actual_gini": actual_gini,
            "returns": returns,
            "cells": cells,
        }

    def solve_sen_welfare(self, maximize=True):
        """Sen Welfare = Σ R_g - (1/n_groups) * Σ_{g<h} |R_g - R_h|

        This is LINEAR after expanding SW = E * (1-Gini).
        """
        sense = pulp.LpMaximize if maximize else pulp.LpMinimize
        prob, x, z = self._build_base(f"SW_{'max' if maximize else 'min'}", sense)

        R = {}
        for g in range(self.n_groups):
            R[g] = self._group_return(z, g)

        if not maximize:
            prob += pulp.lpSum(R[g] for g in range(self.n_groups)) >= 1e-6

        d = {}
        diff_terms = []
        for g in range(self.n_groups):
            for h in range(g + 1, self.n_groups):
                d[g, h] = pulp.LpVariable(f"d_{g}_{h}", lowBound=0, upBound=1.0)
                prob += d[g, h] >= R[g] - R[h]
                prob += d[g, h] >= R[h] - R[g]
                diff_terms.append(d[g, h])

        # SW = Σ R_g - (1/n) * Σ |R_g - R_h|
        n = self.n_groups
        prob += (pulp.lpSum(R[g] for g in range(n))
                 - (1.0 / n) * pulp.lpSum(diff_terms))

        return self._run(prob, x, z, "Sen Welfare", maximize)

    def solve_nash_welfare(self, maximize=True):
        """Nash Welfare = Π R_g

        Cannot directly linearize. Approximate by:
        max Σ_g log(R_g) → not supported in PuLP.

        Practical approach: weighted-sum sweep to find Nash-optimal point
        on the Pareto front, or use Gurobi for direct nonlinear.

        Fallback: solve weighted sum with weights proportional to 1/R_g
        (iterative reweighting converges to Nash optimal).
        """
        print("  [Nash Welfare requires nonlinear solver — using iterative reweighting]")

        # Start with equal weights
        weights = np.ones(self.n_groups) / self.n_groups
        best_nash = -1
        best_result = None

        for iteration in range(5):
            prob, x, z = self._build_base(f"Nash_iter{iteration}", pulp.LpMaximize)
            prob += pulp.lpSum(
                weights[g] * self._group_return(z, g)
                for g in range(self.n_groups)
            )
            self._solve(prob)
            returns = self._extract_returns(z)

            if all(r > 0 for r in returns):
                nash = np.prod(returns)
                if nash > best_nash:
                    best_nash = nash
                    best_result = {
                        "metric": "Nash Welfare",
                        "direction": "max",
                        "status": "Approximate",
                        "value": best_nash,
                        "returns": returns,
                        "cells": self._extract_cells(x),
                    }
                # Reweight: w_g ∝ 1/R_g (move toward proportional fairness)
                weights = np.array([1.0 / r for r in returns])
                weights /= weights.sum()

        return best_result

    def solve_per_group(self, group_idx, maximize=True):
        """Single group R_g."""
        sense = pulp.LpMaximize if maximize else pulp.LpMinimize
        prob, x, z = self._build_base(
            f"R{group_idx}_{'max' if maximize else 'min'}", sense)
        prob += self._group_return(z, group_idx)
        return self._run(prob, x, z, f"R_group{group_idx}", maximize)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _run(self, prob, x, z, metric_name, maximize):
        t0 = time.time()
        status, obj_val = self._solve(prob)
        elapsed = time.time() - t0
        returns = self._extract_returns(z)
        cells = self._extract_cells(x)

        return {
            "metric": metric_name,
            "direction": "max" if maximize else "min",
            "status": status,
            "value": obj_val,
            "returns": returns,
            "cells": cells,
            "time": elapsed,
        }

    @staticmethod
    def _compute_gini(returns):
        r = np.array(returns)
        n = len(r)
        if r.sum() == 0:
            return 0.0
        diffs = sum(abs(r[i] - r[j]) for i in range(n) for j in range(i + 1, n))
        return diffs / (n * r.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# Gurobi-based Solver (fast, supports nonlinear)
# ═══════════════════════════════════════════════════════════════════════════════

class GurobiSolver:
    """Build and solve MO-TNDP ILP with Gurobi."""

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

        self.group_pairs = []
        for g in range(self.n_groups):
            self.group_pairs.append(
                get_nonzero_od_pairs(city.group_od_mx[g], self.G_size)
            )
        all_pairs_set = set()
        for pairs in self.group_pairs:
            all_pairs_set.update(pairs)
        self.all_pairs = sorted(all_pairs_set)

    def _build_base(self, name):
        """Build base Gurobi model with selection + OD + (optional) path."""
        m = gp.Model(name)
        m.Params.TimeLimit = self.time_limit
        m.Params.OutputFlag = 0

        x = m.addVars(self.G_size, vtype=GRB.BINARY, name="x")
        z = m.addVars(self.all_pairs, vtype=GRB.BINARY, name="z")

        m.addConstr(gp.quicksum(x[i] for i in range(self.G_size)) == self.K)
        m.addConstr(x[self.s] == 1)

        for (i, j) in self.all_pairs:
            m.addConstr(z[i, j] <= x[i])
            m.addConstr(z[i, j] <= x[j])
            m.addConstr(z[i, j] >= x[i] + x[j] - 1)

        if self.exact_path:
            y = m.addVars(self.G_size, self.K, vtype=GRB.BINARY, name="y")
            m.addConstr(y[self.s, 0] == 1)
            for t in range(self.K):
                m.addConstr(gp.quicksum(y[i, t] for i in range(self.G_size)) == 1)
            for i in range(self.G_size):
                m.addConstr(gp.quicksum(y[i, t] for t in range(self.K)) <= 1)
            for t in range(self.K - 1):
                for i in range(self.G_size):
                    m.addConstr(
                        gp.quicksum(y[j, t + 1] for j in self.adj[i]) >= y[i, t]
                    )
            for i in range(self.G_size):
                m.addConstr(
                    x[i] == gp.quicksum(y[i, t] for t in range(self.K))
                )

        return m, x, z

    def _R(self, m, z, g):
        """Group return as Gurobi LinExpr."""
        od_g = self.city.group_od_mx[g]
        total_g = self.city.group_od_sum[g]
        return gp.quicksum(
            (od_g[i, j] + od_g[j, i]) * z[i, j]
            for (i, j) in self.group_pairs[g]
        ) / total_g

    def _extract(self, m, x, z):
        returns = []
        for g in range(self.n_groups):
            od_g = self.city.group_od_mx[g]
            total_g = self.city.group_od_sum[g]
            val = sum(
                (od_g[i, j] + od_g[j, i])
                for (i, j) in self.group_pairs[g]
                if z[i, j].X > 0.5
            ) / total_g
            returns.append(val)
        cells = sorted(i for i in range(self.G_size) if x[i].X > 0.5)
        return returns, cells

    def solve_efficiency(self, maximize=True):
        m, x, z = self._build_base("Efficiency")
        obj = gp.quicksum(self._R(m, z, g) for g in range(self.n_groups))
        m.setObjective(obj, GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
        return self._run(m, x, z, "Efficiency", maximize)

    def solve_per_group(self, group_idx, maximize=True):
        m, x, z = self._build_base(f"R{group_idx}_{'max' if maximize else 'min'}")
        m.setObjective(self._R(m, z, group_idx),
                       GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
        return self._run(m, x, z, f"R_group{group_idx}", maximize)

    def solve_maxmin(self):
        m, x, z = self._build_base("MaxMin")
        t = m.addVar(lb=0, name="t_floor")
        for g in range(self.n_groups):
            m.addConstr(t <= self._R(m, z, g))
        m.setObjective(t, GRB.MAXIMIZE)
        return self._run(m, x, z, "Max-Min", True)

    def solve_sen_welfare(self, maximize=True):
        m, x, z = self._build_base("SenWelfare")
        R = {g: self._R(m, z, g) for g in range(self.n_groups)}
        d = {}
        diff_terms = []
        for g in range(self.n_groups):
            for h in range(g + 1, self.n_groups):
                d[g, h] = m.addVar(lb=0, name=f"d_{g}_{h}")
                m.addConstr(d[g, h] >= R[g] - R[h])
                m.addConstr(d[g, h] >= R[h] - R[g])
                diff_terms.append(d[g, h])
        n = self.n_groups
        obj = (gp.quicksum(R[g] for g in range(n))
               - (1.0 / n) * gp.quicksum(diff_terms))
        m.setObjective(obj, GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
        return self._run(m, x, z, "Sen Welfare", maximize)

    def solve_gini(self, minimize=True):
        m, x, z = self._build_base("Gini")
        R = {g: self._R(m, z, g) for g in range(self.n_groups)}
        d = {}
        for g in range(self.n_groups):
            for h in range(g + 1, self.n_groups):
                d[g, h] = m.addVar(lb=0, name=f"d_{g}_{h}")
                m.addConstr(d[g, h] >= R[g] - R[h])
                m.addConstr(d[g, h] >= R[h] - R[g])
        obj = gp.quicksum(d[g, h] for g in range(self.n_groups)
                          for h in range(g + 1, self.n_groups))
        m.setObjective(obj, GRB.MINIMIZE if minimize else GRB.MAXIMIZE)
        m.optimize()
        returns, cells = self._extract(m, x, z) if m.Status == GRB.OPTIMAL else ([], [])
        gini_val = self._compute_gini(returns) if returns else None
        return {
            "metric": "Gini", "direction": "min" if minimize else "max",
            "status": "Optimal" if m.Status == GRB.OPTIMAL else str(m.Status),
            "actual_gini": gini_val, "returns": returns, "cells": cells,
            "time": m.Runtime,
        }

    def solve_nash_welfare(self, maximize=True):
        """Nash Welfare via log transform: max Σ log(R_g)."""
        m, x, z = self._build_base("Nash")

        R_vars = {}
        log_R = {}
        for g in range(self.n_groups):
            R_vars[g] = m.addVar(lb=1e-8, name=f"R_{g}")
            m.addConstr(R_vars[g] == self._R(m, z, g) + 1e-8)
            log_R[g] = m.addVar(lb=-GRB.INFINITY, name=f"logR_{g}")
            m.addGenConstrLog(R_vars[g], log_R[g])

        m.setObjective(
            gp.quicksum(log_R[g] for g in range(self.n_groups)),
            GRB.MAXIMIZE if maximize else GRB.MINIMIZE
        )
        m.Params.NonConvex = 2
        return self._run(m, x, z, "Nash Welfare", maximize)

    def _run(self, m, x, z, name, maximize):
        t0 = time.time()
        m.optimize()
        elapsed = time.time() - t0
        if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
            returns, cells = self._extract(m, x, z)
            return {
                "metric": name, "direction": "max" if maximize else "min",
                "status": "Optimal" if m.Status == GRB.OPTIMAL else "SubOptimal",
                "value": m.ObjVal, "returns": returns, "cells": cells,
                "time": elapsed,
            }
        return {
            "metric": name, "direction": "max" if maximize else "min",
            "status": str(m.Status), "value": None, "returns": [], "cells": [],
            "time": elapsed,
        }

    @staticmethod
    def _compute_gini(returns):
        r = np.array(returns)
        n = len(r)
        if r.sum() == 0:
            return 0.0
        diffs = sum(abs(r[i] - r[j]) for i in range(n) for j in range(i + 1, n))
        return diffs / (n * r.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(result):
    print(f"  Status: {result['status']}")
    if result.get('value') is not None:
        print(f"  Value: {result['value']:.8f}")
    if result.get('actual_gini') is not None:
        print(f"  Actual Gini: {result['actual_gini']:.8f}")
    if result.get('returns'):
        print(f"  Per-group returns: {[f'{r:.6f}' for r in result['returns']]}")
    if result.get('cells'):
        print(f"  Selected cells ({len(result['cells'])}): {result['cells']}")
    if result.get('time'):
        print(f"  Time: {result['time']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="ILP Bounds for MO-TNDP Metrics")
    parser.add_argument("--env", type=str, default="dilemma_5x5")
    parser.add_argument("--groups_file", type=str, default="groups.txt")
    parser.add_argument("--nr_stations", type=int, default=5)
    parser.add_argument("--starting_loc_x", type=int, default=2)
    parser.add_argument("--starting_loc_y", type=int, default=2)
    parser.add_argument("--exact_path", action="store_true",
                        help="Add sequence-based path connectivity constraints")
    parser.add_argument("--solver", type=str, default="pulp", choices=["pulp", "gurobi"])
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["efficiency", "maxmin", "gini", "sen", "nash"],
                        help="Which metrics to compute bounds for")
    args = parser.parse_args()

    city = load_city(args.env, args.groups_file)
    adj = build_adjacency(city)
    starting_loc = args.starting_loc_x * city.grid_y_size + args.starting_loc_y

    print(f"{'='*60}")
    print(f"MO-TNDP Theoretical Bounds")
    print(f"{'='*60}")
    print(f"Env: {args.env} ({city.grid_x_size}x{city.grid_y_size} = {city.grid_size} cells)")
    print(f"Groups: {len(city.groups)}, Stations: {args.nr_stations}")
    print(f"Start: ({args.starting_loc_x},{args.starting_loc_y}) = cell {starting_loc}")
    print(f"Path: {'EXACT (sequence)' if args.exact_path else 'RELAXED (no connectivity)'}")
    print(f"Solver: {args.solver.upper()}")
    print(f"{'='*60}")

    if args.solver == "gurobi":
        if not HAS_GUROBI:
            print("ERROR: Gurobi not available. Use --solver pulp")
            return
        solver = GurobiSolver(city, adj, args.nr_stations, starting_loc,
                              args.exact_path, args.time_limit)
    else:
        solver = PuLPSolver(city, adj, args.nr_stations, starting_loc,
                            args.exact_path, args.time_limit)

    results = {}

    # Per-group bounds (utopia point)
    print(f"\n{'─'*60}")
    print("Per-Group Upper Bounds (Utopia Point)")
    print(f"{'─'*60}")
    utopia = []
    for g in range(len(city.groups)):
        print(f"\n  ▸ Group {int(city.groups[g])} UB:")
        r = solver.solve_per_group(g, maximize=True)
        print_result(r)
        utopia.append(r.get('value'))
    results['utopia'] = utopia

    # Metric bounds
    for metric in args.metrics:
        print(f"\n{'─'*60}")

        if metric == "efficiency":
            print("Efficiency = Σ R_g")
            print(f"{'─'*60}")
            print("\n  ▸ Upper Bound (max efficiency):")
            r = solver.solve_efficiency(maximize=True)
            print_result(r)
            results['efficiency_ub'] = r
            print("\n  ▸ Lower Bound (min efficiency):")
            r = solver.solve_efficiency(maximize=False)
            print_result(r)
            results['efficiency_lb'] = r

        elif metric == "maxmin":
            print("Max-Min Fairness = max min_g R_g")
            print(f"{'─'*60}")
            print("\n  ▸ Upper Bound (fairest possible):")
            r = solver.solve_maxmin()
            print_result(r)
            results['maxmin_ub'] = r

        elif metric == "gini":
            print("Gini Index")
            print(f"{'─'*60}")
            print("\n  ▸ Lower Bound (min Gini = most equal):")
            r = solver.solve_gini(minimize=True)
            print_result(r)
            results['gini_lb'] = r
            print("\n  ▸ Upper Bound (max Gini = most unequal):")
            r = solver.solve_gini(minimize=False)
            print_result(r)
            results['gini_ub'] = r

        elif metric == "sen":
            print("Sen Welfare = Σ R_g - (1/n)·Σ|R_g - R_h|")
            print(f"{'─'*60}")
            print("\n  ▸ Upper Bound (best Sen Welfare):")
            r = solver.solve_sen_welfare(maximize=True)
            print_result(r)
            results['sen_ub'] = r
            print("\n  ▸ Lower Bound (worst Sen Welfare):")
            r = solver.solve_sen_welfare(maximize=False)
            print_result(r)
            results['sen_lb'] = r

        elif metric == "nash":
            print("Nash Welfare = Π R_g")
            print(f"{'─'*60}")
            print("\n  ▸ Upper Bound (best Nash Welfare):")
            r = solver.solve_nash_welfare(maximize=True)
            print_result(r)
            results['nash_ub'] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Path: {'EXACT' if args.exact_path else 'RELAXED'}")
    if utopia:
        print(f"Utopia: {[f'{v:.6f}' if v else 'N/A' for v in utopia]}")
    for key in ['efficiency_ub', 'efficiency_lb', 'maxmin_ub', 'sen_ub', 'sen_lb']:
        if key in results and results[key].get('value') is not None:
            print(f"{key}: {results[key]['value']:.8f}")
    for key in ['gini_lb', 'gini_ub']:
        if key in results and results[key].get('actual_gini') is not None:
            print(f"{key}: {results[key]['actual_gini']:.8f}")
    if 'nash_ub' in results and results['nash_ub'].get('value') is not None:
        print(f"nash_ub: {results['nash_ub']['value']:.8f}")


if __name__ == "__main__":
    main()
