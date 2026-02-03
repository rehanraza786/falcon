from __future__ import annotations

import time
import pulp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from falcon.models import NLIJudge


@dataclass
class SolveResult:
    selected_indices: List[int]
    x_values: List[int]
    objective_value: float
    solve_seconds: float


class FalconSolver:
    def __init__(self, nli_judge: NLIJudge, tau: float = 0.7, mode: str = "hard",
                 lambda_penalty: float = 1.0, max_pairwise: int = 496):
        self.nli = nli_judge
        self.tau = float(tau)
        self.mode = mode
        self.lambda_penalty = float(lambda_penalty)
        self.max_pairwise = int(max_pairwise)

    def solve(self, claims: List[str], weights: Optional[List[float]],
              P: Dict[Tuple[int, int], float]) -> SolveResult:
        n = len(claims)
        weights = weights or [1.0] * n
        t0 = time.time()

        # Binary decision variables
        x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(n)]
        prob = pulp.LpProblem("FALCON_Optimization", pulp.LpMaximize)

        if self.mode == "hard":
            # Maximize: sum(w_i * x_i)
            prob += pulp.lpSum([weights[i] * x[i] for i in range(n)])
            for (i, j), pij in P.items():
                if pij > self.tau:
                    # Constraint: x_i + x_j <= 1 (cannot keep both)
                    prob += x[i] + x[j] <= 1
        else:
            # Soft Logic with McCormick Linearization
            z = {}
            for (i, j), pij in P.items():
                zij = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1)
                z[(i, j)] = zij
                # McCormick Envelopes
                prob += zij <= x[i]
                prob += zij <= x[j]
                prob += zij >= x[i] + x[j] - 1

            # Maximize: sum(w_i * x_i) - lambda * sum(P_ij * z_ij)
            prob += pulp.lpSum([weights[i] * x[i] for i in range(n)]) - \
                    self.lambda_penalty * pulp.lpSum([P[(i, j)] * z[(i, j)] for (i, j) in P])

        solver_cmd = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver_cmd)

        x_vals = [int(pulp.value(var)) for var in x]
        selected_indices = [i for i, val in enumerate(x_vals) if val == 1]

        t1 = time.time()
        return SolveResult(
            selected_indices=selected_indices,
            x_values=x_vals,
            objective_value=pulp.value(prob.objective),
            solve_seconds=t1 - t0
        )