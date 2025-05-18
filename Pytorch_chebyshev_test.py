#!/usr/bin/env python3
# PyTorch-based Tensor-Train completion  +  Chebyshev interpolation
# ============================================================================
# 2025-05-19 • Complete version
# ============================================================================

from __future__ import annotations
from typing import List, Sequence, Tuple, Dict, Any, Optional
from itertools import product
import math, random, time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# helper: convert anything → torch tensor on the right device / dtype
# ---------------------------------------------------------------------------
def _to_tensor(x, *, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# 1.  TTCompletion  (batch rank-increase fit)
# ---------------------------------------------------------------------------
class TTCompletion(nn.Module):
    """Tensor-Train completion with batch rank-increase training loop."""

    # ---------------------------------------------------------------------
    # construction
    # ---------------------------------------------------------------------
    def __init__(
        self,
        shape: Sequence[int],
        initial_tt_ranks: Optional[Sequence[int]] = None,
        *,
        max_rank: int = 30,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.shape = list(map(int, shape))
        self.ndim  = len(self.shape)
        self.max_rank = int(max_rank)
        self.device = torch.device(device)
        self.dtype  = dtype

        # rank vector  [1, R1, …, R_{d-1}, 1]
        if initial_tt_ranks is None:
            ranks = [1] + [2]*(self.ndim-1) + [1]
        else:
            if len(initial_tt_ranks) != self.ndim-1:
                raise ValueError("initial_tt_ranks length mismatch")
            ranks = [1] + [int(r) for r in initial_tt_ranks] + [1]
        self.tt_ranks: List[int] = ranks

        # learnable TT cores  (r_{k-1}, n_k, r_k)
        self.tt_cores = nn.ParameterList()
        for k, n_k in enumerate(self.shape):
            rL, rR = self.tt_ranks[k], self.tt_ranks[k+1]
            core = torch.randn(rL, n_k, rR, dtype=dtype, device=self.device)
            core /= math.sqrt(max(rL, rR))
            self.tt_cores.append(nn.Parameter(core))

        # normalisation statistics (filled during fit)
        self.values_mean: float | torch.Tensor | None = None
        self.values_std:  float | torch.Tensor | None = None

    # ---------------------------------------------------------------------
    # forward: vectorised contraction over a batch of indices
    # ---------------------------------------------------------------------
    def forward(self, idx: torch.LongTensor) -> torch.Tensor:  # type: ignore[override]
        B, d = idx.shape
        if d != self.ndim:
            raise ValueError("idx shape mismatch")
        v = torch.ones(B, 1, dtype=self.dtype, device=self.device)
        for k in range(self.ndim):
            # permute core to (n_k, r_left, r_right)
            G = self.tt_cores[k].permute(1, 0, 2)
            sel = G[idx[:, k]]                   # (B, rL, rR)
            v   = torch.bmm(v.unsqueeze(1), sel).squeeze(1)
        return v.squeeze(-1)                     # (B,)

    # convenience
    def predict(self, coords, *, normalize=False):
        pred = self(_to_tensor(coords, dtype=torch.long, device=self.device))
        if not normalize and self.values_mean is not None:
            pred = pred * self.values_std + self.values_mean
        return pred.detach().cpu().numpy()

    # ---------------------------------------------------------------------
    # _optimize_model: SGD for a fixed rank configuration
    # ---------------------------------------------------------------------
    def _optimize_model(
        self,
        coords_np: np.ndarray,
        vals_norm_np: np.ndarray,
        max_iter: int,
        tol: float,
        verbose: bool,
        batch_size: int = 4096,
        lr: float = 1e-2,
    ):
        coords_t = _to_tensor(coords_np, dtype=torch.long, device=self.device)
        vals_t   = _to_tensor(vals_norm_np, dtype=self.dtype, device=self.device)

        loader = DataLoader(
            TensorDataset(coords_t.cpu(), vals_t.cpu()),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        mse = nn.MSELoss()
        prev_rmse = None

        for it in range(1, max_iter + 1):
            self.train()
            for bc, bv in loader:
                bc = bc.to(self.device, non_blocking=True)
                bv = bv.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = mse(self(bc), bv)
                loss.backward()
                opt.step()

            with torch.no_grad():
                rmse = torch.sqrt(mse(self(coords_t), vals_t)).item()

            if verbose:
                print(f"    iter {it:02d}  RMSE {rmse:.6f}")

            if prev_rmse is not None and abs(prev_rmse - rmse) / prev_rmse < tol:
                break
            prev_rmse = rmse

    # ---------------------------------------------------------------------
    # fit: batch rank-increase loop exactly as requested
    # ---------------------------------------------------------------------
    def fit(self, coords, values, *,
            max_iter_per_rank: int = 40,   # SGD iters *after* each rank change
            burn_in: int = 10,             # warm-up iters before measuring
            tol: float = 1e-4,
            max_time: float = 600,
            verbose: bool = True,
            validation_split: float = 0.2,
            rank_increase_tol: float = 0.01,   # NOT used now
            abs_tol: float = 1e-4,             # new absolute progress test
            ):
        # ----------  split + normalise exactly as before ----------
        n = len(values)
        if coords.shape[0] != n:
            raise ValueError("coords/values size mismatch")
        n_val = int(n * validation_split)
        perm  = np.random.permutation(n)
        c_val, c_train = coords[perm[:n_val]], coords[perm[n_val:]]
        v_val, v_train = values[perm[:n_val]], values[perm[n_val:]]

        self.values_mean = v_train.mean()
        self.values_std  = v_train.std() if v_train.std() > 0 else 1.0
        n_train = (v_train - self.values_mean) / self.values_std
        n_val   = (v_val   - self.values_mean) / self.values_std

        def _rmse(coords_np, vals_np, norm=False):
            p = self.predict(coords_np, normalize=norm)
            return math.sqrt(((p - vals_np) ** 2).mean())

        history = {"train_rmse": [], "val_rmse": [], "time": [], "ranks": []}
        t0 = time.time()

        # ---------- helper that runs k optimisation iters ----------
        def _opt(k, verbose = True):
            self._optimize_model(c_train, n_train, k, tol/10, verbose)

        if verbose: print("Initial optimisation …")
        _opt(max_iter_per_rank, verbose)
        best_val_norm = _rmse(c_val, n_val, norm=True)

        history["train_rmse"].append(_rmse(c_train, v_train))
        history["val_rmse"].append(_rmse(c_val, v_val))
        history["time"].append(time.time() - t0)
        history["ranks"].append(self.tt_ranks.copy())

        rank_iter = 0
        while rank_iter < 10 and (time.time() - t0) < max_time:
            rank_iter += 1
            if verbose:
                print(f"\nRank increase {rank_iter}")

            # backup model
            cores_bak = [c.detach().clone() for c in self.tt_cores]
            ranks_bak = self.tt_ranks.copy()

            grew = False
            for bond in range(1, self.ndim):
                if self.tt_ranks[bond] < self.max_rank:
                    self._increase_bond_rank(bond, self.tt_ranks[bond] + 1)
                    grew = True
            if not grew:
                if verbose: print("All bonds already at max_rank."); break

            # -------- burn-in then full optimisation --------
            _opt(burn_in, verbose=False)              # warm-up
            val_norm_burn = _rmse(c_val, n_val, norm=True)

            _opt(max_iter_per_rank, verbose)          # full train
            val_norm_new  = _rmse(c_val, n_val, norm=True)

            improvement = best_val_norm - val_norm_new
            if verbose:
                print(f"  burn-in RMSE {val_norm_burn:.6f}  "
                      f"after-train RMSE {val_norm_new:.6f}  "
                      f"abs Δ {improvement:.3e}")

            if improvement > abs_tol:
                # accept
                best_val_norm = val_norm_new
                history["train_rmse"].append(_rmse(c_train, v_train))
                history["val_rmse"].append(_rmse(c_val, v_val))
                history["time"].append(time.time() - t0)
                history["ranks"].append(self.tt_ranks.copy())
            else:
                # revert
                if verbose: print("  rejected, reverting.")
                for k in range(self.ndim):
                    self.tt_cores[k] = nn.Parameter(cores_bak[k])
                self.tt_ranks = ranks_bak
                break   # stop trying larger ranks if even abs_tol failed

        if verbose: print("\nFinal polishing …")
        _opt(max_iter_per_rank * 2, verbose)

        history["train_rmse"].append(_rmse(c_train, v_train))
        history["val_rmse"].append(_rmse(c_val, v_val))
        history["time"].append(time.time() - t0)
        history["ranks"].append(self.tt_ranks.copy())
        if verbose:
            rstr = "-".join(map(str, self.tt_ranks[1:-1]))
            print(f"Final ranks [{rstr}]  "
                  f"Train RMSE {history['train_rmse'][-1]:.4f}  "
                  f"Val RMSE {history['val_rmse'][-1]:.4f}")
        return history

    # ---------------------------------------------------------------------
    # helper: pad two adjacent cores to increase a bond rank
    # ---------------------------------------------------------------------
    def _increase_bond_rank(
        self, bond_idx: int, new_rank: int, *, noise_scale: float = 1e-2
    ):
        if new_rank <= self.tt_ranks[bond_idx] or new_rank > self.max_rank:
            return
        k = bond_idx - 1  # left core index
        left  = self.tt_cores[k]
        right = self.tt_cores[k + 1]

        r_prev, n_k, r_k     = left.shape
        r_k2,   n_k1, r_next = right.shape
        assert r_k == r_k2

        pad_left  = noise_scale * torch.randn(
            r_prev, n_k, new_rank - r_k, dtype=self.dtype, device=self.device
        )
        pad_right = noise_scale * torch.randn(
            new_rank - r_k, n_k1, r_next, dtype=self.dtype, device=self.device
        )

        self.tt_cores[k]     = nn.Parameter(
            torch.cat([left.data, pad_left], dim=2)
        )
        self.tt_cores[k + 1] = nn.Parameter(
            torch.cat([pad_right, right.data], dim=0)
        )
        self.tt_ranks[bond_idx] = new_rank

    # ---------------------------------------------------------------------
    # core utilities (needed by Chebyshev)
    # ---------------------------------------------------------------------
    def tt_mode_multiply(self, matrix, mode: int):
        """Multiply in place by a matrix along *mode* (for Chebyshev)."""
        M = _to_tensor(matrix, dtype=self.dtype, device=self.device)
        core = self.tt_cores[mode]
        rl, nk, rr = core.shape
        if M.shape[1] != nk:
            raise ValueError("matrix second dimension mismatch")

        core_mat = core.permute(1, 0, 2).reshape(nk, rl * rr)        # (nk, rl*rr)
        new_core = (M @ core_mat).reshape(M.size(0), rl, rr)         # (m, rl, rr)
        new_core = new_core.permute(1, 0, 2).contiguous()            # (rl, m, rr)

        self.tt_cores[mode] = nn.Parameter(new_core)
        self.shape[mode] = M.size(0)
        return self

    def copy(self) -> "TTCompletion":
        dup = TTCompletion(
            self.shape,
            self.tt_ranks[1:-1],
            max_rank=self.max_rank,
            device=self.device,
            dtype=self.dtype,
        )
        for k in range(self.ndim):
            dup.tt_cores[k].data.copy_(self.tt_cores[k].data)
        dup.values_mean = self.values_mean
        dup.values_std  = self.values_std
        return dup


# ---------------------------------------------------------------------------
# 2.  TTChebyshevInterpolation
# ---------------------------------------------------------------------------
class TTChebyshevInterpolation:
    """Multidimensional Chebyshev interpolation backed by TTCompletion."""

    # ---------------------------------------------------------------------
    def __init__(
        self,
        domains: Sequence[Tuple[float, float]],
        degrees: Sequence[int],
        *,
        initial_tt_ranks: Optional[Sequence[int]] = None,
        max_rank: int = 10,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.domains = list(domains)
        self.degrees = list(degrees)
        self.d       = len(domains)

        if initial_tt_ranks is None:
            initial_tt_ranks = [2] * (self.d - 1)
        self.initial_tt_ranks = initial_tt_ranks
        self.max_rank = max_rank
        self.device   = torch.device(device)
        self.dtype    = dtype

        # Chebyshev nodes in each dimension
        self.nodes: List[np.ndarray] = []
        for (a, b), N in zip(self.domains, self.degrees):
            k = np.arange(N + 1)
            z = np.cos(np.pi * k / N)             # in [-1, 1]
            x = 0.5 * (b - a) * (z + 1) + a       # mapped to [a, b]
            self.nodes.append(x)

        self.tensor_shape = tuple(d + 1 for d in self.degrees)

        self.P_tt: TTCompletion | None = None
        self.C_tt: TTCompletion | None = None

        self._T_cache: Dict[Tuple[int, float], float] = {}

    # ---------------------------------------------------------------------
    # polynomial helper  T_n(x) with caching
    # ---------------------------------------------------------------------
    def _T(self, n: int, x: float) -> float:
        key = (n, float(x))
        if key in self._T_cache:
            return self._T_cache[key]
        val = math.cos(n * math.acos(x)) if abs(x) <= 1 else math.cosh(
            n * math.acosh(x)
        )
        if len(self._T_cache) < 10000:
            self._T_cache[key] = val
        return val

    # discrete cosine transform matrix F_n
    def _F(self, n: int):
        j, k = np.indices((n + 1, n + 1))
        M = np.cos(j * k * np.pi / n) * (2.0 / n)
        M[[0, n], :] *= 0.5
        M[:, [0, n]] *= 0.5
        return _to_tensor(M, dtype=self.dtype, device=self.device)

    # ---------------------------------------------------------------------
    # OFFLINE PHASE
    # ---------------------------------------------------------------------
    def run_offline_phase(
        self, reference_method, *, subset_size: Optional[int] = None, verbose=True
    ):
        """Construct P-tensor then coefficient tensor C = (⊗F)·P."""
        # ---------- sample points ----------
        grid_all = list(product(*[range(d + 1) for d in self.degrees]))
        if subset_size and subset_size < len(grid_all):
            grid = random.sample(grid_all, subset_size)
        else:
            grid = grid_all

        coords = np.array(grid, int)
        vals = np.empty(len(grid), float)
        for i, idx in enumerate(coords):
            pt = tuple(self.nodes[dim][j] for dim, j in enumerate(idx))
            vals[i] = reference_method(pt)
            if verbose and (i + 1) % 500 == 0:
                print(f"  evaluated {i + 1}/{len(grid)} points")

        # ---------- build P ----------
        if verbose:
            print("Fitting TT tensor P …")
        self.P_tt = TTCompletion(
            self.tensor_shape,
            self.initial_tt_ranks,
            max_rank=self.max_rank,
            device=self.device,
            dtype=self.dtype,
        )
        self.P_tt.fit(coords, vals, verbose=verbose)

        # ---------- build C ----------
        if verbose:
            print("Constructing coefficient tensor C …")
        self.C_tt = self.P_tt.copy()
        for mode, deg in enumerate(self.degrees):
            if verbose:
                print(f"  mode {mode} multiply by F_{deg}")
            self.C_tt.tt_mode_multiply(self._F(deg), mode)

        return self.C_tt

    # ---------------------------------------------------------------------
    # ONLINE PHASE
    # ---------------------------------------------------------------------
    def run_online_phase(self, points):
        """Evaluate the Chebyshev interpolant at a list/array of points."""
        if self.C_tt is None:
            raise ValueError("Offline phase not run yet")

        if isinstance(points, tuple):
            points = [points]
        results = np.empty(len(points), float)

        # precompute Chebyshev vectors for each point
        for idx_pt, pt in enumerate(points):
            vecs = []
            for (a, b), deg, x in zip(self.domains, self.degrees, pt):
                z = 2 * (x - a) / (b - a) - 1
                vec = torch.tensor(
                    [self._T(k, z) for k in range(deg + 1)],
                    dtype=self.dtype,
                    device=self.device,
                )
                vecs.append(vec)

            # TT inner product <C, ⊗ vecs>
            res = torch.ones(1, 1, dtype=self.dtype, device=self.device)
            for m, v in enumerate(vecs):
                res = res @ torch.tensordot(
                    self.C_tt.tt_cores[m], v, dims=([1], [0])
                )
            val = res.squeeze()

            if self.C_tt.values_mean is not None:
                val = val * self.C_tt.values_std + self.C_tt.values_mean
            results[idx_pt] = float(val)

        return results


# ---------------------------------------------------------------------------
# 3.  Black-Scholes 2-D demo
# ---------------------------------------------------------------------------
# ======================================================================
# 5-D Black-Scholes convergence test
#    Varies five inputs:  S0,  σ,  strike K,  rate r,  maturity T
# ======================================================================
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # ---------- 1. 5-D reference pricing function ----------
    def bs_call_price(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    def reference_5d(params):
        S0, sigma, K, r, T = params
        return bs_call_price(S0, K, T, r, sigma)

    # ---------- 2. domains (5D) & polynomial degrees ----------
    domains = [
        (80.0, 120.0),   # S0
        (0.10, 0.40),    # sigma
        (90.0, 110.0),   # strike K
        (0.01, 0.10),    # rate r
        (0.5,  2.0),     # maturity T
    ]
    degrees = [10, 10, 10, 10, 10]  # Chebyshev degree in each dimension
    # (40×30×30×20×20 ≈ 14.4 M coefficients – we’ll only sample a tiny subset)

    # ---------- 3. build TT-Chebyshev interpolant ----------
    tt_cheb = TTChebyshevInterpolation(
        domains,
        degrees,
        initial_tt_ranks=[3] * 4,    # start low
        max_rank=25,                 # allow growth up to rank-15
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # offline: sample 0.15 % of the full tensor  (≈ 22 k points)
    full_size  = np.prod([d + 1 for d in degrees])
    subset_pts = int(0.1 * full_size)
    print(f"Offline training on {subset_pts:,} of {full_size:,} tensor points …")
    tt_cheb.run_offline_phase(reference_5d, subset_size=subset_pts, verbose=True)

    # ---------- 4. evaluate on a fresh Monte-Carlo test set ----------
    n_test = 10_000
    rng = np.random.default_rng(0)
    test = np.column_stack(
        [
            rng.uniform(a, b, size=n_test)    # S0
            for (a, b) in domains
        ]
    )
    # transpose domain sampling order
    test[:, 0] = rng.uniform(*domains[0], n_test)   # S0
    test[:, 1] = rng.uniform(*domains[1], n_test)   # sigma
    test[:, 2] = rng.uniform(*domains[2], n_test)   # K
    test[:, 3] = rng.uniform(*domains[3], n_test)   # r
    test[:, 4] = rng.uniform(*domains[4], n_test)   # T

    interp_vals = tt_cheb.run_online_phase([tuple(p) for p in test])
    exact_vals  = np.array([reference_5d(tuple(p)) for p in test])

    rel_err = np.abs(interp_vals - exact_vals) / exact_vals
    print("\n5-D test results")
    print(f"max  relative error : {rel_err.max():.3e}")
    print(f"mean relative error : {rel_err.mean():.3e}")
    print("final TT ranks      :", tt_cheb.C_tt.tt_ranks)

    # ---------- 5. error histogram ----------
    plt.hist(rel_err, bins=50, log=True, edgecolor='k')
    plt.xlabel("relative error")
    plt.ylabel("count (log-scale)")
    plt.title("Distribution of 5-D TT-Chebyshev interpolation errors")
    plt.show()
