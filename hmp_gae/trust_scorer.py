# hmp_gae/trust_scorer.py
# Closed-form trust scoring for HMP-GAE.
#
# Trust score s_i combines two structural signals:
#   1. residual_i  : how "off-cluster" node i is in the reconstructed graph,
#                    measured as 1 - mean_{j != i} A_hat_ij. A well-connected
#                    node (many high-similarity neighbors) gets a low residual.
#   2. hist_dev_i  : how far the current embedding z_i has drifted from its
#                    own EMA history z_hist_i, measured as ||z_i - z_hist_i||_2.
#
# Each signal is z-score normalized across the batch so they contribute on a
# comparable scale, then combined:
#     s_i = - ( alpha_residual * z(residual_i) + beta_hist * z(hist_dev_i) )
# and finally turned into aggregation weights via softmax:
#     alpha_i = softmax( s_i / tau ).
#
# Rationale:
#   - No MLP supervision needed (avoids the "no training signal" pitfall of
#     the paper's MLP trust head).
#   - tau -> 0   reproduces Krum-like hard selection,
#     tau -> inf reproduces uniform averaging,
#     tau in [0.05, 0.5] gives smooth soft-rejection in practice.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class TrustResult:
    alpha: torch.Tensor          # (N,) non-negative weights summing to 1
    s: torch.Tensor              # (N,) trust logits
    # Signal 1: graph-structural residual from the hypergraph incidence H.
    # High residual = this node is included in few hyperedges (isolated).
    graph_residual: torch.Tensor        # (N,) in [0, 1]
    graph_residual_z: torch.Tensor      # z-scored graph_residual
    # Signal 2: decoder-based residual from the reconstructed A_hat.
    # High residual = low average similarity to other nodes in learned
    # latent space. Noisy until the encoder is sufficiently trained.
    recon_residual: torch.Tensor        # (N,)
    recon_residual_z: torch.Tensor      # z-scored recon_residual
    # Signal 3: historical deviation (disabled by default in V1 because
    # benign clients drift more than attackers during real learning).
    hist_dev: torch.Tensor              # (N,)
    hist_dev_z: torch.Tensor            # z-scored hist_dev


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mean = x.mean()
    std = x.std(unbiased=False).clamp(min=eps)
    return (x - mean) / std


def compute_trust_weights(
    A_hat: torch.Tensor,
    Z: torch.Tensor,
    Z_hist: Optional[torch.Tensor],
    H: Optional[torch.Tensor] = None,
    graph_weight: float = 1.0,
    residual_weight_alpha: float = 0.3,
    hist_weight_beta: float = 0.0,
    softmax_tau: float = 0.1,
    min_alpha_clip: float = 1e-6,
) -> TrustResult:
    """
    Compute closed-form trust weights for N clients.

    Combines three signals (each z-scored for scale invariance):

        s_i = - ( graph_weight           * z(graph_residual_i)
                + residual_weight_alpha  * z(recon_residual_i)
                + hist_weight_beta       * z(hist_dev_i) )

    graph_residual uses only the deterministic k-NN hypergraph incidence H
    (so it is robust even when the HMP encoder is only partially trained).
    recon_residual uses the learned A_hat (informative once encoder has
    converged). hist_dev is included for completeness but defaults to weight
    zero -- benign clients learning from data drift more than attackers
    trapped on a fixed mislabel manifold, which can invert the signal.

    Args:
        A_hat:  (N, N) reconstructed adjacency in [0, 1].
        Z:      (N, d) latent embeddings from the HMP encoder.
        Z_hist: (N, d) EMA history embeddings (None on cold start).
        H:      (N, M) incidence matrix (optional; required for graph signal).

    Returns:
        TrustResult with alpha (N,) and diagnostic tensors.
    """
    N = A_hat.shape[0]
    device = A_hat.device
    dtype = A_hat.dtype

    if N == 0:
        empty = torch.zeros(0, device=device, dtype=dtype)
        return TrustResult(
            alpha=empty, s=empty,
            graph_residual=empty, graph_residual_z=empty,
            recon_residual=empty, recon_residual_z=empty,
            hist_dev=empty, hist_dev_z=empty,
        )

    # ---- Signal 1: graph residual from hypergraph incidence H ---- #
    # A node with low "reach" across hyperedges is isolated/anomalous.
    # Specifically, we measure how many other nodes share at least one
    # hyperedge with node i: reach_i = (H H^T)[i, :] count > 0.
    # Normalized to [0, 1]: graph_residual_i = 1 - reach_i / (N - 1).
    if H is not None and N > 1:
        # co_membership[i, j] = #hyperedges shared between i and j.
        co = (H @ H.t())                           # (N, N)
        co.fill_diagonal_(0.0)
        reach = (co > 0).to(dtype).sum(dim=1)      # (N,) # peers touched
        graph_residual = 1.0 - reach / max(1, N - 1)
    else:
        graph_residual = torch.zeros(N, device=device, dtype=dtype)

    # ---- Signal 2: reconstructed adjacency residual ---- #
    off_mask = 1.0 - torch.eye(N, device=device, dtype=dtype)
    if N > 1:
        recon_residual = 1.0 - (A_hat * off_mask).sum(dim=1) / (N - 1)
    else:
        recon_residual = torch.zeros(N, device=device, dtype=dtype)

    # ---- Signal 3: historical deviation ---- #
    if Z_hist is None:
        hist_dev = torch.zeros(N, device=device, dtype=dtype)
        use_hist = False
    else:
        Z_hist_d = Z_hist.detach().to(device=device, dtype=dtype)
        hist_dev = (Z - Z_hist_d).norm(dim=1)
        use_hist = True

    graph_residual_z = _zscore(graph_residual)
    recon_residual_z = _zscore(recon_residual)
    hist_dev_z = _zscore(hist_dev) if use_hist else torch.zeros_like(hist_dev)

    s = -(
        graph_weight * graph_residual_z
        + residual_weight_alpha * recon_residual_z
        + hist_weight_beta * hist_dev_z
    )

    tau = max(float(softmax_tau), 1e-4)
    alpha = torch.softmax(s / tau, dim=0)
    if min_alpha_clip > 0:
        alpha = alpha.clamp(min=min_alpha_clip)
        alpha = alpha / alpha.sum()

    return TrustResult(
        alpha=alpha, s=s,
        graph_residual=graph_residual,
        graph_residual_z=graph_residual_z,
        recon_residual=recon_residual,
        recon_residual_z=recon_residual_z,
        hist_dev=hist_dev,
        hist_dev_z=hist_dev_z,
    )


def weighted_aggregate(updates, alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute sum_i alpha_i * update_i with shape-robust accumulation.

    Works with either a list of 1-D tensors or a (N, D) stacked tensor.
    """
    if isinstance(updates, list):
        stacked = torch.stack(updates)
    else:
        stacked = updates
    stacked = stacked.to(device=alpha.device, dtype=alpha.dtype)
    return (stacked * alpha.view(-1, 1)).sum(dim=0)


def reject_then_weighted(
    trust: "TrustResult",
    data_sizes: torch.Tensor,
    reject_z_threshold: float = 1.0,
    keep_min: int = 1,
) -> torch.Tensor:
    """
    Hybrid aggregation: use HMP-GAE trust signals to *detect* attackers,
    then fall back to data-size-weighted FedAvg among trusted clients.

    Rationale: the softmax on trust logits is great at flagging outliers
    but tends to concentrate weight on 1-2 benign clients when benign
    graph_residual values are nearly tied -- this wastes the collaborative
    learning benefit. Splitting detection from weighting gives both:
      1. attacker contributions are zeroed out,
      2. benign contributions aggregate at their natural data-size weights.

    Detection rule: a client i is rejected when graph_residual_z_i exceeds
    `reject_z_threshold` (default 1.0, i.e. > 1 sigma above mean isolation).
    `keep_min` guarantees at least k clients are kept even in degenerate cases.
    """
    device = trust.alpha.device
    dtype = trust.alpha.dtype
    N = trust.alpha.numel()

    gr_z = trust.graph_residual_z.detach().clone()
    mask = gr_z <= float(reject_z_threshold)

    if int(mask.sum().item()) < max(1, keep_min):
        # Too aggressive: keep keep_min most-trusted by lowest gr_z.
        k = max(1, keep_min)
        idx = torch.topk(-gr_z, k=min(k, N)).indices
        mask = torch.zeros(N, device=device, dtype=torch.bool)
        mask[idx] = True

    ds = data_sizes.to(device=device, dtype=dtype) * mask.to(dtype)
    total = ds.sum()
    if total.item() <= 0:
        # Fallback: uniform over kept clients.
        uniform = mask.to(dtype)
        ds = uniform
        total = ds.sum().clamp(min=1.0)
    return ds / total


def reject_soft_weighted(
    trust: "TrustResult",
    data_sizes: torch.Tensor,
    reject_z_threshold: float = 0.75,
    soft_reject_k: float = 2.0,
    keep_min: int = 1,
) -> torch.Tensor:
    """
    Soft-rejection variant of reject_then_weighted.

    Instead of a binary mask (gr_z > threshold → weight=0), applies a sigmoid
    gate that smoothly reduces weight for suspicious clients:

        gate_i = sigmoid( -k * (gr_z_i - threshold) )

    Interpretation:
        gr_z_i << threshold  →  gate ≈ 1.0  (clearly benign, full weight)
        gr_z_i == threshold  →  gate = 0.5  (at decision boundary, halved)
        gr_z_i >> threshold  →  gate ≈ 0.0  (clearly attacker, near-zero)

    Final weight = data_size_i * gate_i / sum_j(data_size_j * gate_j)

    Advantages over hard rejection:
    - No cliff at a single threshold value; miscalibration degrades gracefully.
    - Works for any N without re-tuning the threshold as a hard cutoff.
    - The steepness k controls how "hard" the boundary is:
        k=1  very smooth, k=3  near-binary, k=2  recommended default.
    - The threshold parameter controls the midpoint (same scale as before,
      but semantics shift from "reject above" to "sigmoid centre").

    Args:
        trust:               TrustResult from compute_trust_weights.
        data_sizes:          (N,) raw data-size weights (for FedAvg scaling).
        reject_z_threshold:  sigmoid midpoint on the gr_z scale.
        soft_reject_k:       sigmoid steepness (higher = closer to hard reject).
        keep_min:            if all gates fall below 0.1, force top-k by gr_z.
    """
    device = trust.alpha.device
    dtype = trust.alpha.dtype
    N = trust.alpha.numel()

    gr_z = trust.graph_residual_z.detach().clone()
    gate = torch.sigmoid(-soft_reject_k * (gr_z - float(reject_z_threshold)))

    # Safety: if every client's gate is tiny (all look suspicious), fall back
    # to keeping the keep_min least-isolated clients with uniform weight.
    if int((gate > 0.1).sum().item()) < max(1, keep_min):
        k = max(1, keep_min)
        idx = torch.topk(-gr_z, k=min(k, N)).indices
        gate = torch.zeros(N, device=device, dtype=dtype)
        gate[idx] = 1.0

    ds = data_sizes.to(device=device, dtype=dtype) * gate
    total = ds.sum()
    if total.item() <= 0:
        ds = gate
        total = ds.sum().clamp(min=1.0)
    return ds / total
