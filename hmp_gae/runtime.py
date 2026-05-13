# hmp_gae/runtime.py
# HMPGAERuntime: the stateful engine that performs one round of HMP-GAE
# self-supervised training + trust scoring + weighted aggregation.
#
# Called from defense.HMPGAEDefense. Keeps:
#   - a fixed random projection (buffer, not trained)
#   - a NodeFeatureEncoder (trained jointly)
#   - an HMPEncoder (trained jointly)
#   - a HyperedgeDecoder (trained jointly)
#   - an EMA cache Z_hist of previous-round embeddings (detached)
#
# The whole runtime defaults to CPU because N is small and running on CPU
# avoids frequent host<->device transfers of the aggregated update.

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .node_features import (
    FixedRandomProjection,
    NodeFeatureEncoder,
    compute_node_features,
    CONTEXT_DIM,
)
from .hypergraph import knn_hypergraph
from .encoder import HMPEncoder
from .decoder import inner_product_decoder, HyperedgeDecoder
from .losses import total_loss
from .trust_scorer import compute_trust_weights, weighted_aggregate, reject_then_weighted


class HMPGAERuntime:
    def __init__(
        self,
        num_clients: int,
        flat_update_dim: int,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.num_clients = int(num_clients)
        self.flat_update_dim = int(flat_update_dim)
        self.cfg = dict(config or {})
        self.device = torch.device(self.cfg.get("device", device))

        # ---- Hyperparameters with sane defaults ---- #
        self.proj_dim = int(self.cfg.get("proj_dim", 64))
        self.eta_dim = int(self.cfg.get("eta_dim", 64))
        self.hidden_dim = int(self.cfg.get("hidden_dim", 64))
        self.latent_dim = int(self.cfg.get("latent_dim", 32))
        self.num_hmp_layers = int(self.cfg.get("num_hmp_layers", 2))
        self.knn_k = int(self.cfg.get("knn_k", 3))

        self.train_steps_per_round = int(self.cfg.get("train_steps_per_round", 5))
        self.train_lr = float(self.cfg.get("train_lr", 1e-3))
        self.weight_decay = float(self.cfg.get("weight_decay", 1e-5))
        self.lambda_H = float(self.cfg.get("lambda_H", 1.0))
        self.lambda_A = float(self.cfg.get("lambda_A", 1.0))
        self.lambda_hist = float(self.cfg.get("lambda_hist", 0.5))

        # Trust score signal weights.
        # Defaults: graph-structural signal dominates, with a small amount of
        # decoder-residual refinement. Historical deviation is disabled by
        # default (benign clients drift more than attackers during real
        # learning, which can invert the signal).
        self.graph_weight = float(self.cfg.get("graph_weight", 1.0))
        self.residual_weight_alpha = float(self.cfg.get("residual_weight_alpha", 0.3))
        self.hist_weight_beta = float(self.cfg.get("hist_weight_beta", 0.0))
        self.softmax_tau = float(self.cfg.get("softmax_tau", 0.1))

        # Trust-to-weight mapping:
        #   'reject_then_fedavg' (default, recommended for V1): use trust
        #     signals to flag attackers (graph_residual_z > threshold), then
        #     aggregate the non-rejected with their natural FedAvg weights.
        #     Preserves collaborative learning benefit among benigns.
        #   'softmax': pure softmax of trust logits. Simpler but tends to
        #     concentrate weight on 1-2 benign clients when their residuals
        #     are nearly tied.
        self.trust_mode = str(self.cfg.get("trust_mode", "reject_then_fedavg"))
        # Threshold picked by scanning 0.5/0.75/1.0/1.25/1.5 on a synthetic
        # 8-benign/2-attacker setup: 0.75 gives the best balance between
        # attacker detection rate (4/5 rounds) and benign false-reject rate
        # (3/40 spots). Defaulting to 0.75 for V1.
        self.reject_z_threshold = float(self.cfg.get("reject_z_threshold", 0.75))
        self.keep_min = int(self.cfg.get("keep_min", 1))
        self.hist_ema_beta = float(self.cfg.get("hist_ema_beta", 0.9))
        self.proj_seed = int(self.cfg.get("random_proj_seed", 42))
        # Cold-start policy: when no historical embedding is yet available,
        # the trust score has only a single weak signal (residual on an
        # untrained graph) and easily overfits to noise. Falling back to
        # FedAvg keeps round-0 behavior identical to the baseline while
        # still letting HMP-GAE train on the data so it can kick in from
        # round 1 onwards. Set to False to force HMP-GAE from round 0.
        self.cold_start_fallback = bool(self.cfg.get("cold_start_fallback", True))
        # After how many rounds of collected history is HMP trust considered
        # reliable (>= this many). 1 is enough because EMA retains memory.
        self.min_history_for_trust = int(self.cfg.get("min_history_for_trust", 1))

        # ---- Modules ---- #
        self.projection = FixedRandomProjection(
            d_in=self.flat_update_dim, d_out=self.proj_dim, seed=self.proj_seed
        )
        self.node_encoder = NodeFeatureEncoder(
            proj_dim=self.proj_dim, hist_dim=self.latent_dim, eta_dim=self.eta_dim
        ).to(self.device)
        self.hmp_encoder = HMPEncoder(
            eta_dim=self.eta_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_hmp_layers,
        ).to(self.device)
        # M = N: one hyperedge per client (center-node construction).
        self.hyperedge_decoder = HyperedgeDecoder(
            latent_dim=self.latent_dim, num_hyperedges=self.num_clients
        ).to(self.device)

        params = (
            list(self.node_encoder.parameters())
            + list(self.hmp_encoder.parameters())
            + list(self.hyperedge_decoder.parameters())
        )
        self.optim = torch.optim.Adam(
            params, lr=self.train_lr, weight_decay=0.0  # L2 handled in loss
        )

        # ---- State ---- #
        # z_hist is a per-client EMA buffer of the latent embedding.
        self.z_hist: Dict[int, torch.Tensor] = {}

    # --------------------------------------------------------------------- #
    # Helper: pack updates into a tensor aligned with self.num_clients order #
    # --------------------------------------------------------------------- #

    def _stack_updates(self, updates: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([u.detach() for u in updates]).to(
            device=self.device, dtype=torch.float32
        )
        return stacked

    def _history_matrix(
        self, client_ids: List[int]
    ) -> Tuple[torch.Tensor, bool]:
        """
        Build (N, latent_dim) history matrix indexed by the given client_ids.
        Returns (matrix, has_any_history).

        Cold-start clients contribute zero rows.
        """
        n = len(client_ids)
        out = torch.zeros(n, self.latent_dim, device=self.device, dtype=torch.float32)
        any_hist = False
        for i, cid in enumerate(client_ids):
            h = self.z_hist.get(int(cid))
            if h is not None:
                out[i] = h.to(device=self.device, dtype=torch.float32)
                any_hist = True
        return out, any_hist

    def _update_history(self, client_ids: List[int], Z_new: torch.Tensor) -> None:
        Z_detached = Z_new.detach()
        beta = self.hist_ema_beta
        for i, cid in enumerate(client_ids):
            key = int(cid)
            prev = self.z_hist.get(key)
            cur = Z_detached[i].clone().cpu()
            if prev is None:
                self.z_hist[key] = cur
            else:
                self.z_hist[key] = beta * prev + (1.0 - beta) * cur

    # --------------------------------------------------------------------- #
    # Main entry                                                            #
    # --------------------------------------------------------------------- #

    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t0 = time.perf_counter()

        N = len(updates)
        assert N == len(client_ids) == len(data_sizes)

        # ---- 1) pack updates on device ---- #
        updates_stack = self._stack_updates(updates)   # (N, d_update)
        hist_mat, has_hist = self._history_matrix(client_ids)
        Z_hist_arg = hist_mat if has_hist else None

        # ---- 2) self-supervised training steps ---- #
        self.node_encoder.train()
        self.hmp_encoder.train()
        self.hyperedge_decoder.train()
        last_loss_bundle = None
        for step in range(self.train_steps_per_round):
            self.optim.zero_grad(set_to_none=True)

            eta = compute_node_features(
                updates=updates_stack,
                projection=self.projection,
                encoder=self.node_encoder,
                history=hist_mat if has_hist else None,
            )
            H, D_V_inv, D_E_inv = knn_hypergraph(eta, k=self.knn_k)
            Z = self.hmp_encoder(eta, H, D_V_inv, D_E_inv)

            _, A_probs = inner_product_decoder(Z)
            H_hat_logits, _ = self.hyperedge_decoder(Z)

            bundle = total_loss(
                H=H,
                H_hat_logits=H_hat_logits,
                A_hat=A_probs,
                Z=Z,
                Z_hist=Z_hist_arg,
                lambda_H=self.lambda_H,
                lambda_A=self.lambda_A,
                lambda_hist=self.lambda_hist,
                weight_decay=self.weight_decay,
                params=list(self.node_encoder.parameters())
                    + list(self.hmp_encoder.parameters())
                    + list(self.hyperedge_decoder.parameters()),
            )
            bundle.total.backward()
            # Mild gradient clipping for stability when N is small.
            torch.nn.utils.clip_grad_norm_(
                list(self.node_encoder.parameters())
                + list(self.hmp_encoder.parameters())
                + list(self.hyperedge_decoder.parameters()),
                max_norm=5.0,
            )
            self.optim.step()
            last_loss_bundle = bundle

        # ---- 3) eval mode forward for trust scoring ---- #
        self.node_encoder.eval()
        self.hmp_encoder.eval()
        self.hyperedge_decoder.eval()
        with torch.no_grad():
            eta = compute_node_features(
                updates=updates_stack,
                projection=self.projection,
                encoder=self.node_encoder,
                history=hist_mat if has_hist else None,
            )
            H, D_V_inv, D_E_inv = knn_hypergraph(eta, k=self.knn_k)
            Z = self.hmp_encoder(eta, H, D_V_inv, D_E_inv)
            _, A_probs = inner_product_decoder(Z)

            trust = compute_trust_weights(
                A_hat=A_probs,
                Z=Z,
                Z_hist=Z_hist_arg,
                H=H,
                graph_weight=self.graph_weight,
                residual_weight_alpha=self.residual_weight_alpha,
                hist_weight_beta=self.hist_weight_beta,
                softmax_tau=self.softmax_tau,
            )

        # ---- 3b) Map trust signals to aggregation weights ---- #
        ds_tensor = torch.tensor(
            data_sizes, dtype=torch.float32, device=self.device
        )
        ds_total = ds_tensor.sum()
        if ds_total.item() > 0:
            alpha_cold = ds_tensor / ds_total
        else:
            alpha_cold = torch.ones(N, device=self.device) / N

        # Cold-start: HMP trust is unreliable without at least one round of
        # history -- defer to data-size FedAvg for round 0, then switch on.
        use_cold_start_fallback = (
            self.cold_start_fallback and (not has_hist)
        )
        if use_cold_start_fallback:
            used_alpha = alpha_cold
            used_mode = "cold_start_fedavg"
        elif self.trust_mode == "reject_then_fedavg":
            # Reject attackers by structural z-score threshold, then
            # aggregate the kept clients with their natural FedAvg weights.
            used_alpha = reject_then_weighted(
                trust=trust,
                data_sizes=ds_tensor,
                reject_z_threshold=self.reject_z_threshold,
                keep_min=self.keep_min,
            )
            used_mode = "reject_then_fedavg"
        else:
            # Pure softmax over trust logits.
            used_alpha = trust.alpha
            used_mode = "softmax"

        # ---- 4) weighted aggregation ---- #
        aggregated = weighted_aggregate(updates_stack, used_alpha)

        # ---- 5) update EMA history ---- #
        self._update_history(client_ids, Z)

        # ---- 6) stats dict ---- #
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats: Dict[str, Any] = {
            # `alpha` = weights actually used for aggregation this round.
            "alpha": used_alpha.detach().cpu().tolist(),
            # `alpha_hmp` = what HMP-GAE trust scoring would give (for
            # diagnostics even when cold-start fallback overrides it).
            "alpha_hmp": trust.alpha.detach().cpu().tolist(),
            # Kept as 'residual' (backward-compat field name in logs /
            # visualization) -- this is the graph-structural signal, the
            # primary driver of trust in V1.
            "residual": trust.graph_residual.detach().cpu().tolist(),
            "recon_residual": trust.recon_residual.detach().cpu().tolist(),
            "hist_dev": trust.hist_dev.detach().cpu().tolist(),
            "has_history": bool(has_hist),
            "cold_start_fallback_used": bool(use_cold_start_fallback),
            "trust_mode_used": used_mode,
            "defense_time_ms": float(elapsed_ms),
        }
        if last_loss_bundle is not None:
            stats["L_rec"] = float(last_loss_bundle.L_rec_H.item())
            stats["L_smooth"] = float(last_loss_bundle.L_smooth.item())
            stats["L_hist"] = float(last_loss_bundle.L_hist.item())
        # Keep Z around so the caller can persist it for visualization,
        # but do not let it leak into the standard JSON log (defense
        # package strips this key before logging).
        stats["Z"] = Z.detach().cpu().numpy()
        return aggregated.detach().cpu(), stats
