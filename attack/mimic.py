# attack/mimic.py
# Copycat attacker: submits a close copy of one benign client's update.
#
# Target: FoolsGold (Fung et al., RAID '20), whose per-client weight is
#
#     pardoned[i][j] = cos[i][j] * min(1, max_cs[i] / max_cs[j])
#     wv[i]          = 1 - max_j pardoned[i][j]
#
# The pardoning factor only protects a benign client whose OWN max similarity is
# lower than its accuser's.  An exact copy makes the two equal by construction --
# cos = 1 on both sides, so max_cs is 1 for both, so the ratio is 1 and nothing is
# pardoned -- and both the victim and the copycat collapse to wv = 0.  The copycat
# therefore manufactures a FALSE POSITIVE: the defense ejects a benign client.
# It performs no label poisoning and trains honestly. Optional noise makes the
# copied updates close but non-identical while preserving the target norm.
#
# Why update-space and not label-space:
#   The two label-space arms in attack/hallucination.py
#   (hallu_disjoint_target_subsets / hallu_opposite_directions) try to lower the
#   attacker-attacker cosine INDIRECTLY, through the label targets.  Run Y20
#   (2026-08-06) refuted that: under LoRA fine-tuning the shared "learn the task"
#   component dominates the update vector, so the attackers remained the MOST
#   mutually similar clients in the pool (final-round pairwise cosine 0.470 /
#   0.517 vs 0.299-0.385 for benign) and FoolsGold zeroed them from round 5
#   onward. This attacker skips the indirection and controls the cosine directly
#   in the space where the defense actually measures it.
#
# Deliberately NOT an update FORGER in the AugMP sense: the malicious signal does
# not live in a crafted vector, it lives in the defense's reaction to a genuine
# benign vector.  `crafts_update` is therefore left unset (server.py reads it via
# getattr(..., False)), so trust_mode='v4_cse_reject' runs normally -- and V4 is
# not being deceived, because this client's local model genuinely IS benign and
# its local CSE genuinely IS benign.

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from client import BenignClient


class MimicAttackerClient(BenignClient):
    """
    Copycat attacker: trains honestly, then submits benign `mimic_target_id`'s
    update exactly or with a controlled, norm-preserving honest residual.

    Notes on the FL server interface:
      - is_attacker = True, so the server routes this client through Phase 3 and
        labels it "Attacker" in the round logs. It is an attacker by ROLE, not by
        payload -- the update it submits is a real benign client's.
      - `mimic_target_id` is chosen by the caller (main.py) rather than here: the
        client only receives updates + ids from the server, never data sizes, and
        the "largest benign shard" policy needs the partition, which main.py has.
      - camouflage_update uses the honestly-trained update only to supply a
        client-specific direction orthogonal to the target. The client MODEL
        keeps the honest weights, which is what V4+'s pre-aggregation local CSE
        evaluates -- intentionally, see the module docstring.
    """

    def __init__(
        self,
        client_id: int,
        model,
        data_loader: DataLoader,
        lr: float,
        local_epochs: int,
        alpha: float,
        mimic_target_id: int,
        data_indices: Optional[List[int]] = None,
        grad_clip_norm: float = 1.0,
        claimed_data_size: float = 1.0,
        mimic_cosine_range: Optional[Sequence[float]] = None,
        mimic_noise_seed: int = 42,
    ):
        super().__init__(
            client_id=client_id,
            model=model,
            data_loader=data_loader,
            lr=lr,
            local_epochs=local_epochs,
            alpha=alpha,
            data_indices=data_indices,
            grad_clip_norm=grad_clip_norm,
        )
        if int(mimic_target_id) == int(client_id):
            raise ValueError(
                f"MimicAttackerClient {client_id} cannot mimic itself "
                f"(mimic_target_id={mimic_target_id}); the target must be a benign client."
            )
        self.is_attacker = True
        self.attack_method = "Mimic"
        self.claimed_data_size = float(claimed_data_size)
        self.mimic_target_id = int(mimic_target_id)
        cosine_range = [1.0, 1.0] if mimic_cosine_range is None else mimic_cosine_range
        if len(cosine_range) != 2:
            raise ValueError("hallu_mimic_cosine_range must be [lo, hi]")
        lo, hi = float(cosine_range[0]), float(cosine_range[1])
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError("hallu_mimic_cosine_range must satisfy 0 < lo <= hi <= 1")
        self.mimic_cosine_range = (lo, hi)
        self.mimic_noise_seed = int(mimic_noise_seed)
        # Filled every round by receive_benign_updates (server Phase 3).
        self._benign_updates: Dict[int, torch.Tensor] = {}
        copy_desc = (
            "verbatim"
            if lo == hi == 1.0
            else f"imperfect, target cosine U[{lo:.3f}, {hi:.3f}], norm-preserving"
        )
        print(
            f"  [Mimic Attacker {client_id}] copies benign client "
            f"{self.mimic_target_id} ({copy_desc}); trains honestly, no label poison "
            f"(claimed_data_size={self.claimed_data_size:.0f})"
        )

    # ---------------------------- attack hooks ----------------------------- #

    def receive_benign_updates(
        self,
        updates: List[torch.Tensor],
        client_ids: Optional[List[int]] = None,
    ) -> None:
        if client_ids is None:
            raise ValueError(
                "MimicAttackerClient needs client_ids to identify its target; "
                "the server passes them (server.py Phase 3)."
            )
        self._benign_updates = {
            int(cid): upd for cid, upd in zip(client_ids, updates)
        }

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """Submit an exact or controlled imperfect copy of the target update."""
        target = self._benign_updates.get(self.mimic_target_id)
        if target is None:
            # Loud rather than silent: falling back to our own update would turn
            # this client into an extra benign participant and quietly void the
            # whole experiment (project convention, cf. V4's missing-local_cse).
            raise RuntimeError(
                f"MimicAttackerClient {self.client_id}: no update received for "
                f"mimic_target_id={self.mimic_target_id}; got "
                f"{sorted(self._benign_updates)}. Is the target actually benign?"
            )
        target = target.clone()
        lo, hi = self.mimic_cosine_range
        if lo == hi == 1.0:
            return target
        if target.shape != poisoned_update.shape:
            raise RuntimeError(
                f"MimicAttackerClient {self.client_id}: own/target update shape "
                f"mismatch {tuple(poisoned_update.shape)} != {tuple(target.shape)}"
            )
        target_norm = torch.linalg.vector_norm(target)
        if not torch.isfinite(target_norm) or target_norm.item() <= 1e-12:
            raise RuntimeError(
                f"MimicAttackerClient {self.client_id}: target update has zero or "
                "non-finite norm; target cosine is undefined"
            )
        seed = (
            self.mimic_noise_seed * 1_000_003
            + self.client_id * 100_069
            + self.current_round * 10_007
        )
        target_cosine = random.Random(seed).uniform(lo, hi)
        target_hat = target / target_norm
        own = poisoned_update.to(device=target.device, dtype=target.dtype)
        residual = own - torch.dot(own.reshape(-1), target_hat.reshape(-1)) * target_hat
        residual_norm = torch.linalg.vector_norm(residual)
        if not torch.isfinite(residual_norm) or residual_norm.item() <= 1e-12:
            if target_hat.numel() < 2:
                raise RuntimeError("imperfect mimic needs an update with at least 2 values")
            residual = torch.zeros_like(target_hat)
            index = int(torch.argmin(target_hat.abs()).item())
            residual.reshape(-1)[index] = 1.0
            residual -= target_hat.reshape(-1)[index] * target_hat
            residual_norm = torch.linalg.vector_norm(residual)
        residual_hat = residual / residual_norm
        sine = math.sqrt(max(0.0, 1.0 - target_cosine * target_cosine))
        return target_norm * (target_cosine * target_hat + sine * residual_hat)

    # --------------------- server compatibility no-ops --------------------- #

    def receive_attacker_updates(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: Optional[Dict[int, float]] = None,
    ) -> None:
        pass

    def set_global_model_params(self, global_params: torch.Tensor) -> None:
        pass

    def set_constraint_params(self, **kwargs) -> None:
        pass

    def set_lagrangian_params(self, **kwargs) -> None:
        pass
