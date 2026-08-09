# attack/mimic.py
# Copycat attacker: submits a verbatim copy of one benign client's update.
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
# It injects no poison of its own (it trains honestly), so every point the
# federation loses to it is inflicted by the defense, not by the attack.
#
# Why update-space and not label-space:
#   The two label-space arms in attack/hallucination.py
#   (hallu_disjoint_target_subsets / hallu_opposite_directions) try to lower the
#   attacker-attacker cosine INDIRECTLY, through the label targets.  Run Y20
#   (2026-08-06) refuted that: under LoRA fine-tuning the shared "learn the task"
#   component dominates the update vector, so the attackers remained the MOST
#   mutually similar clients in the pool (final-round pairwise cosine 0.470 /
#   0.517 vs 0.299-0.385 for benign) and FoolsGold zeroed them from round 5
#   onward.  This attacker skips the indirection and sets the cosine to 1.0
#   directly in the space where the defense actually measures it.
#
# Deliberately NOT an update FORGER in the AugMP sense: the malicious signal does
# not live in a crafted vector, it lives in the defense's reaction to a genuine
# benign vector.  `crafts_update` is therefore left unset (server.py reads it via
# getattr(..., False)), so trust_mode='v4_cse_reject' runs normally -- and V4 is
# not being deceived, because this client's local model genuinely IS benign and
# its local CSE genuinely IS benign.

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from client import BenignClient


class MimicAttackerClient(BenignClient):
    """
    Copycat attacker: trains honestly, then submits benign `mimic_target_id`'s
    update verbatim instead of its own.

    Notes on the FL server interface:
      - is_attacker = True, so the server routes this client through Phase 3 and
        labels it "Attacker" in the round logs. It is an attacker by ROLE, not by
        payload -- the update it submits is a real benign client's.
      - `mimic_target_id` is chosen by the caller (main.py) rather than here: the
        client only receives updates + ids from the server, never data sizes, and
        the "largest benign shard" policy needs the partition, which main.py has.
      - camouflage_update discards the honestly-trained update. The client MODEL
        keeps the honest weights, which is what V4's pre-aggregation local CSE
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
        # Filled every round by receive_benign_updates (server Phase 3).
        self._benign_updates: Dict[int, torch.Tensor] = {}
        print(
            f"  [Mimic Attacker {client_id}] copies benign client "
            f"{self.mimic_target_id} verbatim; trains honestly, injects no poison "
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
        """Discard our own honest update, submit the target's instead."""
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
        return target.clone()

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
