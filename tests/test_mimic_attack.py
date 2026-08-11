# tests/test_mimic_attack.py
# CPU-only sanity tests for the mimic (copycat) attacker, arm C of the
# heterogeneous-attacker experiments (2026-08-09):
#   - the client submits its target's update, not its own
#   - a missing / self target crashes loudly instead of degrading to benign
#   - THE POINT OF THE ARM: an exact copy drives the mimicked BENIGN client's
#     FoolsGold weight to zero, i.e. manufactures a false positive
#
# The last test is the one that matters. If it ever fails, the whole arm is
# pointless and no GPU hours should be spent on it.
#
# Pure tensors, no FL training, no GPU, no dataset — runs in ~1s:
#
#     python tests/test_mimic_attack.py
#
# Needs torch (like tests/test_trust_robustness.py) -> run on Colab, not on a
# pure-editing machine.
#
# Intentionally plain asserts (no pytest dependency, matching the repo).

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from attack.mimic import MimicAttackerClient
from defense.foolsgold import FoolsGoldDefense


def _make_mimic(client_id: int = 3, target: int = 0) -> MimicAttackerClient:
    # MimicAttackerClient.__init__ only deep-copies the model and stores config;
    # nothing here touches the DataLoader or the parameters, so a bare Linear and
    # data_loader=None are enough for the update-space contract.
    return MimicAttackerClient(
        client_id=client_id,
        model=nn.Linear(2, 2),
        data_loader=None,
        lr=1e-4,
        local_epochs=1,
        alpha=0.0,
        mimic_target_id=target,
        claimed_data_size=100.0,
    )


def test_submits_target_update_not_own():
    mimic = _make_mimic(client_id=3, target=1)
    own = torch.tensor([9.0, 9.0, 9.0])
    b0, b1, b2 = (
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    mimic.receive_benign_updates([b0, b1, b2], client_ids=[0, 1, 2])
    out = mimic.camouflage_update(own)
    assert torch.equal(out, b1), f"expected client 1's update, got {out}"
    assert not torch.equal(out, own), "mimic must discard its own update"
    # A clone, not an alias -- otherwise a later in-place op on the aggregated
    # tensor would silently mutate the victim's real update.
    out += 1.0
    assert torch.equal(b1, torch.tensor([0.0, 1.0, 0.0])), "target update was aliased"
    assert mimic.is_attacker is True
    assert getattr(mimic, "crafts_update", False) is False, (
        "mimic must not advertise crafts_update: server.py refuses to run "
        "trust_mode='v4_cse_reject' alongside update-forging attackers, and this "
        "client hides no poison in its vector"
    )


def test_missing_target_crashes_loudly():
    mimic = _make_mimic(client_id=3, target=7)  # 7 is not in the federation
    mimic.receive_benign_updates([torch.zeros(3)], client_ids=[0])
    try:
        mimic.camouflage_update(torch.ones(3))
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "a missing mimic target must raise, not fall back to the own update "
            "(that would quietly turn the attacker into an extra benign client)"
        )
    try:
        _make_mimic(client_id=3, target=3)
    except ValueError:
        pass
    else:
        raise AssertionError("mimicking itself must raise")
    try:
        _make_mimic().receive_benign_updates([torch.zeros(3)], client_ids=None)
    except ValueError:
        pass
    else:
        raise AssertionError("missing client_ids must raise")


def test_exact_copy_zeroes_the_victims_foolsgold_weight():
    """The arm's whole premise: FoolsGold pardoning cannot save an exact copy.

    pardoned[i][j] = cos[i][j] * min(1, max_cs[i]/max_cs[j]) only rescues a
    victim whose own max similarity is LOWER than its accuser's. An exact copy
    makes both exactly 1, so the ratio is 1, nothing is pardoned, and victim and
    copycat collapse together.
    """
    d = 8
    # Four mutually orthogonal unit updates: no client is similar to any other,
    # so FoolsGold has nothing to penalise and every alpha is equal.
    base = [torch.zeros(d) for _ in range(4)]
    for i, u in enumerate(base):
        u[i] = 1.0

    control = FoolsGoldDefense(num_clients=4)
    _, control_stats = control.aggregate(
        base, [0, 1, 2, 3], [1.0] * 4, round_num=0, device=torch.device("cpu")
    )
    alpha_control = control_stats["alpha"]
    assert min(alpha_control) > 0.0, (
        f"control federation should keep every client, got {alpha_control}"
    )

    # Same federation, except client 3 now submits client 0's update verbatim.
    mimicked = [base[0], base[1], base[2], base[0].clone()]
    attacked = FoolsGoldDefense(num_clients=4)
    _, stats = attacked.aggregate(
        mimicked, [0, 1, 2, 3], [1.0] * 4, round_num=0, device=torch.device("cpu")
    )
    alpha = stats["alpha"]
    assert alpha[0] == 0.0, (
        f"FALSE POSITIVE not reproduced: benign victim c0 kept alpha={alpha[0]} "
        f"(full alpha={alpha}). The arm has no effect -- do not run it."
    )
    assert alpha[3] == 0.0, f"copycat c3 should also be zeroed, got {alpha}"
    assert alpha[1] > 0.0 and alpha[2] > 0.0, (
        f"uninvolved benign clients must survive, got {alpha}"
    )


def test_all_attackers_copying_one_victim_zero_the_whole_trio():
    """Active arm: EVERY attacker copies the SAME benign client (Karimireddy '22).

    With 5 clients = 3 benign + 2 copycats of benign c0, the accumulated histories
    of {c0, c3, c4} are identical, so every pairwise cosine inside that trio is 1
    and every max_cs is 1 -- no pardoning is possible in any direction and all
    three collapse to wv = 0. Only c1 and c2 survive to be aggregated.

    This is the property the whole arm rests on: no attacker flipped a label, so
    every point the federation loses here is a pure false positive.
    """
    d = 8
    base = [torch.zeros(d) for _ in range(3)]
    for i, u in enumerate(base):
        u[i] = 1.0

    updates = [base[0], base[1], base[2], base[0].clone(), base[0].clone()]
    defense = FoolsGoldDefense(num_clients=5)
    _, stats = defense.aggregate(
        updates, [0, 1, 2, 3, 4], [1.0] * 5, round_num=0, device=torch.device("cpu")
    )
    alpha = stats["alpha"]
    assert alpha[0] == 0.0, (
        f"FALSE POSITIVE not reproduced: benign victim c0 kept alpha={alpha[0]} "
        f"(full alpha={alpha}). The arm has no effect -- do not run it."
    )
    assert alpha[3] == 0.0 and alpha[4] == 0.0, (
        f"both copycats should be zeroed, got {alpha}"
    )
    assert alpha[1] > 0.0 and alpha[2] > 0.0, (
        f"uninvolved benign clients must survive, got {alpha}"
    )


if __name__ == "__main__":
    test_submits_target_update_not_own()
    test_missing_target_crashes_loudly()
    test_exact_copy_zeroes_the_victims_foolsgold_weight()
    test_all_attackers_copying_one_victim_zero_the_whole_trio()
    print("\nAll mimic-attack tests passed.")
