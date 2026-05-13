# attack/__init__.py
# Attack baselines (model poisoning + hallucination). Import concrete clients from submodules, e.g.:
#   from attack.alie import ALIEAttackerClient

from .alie import ALIEAttackerClient
from .gaussian import GaussianAttackerClient
from .hallucination import HallucinationAttackerClient
from .sign_flipping import SignFlippingAttackerClient

__all__ = [
    "ALIEAttackerClient",
    "GaussianAttackerClient",
    "HallucinationAttackerClient",
    "SignFlippingAttackerClient",
]
