# client.py
# Federated learning client base classes.
#
# Provides:
#   - Client       : abstract base (device handling, optimizer lifecycle, etc.)
#   - BenignClient : honest FedProx-style local training
#
# Attacker implementations live under the ``attack/`` package:
#   - attack.hallucination  (label-flipping hallucination -- this paper)
#   - attack.sign_flipping  (ICML '18 baseline)
#   - attack.gaussian       (USENIX Security '20 baseline)
#   - attack.alie           (NeurIPS '19 baseline)

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Base Client                                                                 #
# --------------------------------------------------------------------------- #

class Client:
    """
    Generic federated learning client.

    Keeps a deep-copied model per client. The model is held on CPU by default
    to save GPU memory and is moved on-demand to the server device during
    local training (see ``BenignClient.local_train``).
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data_loader,
        lr: float,
        local_epochs: int,
        alpha: float,
    ):
        """
        Args:
            client_id: Unique identifier for the client.
            model: The neural network model (will be deep copied).
            data_loader: DataLoader for local training data.
            lr: Learning rate for local optimizer (required, no default).
            local_epochs: Number of local training epochs per round (required).
            alpha: FedProx proximal coefficient mu; loss += (mu/2)*||w-w_t||^2.
                   Set 0 for standard FedAvg.
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.alpha = alpha

        # Explicit cuda:0 avoids subtle device-id mismatches between
        # 'cuda' and 'cuda:0' on multi-GPU hosts.
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.optimizer = None            # lazy-created in local_train
        self.current_round = 0
        self.is_attacker = False
        self._model_on_gpu = False       # tracks device placement for reset_optimizer

    def reset_optimizer(self) -> None:
        """Reset the optimizer. Only valid when the model is on GPU."""
        if self._model_on_gpu:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=self.lr)
        else:
            self.optimizer = None

    def set_round(self, round_num: int) -> None:
        """Set the current training round."""
        self.current_round = int(round_num)

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """
        Compute Delta = current_params - initial_params.

        Returns a CPU tensor regardless of model placement so that
        aggregation logic remains device-agnostic.
        """
        current_params = self.model.get_flat_params()
        if current_params.device.type == "cuda":
            current_params = current_params.cpu()
        if initial_params.device.type == "cuda":
            initial_params = initial_params.cpu()
        return current_params - initial_params

    def local_train(self, epochs: Optional[int] = None) -> torch.Tensor:
        """Subclasses must implement."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Benign Client (FedProx local training)                                      #
# --------------------------------------------------------------------------- #

class BenignClient(Client):
    """
    Honest client performing FedProx-style local training.

    Standard formulation: minimize_w  F_k(w) + (mu/2) * ||w - w_t||^2
    where ``mu`` is self.alpha and ``w_t`` is the global model at round start.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data_loader,
        lr: float,
        local_epochs: int,
        alpha: float,
        data_indices: Optional[List[int]] = None,
        grad_clip_norm: float = 1.0,
    ):
        super().__init__(client_id, model, data_loader, lr, local_epochs, alpha)
        self.data_indices = data_indices or []
        self.grad_clip_norm = grad_clip_norm

    def prepare_for_round(self, round_num: int) -> None:
        """Benign clients don't need special preparation -- just track the round."""
        self.set_round(round_num)

    def local_train(self, epochs: Optional[int] = None) -> torch.Tensor:
        """
        Run FedProx local training for ``epochs`` (defaults to ``self.local_epochs``).
        
        Returns the model update ``Delta = w_local - w_global`` (on CPU).
        """
        if epochs is None:
            epochs = self.local_epochs
            
        if not self._model_on_gpu:
            self.model.to(self.device)
            self._model_on_gpu = True
            if self.optimizer is None:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = optim.Adam(trainable_params, lr=self.lr)
            
        self.model.train()
        initial_params = self.model.get_flat_params().clone().cpu()
        mu = self.alpha

        for epoch in range(epochs):
            pbar = tqdm(
                self.data_loader,
                desc=f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                ce_loss = nn.CrossEntropyLoss()(logits, labels)
                
                current_params = self.model.get_flat_params(requires_grad=True)
                initial_params_gpu = initial_params.to(self.device)
                proximal_term = (mu / 2.0) * torch.norm(current_params - initial_params_gpu) ** 2
                initial_params_gpu = None  # release reference
                
                loss = ce_loss + proximal_term
                
                if not torch.isfinite(loss).item():
                    # Skip non-finite batches to avoid corrupting the model.
                    import warnings
                    warnings.warn(
                        f"[Client {self.client_id}] Skipping batch: loss={loss.item()} (non-finite). "
                        "Consider lowering client_lr or grad_clip_norm for decoder models (e.g. Pythia-160m)."
                    )
                    pbar.set_postfix({"loss": "nan(skip)"})
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )
                self.optimizer.step()
                pbar.set_postfix({"loss": loss.item()})

        update = self.get_model_update(initial_params)
        
        # Release GPU memory between clients to keep peak usage bounded.
        self.model.cpu()
        self._model_on_gpu = False
        del self.optimizer
        self.optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return update

    def receive_benign_updates(self, updates: List[torch.Tensor]) -> None:
        """No-op -- benign clients don't consume peer updates."""
        pass
