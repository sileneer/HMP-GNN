# server.py
# This module implements the Server class for federated learning, including model aggregation.

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import copy
from client import BenignClient
import torch.nn.functional as F
from defense import Defense, FedAvgDefense, build_defense


class Server:
    """Server class for federated learning with model aggregation"""
    def __init__(self, global_model: nn.Module, test_loader,
                total_rounds=20, server_lr=1.0,
                similarity_mode='pairwise',
                defense_method: str = 'fedavg',
                defense_config: Optional[Dict[str, Any]] = None,
                num_clients: Optional[int] = None,
                compute_classification_semantic_entropy: bool = True):
        self.global_model = copy.deepcopy(global_model)
        self.test_loader = test_loader
        self.total_rounds = total_rounds
        # CRITICAL: Use explicit cuda:0 instead of 'cuda' to ensure device consistency
        # This prevents issues where 'cuda' and 'cuda:0' are treated as different devices
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.global_model.to(self.device)
        self.clients = []
        self.client_dict = {}  # client_id -> client mapping for O(1) lookup
        self.log_data = []

        # Server parameters
        self.server_lr = server_lr  # Server learning rate
        # Similarity mode (diagnostics only, consumed by visualization):
        # 'local_vs_global' | 'pairwise' | 'both'
        self.similarity_mode = str(similarity_mode).lower() if similarity_mode else 'pairwise'
        if self.similarity_mode not in ('local_vs_global', 'pairwise', 'both'):
            self.similarity_mode = 'pairwise'

        # Defense strategy (pluggable aggregation rule)
        # 'fedavg' (default, backward-compatible) or 'hmp_gae' (this paper).
        self.defense_method = (defense_method or 'fedavg').lower()
        self.defense_config = defense_config or {}
        self.defense: Defense = build_defense(
            method=self.defense_method,
            num_clients=num_clients if num_clients is not None else 0,
            defense_config=self.defense_config,
        )
        # Track the round currently being aggregated (set in run_round).
        self._current_round = 0
        self.compute_classification_semantic_entropy = bool(
            compute_classification_semantic_entropy)

        # Track historical data
        self.history = {
            'clean_acc': [],
            'local_accuracies': {},   # {client_id: [acc_r0, acc_r1, ...]}
            'local_cse': {},          # {client_id: [cse_r0, cse_r1, ...]}
        }

    def register_client(self, client):
        """Register a client to the server."""
        self.clients.append(client)
        # Update client_id -> client mapping for O(1) lookup
        self.client_dict[client.client_id] = client

    def broadcast_model(self):
        """Broadcast the global model to all clients."""
        global_params = self.global_model.get_flat_params()
        # Clone and move to CPU to save GPU memory
        global_params_cpu = global_params.clone().cpu()
        for client in self.clients:
            # set_flat_params works on CPU models
            client.model.set_flat_params(global_params_cpu.clone())
            # Reset optimizer if model is on GPU (rarely needed now)
            if hasattr(client, '_model_on_gpu') and client._model_on_gpu:
                client.reset_optimizer()
            else:
                client.optimizer = None

    def _compute_weighted_average(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute weighted average update (FedAvg style) shared by similarity and distance calculations.
        
        Args:
            updates: List of client update tensors
            client_ids: List of client IDs (optional, for weighted aggregation)
            
        Returns:
            weighted_avg: Weighted average update tensor
            weights: List of weights used for each client
        """
        if client_ids is not None and len(client_ids) == len(updates):
            weights = []
            # Use dictionary lookup for O(1) access instead of linear search
            client_dict = getattr(self, 'client_dict', {c.client_id: c for c in self.clients})
            for cid in client_ids:
                client = client_dict.get(cid)
                if client:
                    if getattr(client, 'is_attacker', False):
                        D_i = float(getattr(client, 'claimed_data_size', 1.0))
                    else:
                        D_i = float(len(getattr(client, 'data_indices', [])) or 1.0)
                else:
                    D_i = 1.0
                weights.append(D_i)
            
            total_D = sum(weights) + 1e-12
            weighted_avg = torch.zeros_like(updates[0])
            for update, w in zip(updates, weights):
                weighted_avg += (w / total_D) * update
        else:
            weighted_avg = torch.stack(updates).mean(dim=0)
            weights = [1.0 / len(updates)] * len(updates)
        
        return weighted_avg, weights

    def _compute_similarities(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> np.ndarray:
        """
        Compute cosine similarities between each update and the weighted average update.
        
        CRITICAL: Uses weighted aggregation (FedAvg style) to match attack optimization distance definition.
        
        Definition (consistent with attack optimization):
            sim_i = cosine_similarity(Δ_i, Δ_g)
            where Δ_g = Σ_j (D_j / D_total) * Δ_j (weighted average, FedAvg style)
        
        This matches the distance definition used in _compute_distance_update_space:
            dist = ||Δ_att - Δ_g|| where Δ_g is weighted aggregate
        
        Args:
            updates: List of client update tensors
            client_ids: List of client IDs (optional, for weighted aggregation)
            
        Returns:
            numpy array of cosine similarities (one per client)
        """
        n_updates = len(updates)

        print("  📊 Computing cosine similarities (weighted aggregation, matches attack optimization)")

        # Compute weighted average (shared with distance calculation)
        weighted_avg, _ = self._compute_weighted_average(updates, client_ids)
        
        # Compute cosine similarity for all updates at once (batch computation)
        updates_stack = torch.stack(updates)  # (N, D)
        weighted_avg_expanded = weighted_avg.unsqueeze(0).expand_as(updates_stack)  # (N, D)
        similarities = torch.cosine_similarity(updates_stack, weighted_avg_expanded, dim=1).cpu().numpy()

        # Print information
        print(f"  📈 Cosine Similarity - Mean: {similarities.mean():.3f}, "
              f"Std Dev: {similarities.std():.3f}")

        # Display similarity for each client
        # Note: similarities are ordered by updates, which match client_ids order from aggregate_updates
        attacker_ids = {client.client_id for client in self.clients if getattr(client, 'is_attacker', False)}
        for i, sim in enumerate(similarities):
            if hasattr(self, '_sorted_client_ids') and i < len(self._sorted_client_ids):
                client_id = self._sorted_client_ids[i]
                client = next((c for c in self.clients if c.client_id == client_id), None)
                if client:
                    client_type = "Attacker" if getattr(client, 'is_attacker', False) else "Benign"
                    print(f"    Client {client_id} ({client_type}): {sim:.3f}")
                else:
                    print(f"    Client {client_id}: {sim:.3f}")
            else:
                print(f"    Update {i}: {sim:.3f}")

        return similarities

    def _compute_euclidean_distances(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> np.ndarray:
        """
        Compute Euclidean distances between each update and the weighted average update.
        
        CRITICAL: Uses weighted aggregation (FedAvg style) to match attack optimization distance definition.
        
        Definition (consistent with attack optimization):
            dist_i = ||Δ_i - Δ_g||
            where Δ_g = Σ_j (D_j / D_total) * Δ_j (weighted average, FedAvg style)
        
        This matches the distance definition used in _compute_distance_update_space:
            dist = ||Δ_att - Δ_g|| where Δ_g is weighted aggregate
        
        Args:
            updates: List of client update tensors
            client_ids: List of client IDs (optional, for weighted aggregation)
            
        Returns:
            numpy array of Euclidean distances (one per client)
        """
        n_updates = len(updates)
        
        print("  📊 Computing Euclidean distances (weighted aggregation, matches attack optimization)")
        
        # Compute weighted average (shared with similarity calculation)
        weighted_avg, _ = self._compute_weighted_average(updates, client_ids)
        
        # Compute Euclidean distance for all updates at once (batch computation)
        updates_stack = torch.stack(updates)  # (N, D)
        weighted_avg_expanded = weighted_avg.unsqueeze(0).expand_as(updates_stack)  # (N, D)
        diff = updates_stack - weighted_avg_expanded  # (N, D)
        distances = torch.norm(diff, dim=1).cpu().numpy()
        
        # Print information
        print(f"  📈 Euclidean Distance - Mean: {distances.mean():.6f}, "
              f"Std Dev: {distances.std():.6f}")
        
        # Display distance for each client
        attacker_ids = {client.client_id for client in self.clients if getattr(client, 'is_attacker', False)}
        for i, dist in enumerate(distances):
            if hasattr(self, '_sorted_client_ids') and i < len(self._sorted_client_ids):
                client_id = self._sorted_client_ids[i]
                client = next((c for c in self.clients if c.client_id == client_id), None)
                if client:
                    client_type = "Attacker" if getattr(client, 'is_attacker', False) else "Benign"
                    print(f"    Client {client_id} ({client_type}): {dist:.6f}")
                else:
                    print(f"    Client {client_id}: {dist:.6f}")
            else:
                print(f"    Update {i}: {dist:.6f}")
        
        return distances

    def _compute_similarities_pairwise(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise cosine similarities between all client updates (no self, no global).
        S[i,j] = cosine_similarity(Δ_i, Δ_j). Per-client metric: mean similarity to other clients (exclude self).
        
        Returns:
            similarity_matrix: (N, N) numpy array
            similarities_derived: (N,) per-client mean similarity to others (same order as client_ids)
        """
        n = len(updates)
        print("  📊 Computing cosine similarities (pairwise: local vs local, no self)")
        if n == 0:
            return np.array([]).reshape(0, 0), np.array([])
        updates_stack = torch.stack(updates)  # (N, D)
        normalized = F.normalize(updates_stack.float(), p=2, dim=1)  # (N, D)
        similarity_matrix = (normalized @ normalized.T).cpu().numpy()  # (N, N), diagonal = 1
        # Per-client: mean over j != i (exclude self)
        similarities_derived = np.zeros(n)
        if n == 1:
            similarities_derived[0] = 1.0
        else:
            for i in range(n):
                others = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
                similarities_derived[i] = float(np.mean(others))
        print(f"  📈 Cosine Similarity (pairwise mean) - Mean: {similarities_derived.mean():.3f}, Std Dev: {similarities_derived.std():.3f}")
        attacker_ids = {client.client_id for client in self.clients if getattr(client, 'is_attacker', False)}
        for i, sim in enumerate(similarities_derived):
            if hasattr(self, '_sorted_client_ids') and i < len(self._sorted_client_ids):
                client_id = self._sorted_client_ids[i]
                client = next((c for c in self.clients if c.client_id == client_id), None)
                if client:
                    client_type = "Attacker" if getattr(client, 'is_attacker', False) else "Benign"
                    print(f"    Client {client_id} ({client_type}): {sim:.3f}")
                else:
                    print(f"    Client {client_id}: {sim:.3f}")
            else:
                print(f"    Update {i}: {sim:.3f}")
        return similarity_matrix, similarities_derived

    def _compute_raw_weights(self, client_ids: List[int]) -> List[float]:
        """
        Data-size-based weights used by FedAvg (and as the default prior for
        defenses that do not override them).
        """
        weights: List[float] = []
        for cid in client_ids:
            client = self.clients[cid]
            if getattr(client, 'is_attacker', False):
                w = float(getattr(client, 'claimed_data_size', 1.0))
            else:
                w = float(len(getattr(client, 'data_indices', [])) or 1.0)
            weights.append(w)
        return weights

    def aggregate_updates(self, updates: List[torch.Tensor],
                          client_ids: List[int]) -> Dict:
        # Store client_ids for similarity display
        self._current_client_ids = client_ids
        self._sorted_client_ids = client_ids

        # Raw aggregation weights (data-size-based), passed to the defense as a prior.
        raw_weights = self._compute_raw_weights(client_ids)

        # Delegate to the configured defense strategy.
        aggregated_update, defense_stats = self.defense.aggregate(
            updates=updates,
            client_ids=client_ids,
            data_sizes=raw_weights,
            round_num=self._current_round,
            device=self.device,
        )
        # Ensure aggregated update is on the server device with consistent dtype.
        aggregated_update = aggregated_update.to(device=self.device, dtype=updates[0].dtype)
        aggregated_update_norm = torch.norm(aggregated_update).item()

        # Update global model (standard FedAvg update rule: w_{t+1} = w_t + eta * Delta).
        current_params = self.global_model.get_flat_params()
        new_params = current_params + self.server_lr * aggregated_update
        self.global_model.set_flat_params(new_params)

        defense_label = defense_stats.get('defense_name', self.defense_method)
        print(f"  📊 Defense [{defense_label}]: Aggregated {len(updates)}/{len(updates)} updates")
        print(f"  🔧 Server Learning Rate: {self.server_lr}")
        print(f"  📐 Aggregated update norm: {aggregated_update_norm:.6f}")
        alpha_list = defense_stats.get('alpha')
        if isinstance(alpha_list, list) and len(alpha_list) == len(client_ids):
            alpha_summary = ", ".join(
                f"c{cid}={a:.3f}" for cid, a in zip(client_ids, alpha_list)
            )
            print(f"  ⚖️  Trust weights: {alpha_summary}")

        # Compute similarity and distance metrics for visualization (unchanged).
        mode = getattr(self, 'similarity_mode', 'local_vs_global')
        if mode == 'local_vs_global':
            similarities = self._compute_similarities(updates, client_ids)
            similarity_matrix = None
            similarities_vs_global = None
        elif mode == 'pairwise':
            similarity_matrix, similarities = self._compute_similarities_pairwise(updates, client_ids)
            similarities_vs_global = None
        else:  # 'both'
            similarities_vs_global = self._compute_similarities(updates, client_ids)
            similarity_matrix, similarities = self._compute_similarities_pairwise(updates, client_ids)
        euclidean_distances = self._compute_euclidean_distances(updates, client_ids) if len(updates) > 0 else np.array([])

        aggregation_log = {
            'similarities': similarities.tolist(),
            'euclidean_distances': euclidean_distances.tolist() if len(euclidean_distances) > 0 else [],
            'accepted_clients': client_ids.copy(),
            'mean_similarity': float(similarities.mean()) if len(similarities) > 0 else 1.0,
            'std_similarity': float(similarities.std()) if len(similarities) > 0 else 0.0,
            'mean_euclidean_distance': euclidean_distances.mean().item() if len(euclidean_distances) > 0 else 0.0,
            'std_euclidean_distance': euclidean_distances.std().item() if len(euclidean_distances) > 0 else 0.0,
            'aggregated_update_norm': aggregated_update_norm,
            'defense_method': defense_label,
            'trust_weights': alpha_list if isinstance(alpha_list, list) else None,
            'raw_weights': raw_weights,
        }
        # Persist extra defense stats (skip bulky numpy blobs like 'Z' from the
        # main JSON log to keep result files lean; HMP runtime writes its own
        # stats file if enabled).
        for k in ('residual', 'hist_dev', 'L_rec', 'L_smooth', 'L_hist',
                  'fallback_reason', 'defense_time_ms'):
            if k in defense_stats:
                aggregation_log[k] = defense_stats[k]
        if similarity_matrix is not None:
            aggregation_log['similarity_matrix'] = similarity_matrix.tolist()
        if similarities_vs_global is not None:
            aggregation_log['similarities_vs_global'] = similarities_vs_global.tolist()
        aggregation_log['similarity_mode'] = mode

        return aggregation_log

    def evaluate_local_metrics(self, client) -> Tuple[float, float]:
        """
        Evaluate a client's local model on the server test set in a single forward pass.

        Returns (accuracy, classification_semantic_entropy).

        In real FL the server never sees client.model directly — it reconstructs
        the local model as w_global + Δ_i.  In this simulation the two are
        equivalent because client.model == w_global + Δ_i after local_train().
        Using the server's public test set is inherent to FedLLMs evaluation.
        """
        model_was_on_cpu = not getattr(client, '_model_on_gpu', False)
        if model_was_on_cpu:
            client.model.to(self.device)
            client._model_on_gpu = True

        try:
            client.model.eval()
            correct = 0
            total = 0
            total_cse = 0.0

            with torch.no_grad():
                for batch in self.test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = client.model(input_ids, attention_mask)

                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                    log_probs = F.log_softmax(outputs, dim=1)
                    probs = log_probs.exp()
                    batch_cse = -(probs * log_probs).sum(dim=1)
                    total_cse += batch_cse.sum().item()

            accuracy = correct / total if total > 0 else 0.0
            cse = total_cse / total if total > 0 else 0.0
        finally:
            if model_was_on_cpu:
                client.model.cpu()
                client._model_on_gpu = False

        return accuracy, cse

    def evaluate_local_accuracy(self, client) -> float:
        """Backward-compatible wrapper; prefer evaluate_local_metrics."""
        acc, _ = self.evaluate_local_metrics(client)
        return acc
    
    def evaluate(self) -> float:
        """
        Evaluate the global model's performance.

        Returns:
            Clean accuracy (float) on the test set
        """
        accuracy, _, _ = self.evaluate_with_loss()
        return accuracy

    def evaluate_with_loss(self) -> Tuple[float, float, Optional[float]]:
        """
        Evaluate the global model's performance in a single pass and also
        compute the Classification Semantic Entropy (CSE) on the SeqCLS head.

        Returns:
            Tuple of (clean_accuracy, global_loss, classification_semantic_entropy_or_none).
            The third value is ``None`` when ``compute_classification_semantic_entropy`` is False.

        The CSE is the mean Shannon entropy of the softmax class distribution
        p(y|x) over the test set. Lower = more confident predictions; under a
        hallucination-inducing attack the model becomes less confident and CSE
        increases. A principled no-generation surrogate for Farquhar-style
        semantic entropy, using the C class labels as the "semantic clusters".
        """
        self.global_model.eval()

        # Evaluate clean accuracy, loss and CSE in one forward pass.
        correct = 0
        total = 0
        total_loss = 0.0
        total_cse = 0.0
        do_cse = self.compute_classification_semantic_entropy

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.global_model(input_ids, attention_mask)

                # Accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Cross-entropy loss (sum over batch for later averaging).
                loss = F.cross_entropy(outputs, labels, reduction='sum')
                total_loss += loss.item()

                if do_cse:
                    # Classification Semantic Entropy (per-sample Shannon entropy,
                    # summed here and averaged at the end).
                    # Use log_softmax for numerical stability.
                    log_probs = F.log_softmax(outputs, dim=1)
                    probs = log_probs.exp()
                    batch_cse = -(probs * log_probs).sum(dim=1)  # (B,)
                    total_cse += batch_cse.sum().item()

        clean_accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        mean_cse: Optional[float]
        if do_cse:
            mean_cse = total_cse / total if total > 0 else 0.0
        else:
            mean_cse = None

        # Record historical metrics.
        self.history['clean_acc'].append(clean_accuracy)
        if 'cse' not in self.history:
            self.history['cse'] = []
        self.history['cse'].append(mean_cse)

        return clean_accuracy, avg_loss, mean_cse
    
    def evaluate_global_loss(self) -> float:
        """
        Evaluate the global model's loss on the test set.
        For efficiency, use evaluate_with_loss() if you also need accuracy.
        
        Returns:
            Global loss (float) on the test set (cross-entropy loss)
        """
        _, loss, _ = self.evaluate_with_loss()
        return loss

    def adaptive_adjustment(self, round_num: int):
        """Adaptively adjust parameters based on historical performance."""
        # Fixed server_lr (no adaptive change)
        pass

    def run_round(self, round_num: int) -> Dict:
        """Execute one round of federated learning - stable version."""
        print(f"\n{'=' * 60}")
        print(f"Round {round_num + 1}/{self.total_rounds}")

        # Track the current round so the defense plugin can use it for history.
        self._current_round = int(round_num)

        # Adaptive adjustment
        self.adaptive_adjustment(round_num)

        # Display current parameters
        print(f"Current Parameters: server_lr={self.server_lr:.2f}")
        print(f"{'=' * 60}")

        # Broadcast the model
        print("📡 Broadcasting the global model...")
        self.broadcast_model()

        # Phase 1: Preparation
        print("\n🔧 Phase 1: Client Preparation")
        for client in self.clients:
            client.set_round(round_num)
            # Use is_attacker attribute instead of isinstance to support both GRMP and ALIE attackers
            if getattr(client, 'is_attacker', False):
                client.prepare_for_round(round_num)

        # Phase 2: Local Training
        print("\n💪 Phase 2: Local Training")
        initial_updates = {}
        for client in self.clients:
            update = client.local_train()
            initial_updates[client.client_id] = update
            print(f"  ✓ Client {client.client_id} completed training")

        # Phase 3: Attacker Camouflage
        print("\n🎭 Phase 3: Attacker Camouflage")
        benign_updates = []
        benign_client_ids = []
        for client_id, update in initial_updates.items():
            client = self.clients[client_id]
            if not getattr(client, 'is_attacker', False):
                benign_updates.append(update)
                benign_client_ids.append(client_id)
        
        print(f"  Captured {len(benign_updates)} benign updates for camouflage.")
        
        # ===== NEW: Store completed attacker updates for coordinated optimization =====
        completed_attacker_updates = {}  # {client_id: update_tensor}
        completed_attacker_client_ids = []  # Keep order
        completed_attacker_data_sizes = {}  # {client_id: claimed_data_size}
        # ==============================================================================
        
        final_updates = {}
        for client_id, update in initial_updates.items():
            client = self.clients[client_id]
            if getattr(client, 'is_attacker', False):
                print(f"  ⚠️ Triggering camouflage logic for Client {client_id}")
                client.receive_benign_updates(benign_updates, client_ids=benign_client_ids)
                
                # ===== NEW: Pass completed attacker updates to current attacker =====
                if completed_attacker_updates:
                    client.receive_attacker_updates(
                        updates=list(completed_attacker_updates.values()),
                        client_ids=completed_attacker_client_ids,
                        data_sizes=completed_attacker_data_sizes
                    )
                # ====================================================================
                
                final_updates[client_id] = client.camouflage_update(update)
                
                # ===== NEW: Store current attacker's update for subsequent attackers =====
                completed_attacker_updates[client_id] = final_updates[client_id]
                completed_attacker_client_ids.append(client_id)
                completed_attacker_data_sizes[client_id] = float(getattr(client, 'claimed_data_size', 1.0))
                # =========================================================================
            else:
                final_updates[client_id] = update

        # Phase 4: Aggregation
        print("\n📊 Phase 4: Model Aggregation")
        # Ensure deterministic order of keys
        sorted_client_ids = sorted(final_updates.keys())
        final_update_list = [final_updates[cid] for cid in sorted_client_ids]
        
        aggregation_log = self.aggregate_updates(final_update_list, sorted_client_ids)

        # Evaluate the global model (compute accuracy, loss and CSE in one pass).
        clean_acc, global_loss, mean_cse = self.evaluate_with_loss()

        # Evaluate per-client local accuracy and CSE (single forward pass each).
        local_accs_this_round = {}
        local_cse_this_round = {}
        for client in self.clients:
            try:
                local_acc, local_cse = self.evaluate_local_metrics(client)
                local_accs_this_round[client.client_id] = local_acc
                local_cse_this_round[client.client_id] = local_cse

                if client.client_id not in self.history['local_accuracies']:
                    self.history['local_accuracies'][client.client_id] = []
                self.history['local_accuracies'][client.client_id].append(local_acc)

                if client.client_id not in self.history['local_cse']:
                    self.history['local_cse'][client.client_id] = []
                self.history['local_cse'][client.client_id].append(local_cse)
            except Exception as e:
                print(f"  ⚠️  Could not evaluate local metrics for client {client.client_id}: {e}")

        # Create log for the current round
        round_log = {
            'round': round_num + 1,
            'clean_accuracy': clean_acc,
            'global_loss': global_loss,
            'classification_semantic_entropy': mean_cse,
            'acc_diff': (abs(clean_acc - self.history['clean_acc'][-2])
                         if len(self.history['clean_acc']) > 1 else 0.0),
            'aggregation': aggregation_log,
            'server_lr': self.server_lr,
            'local_accuracies': local_accs_this_round,
            'local_cse': local_cse_this_round,
        }

        self.log_data.append(round_log)

        # Display results
        print(f"\n📊 Round {round_num + 1} Results:")
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        if len(self.history['clean_acc']) > 1:
            prev_clean = self.history['clean_acc'][-2]
            delta_prev = clean_acc - prev_clean
            best_clean = max(self.history['clean_acc'])
            delta_best = clean_acc - best_clean
            print(f"  ΔClean vs prev: {delta_prev:+.4f}")
            print(f"  ΔClean vs best: {delta_best:+.4f}")
        print(f"  Global Loss: {global_loss:.4f}")
        if mean_cse is not None:
            print(f"  Global CSE: {mean_cse:.4f}")
        else:
            print("  Global CSE: (disabled via config)")

        # Per-client local metrics table
        print(f"  Per-client local metrics (local model on server test set):")
        attacker_ids = {c.client_id for c in self.clients if getattr(c, 'is_attacker', False)}
        for cid in sorted(local_accs_this_round):
            tag = "ATK" if cid in attacker_ids else "BGN"
            acc_v = local_accs_this_round[cid]
            cse_v = local_cse_this_round.get(cid, float('nan'))
            print(f"    [{tag}] Client {cid}: acc={acc_v:.4f}  cse={cse_v:.4f}")

        return round_log
