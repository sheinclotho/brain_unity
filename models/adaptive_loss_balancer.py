"""
Adaptive Loss Balancing for Multi-Task Multi-Modal Learning
============================================================

Implements GradNorm-inspired adaptive loss weighting to handle:
1. EEG-fMRI energy imbalance (10-100x difference)
2. Multi-task learning with varying difficulty
3. Dynamic loss weight adjustment based on training dynamics

Key Features:
- Energy-aware initial task weights (EEG tasks start with higher weight to
  counteract the ~50x lower signal amplitude relative to fMRI)
- Automatic loss weight adaptation after warmup
- Gradient-free weight update using loss magnitudes as proxy

References:
    - GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., 2018)
    - Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveLossBalancer(nn.Module):
    """
    Adaptive loss balancing for multi-modal multi-task learning.

    Dynamically adjusts loss weights based on relative task difficulty, with
    energy-aware initialisation to handle the EEG/fMRI amplitude imbalance
    from the very first training step (including warmup where adaptation is off).
    """
    
    def __init__(
        self,
        task_names: List[str],
        modality_names: Optional[List[str]] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        alpha: float = 1.5,
        update_frequency: int = 10,
        learning_rate: float = 0.025,
        warmup_epochs: int = 5,
        modality_energy_ratios: Optional[Dict[str, float]] = None,
        task_priorities: Optional[Dict[str, float]] = None,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
        pred_weight_floor: float = 0.5,
    ):
        """
        Initialize adaptive loss balancer.

        Args:
            task_names: Task names, e.g. ['recon_eeg', 'recon_fmri', 'pred_eeg', 'pred_fmri'].
                Task names ending with a modality name (e.g. '_eeg') are matched to that
                modality's energy ratio for initial weight seeding.
            modality_names: Modality names, e.g. ['eeg', 'fmri'].
            initial_weights: Explicit initial task weights.  When None (default), weights
                are seeded from the inverse of ``modality_energy_ratios`` so that low-energy
                modalities (EEG) receive a higher initial weight.  This implements the valid
                design intent of the removed ``ModalityGradientScaler`` without any
                post-backward graph manipulation.
            alpha: Restoring force for balancing (higher = more aggressive).
            update_frequency: Update weights every N steps.
            learning_rate: Learning rate for weight updates.
            warmup_epochs: Number of epochs before enabling adaptation.
            modality_energy_ratios: Relative signal energy per modality.
                Default: {'eeg': 1.0, 'fmri': 1.0} (equal after z-score normalisation).
            task_priorities: Initial relative importance per task *type*.
                Keys are task-name prefixes; the matching rule is
                ``task_name.startswith(key + '_')``.
                Built-in prefixes and recommended values:
                  'pred'     — future-signal prediction tasks (pred_*, pred_sig_*).
                               Higher → model focuses on predicting future brain activity.
                               Default 6.0.
                  'recon'    — reconstruction tasks (recon_*).
                               Default 1.0 (baseline).
                  'spectral' — frequency-domain reconstruction tasks (spectral_*).
                               Default 0.5 (auxiliary, should not dominate).
                Extensible: add 'bidir_pred' here when bidirectional prediction is
                implemented — no code changes required.
            min_weight: Minimum allowed weight.
            max_weight: Maximum allowed weight.
            pred_weight_floor: Minimum allowed weight for prediction tasks expressed as
                a log-space offset from their initial value.  After warmup, the GradNorm
                update can reduce pred weights if the latent-space pred losses converge
                faster than reconstruction.  However, low pred loss in latent space does
                NOT guarantee positive pred_r2 in signal space; the floor prevents the
                adaptation mechanism from inadvertently starving the prediction objective.
                Default 0.5: pred task log_weight never drops below
                log(initial_weight) - 0.5, i.e. weight ≥ initial_weight * exp(-0.5) ≈
                initial_weight * 0.6.  Set to 0.0 to disable (original behaviour).
        """
        super().__init__()
        
        self.task_names = task_names
        self.modality_names = modality_names or []
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.pred_weight_floor = pred_weight_floor

        # ── Energy-aware + priority-weighted initial task weights ─────────────
        # Formula per task:  w ∝ task_priority / modality_energy
        #
        # task_priority comes from `task_priorities` (config-driven):
        #   'pred'     → prediction tasks (pred_*, pred_sig_*); default 3.0
        #   'recon'    → reconstruction tasks (recon_*);        default 1.0
        #   'spectral' → spectral tasks (spectral_*);           default 0.5
        #   (add 'bidir_pred' etc. for future task types — no code change needed)
        #
        # modality_energy handles amplitude differences between EEG and fMRI
        # (mostly equal after per-channel z-score normalisation).
        _priorities: Dict[str, float] = task_priorities or {
            'pred': 6.0,      # prediction is primary objective
            'recon': 1.0,     # reconstruction is auxiliary
            'spectral': 0.5,  # spectral is a secondary reconstruction supplement
        }
        if initial_weights is None:
            if modality_energy_ratios is None:
                # After EEG per-channel z-score normalisation (map_eeg_to_graph),
                # both EEG and fMRI have ~unit variance, so the modality energy
                # ratio is ~1:1.  Setting eeg=1.0 / fmri=1.0 means the only
                # differentiation between tasks is the task_priorities below.
                # (Legacy value was eeg=0.02 / fmri=1.0, calibrated for raw MNE
                # Volts.  That is now incorrect because EEG is pre-normalised.)
                modality_energy_ratios = {'eeg': 1.0, 'fmri': 1.0}
            raw: Dict[str, float] = {}
            for name in task_names:
                # Match task name suffix to a known modality (e.g. 'recon_eeg' → 'eeg')
                matched = next((m for m in self.modality_names if name.endswith(m)), None)
                energy = modality_energy_ratios.get(matched, 1.0) if matched else 1.0
                # Match task name prefix to a task type (e.g. 'pred_eeg' → 'pred',
                # 'pred_sig_eeg' → 'pred', 'spectral_eeg' → 'spectral').
                # Sort keys by length DESCENDING so more specific prefixes (e.g.
                # 'pred_sig') are always checked before shorter ones (e.g. 'pred').
                # This gives deterministic longest-prefix-match semantics when a user
                # adds a fine-grained key like 'pred_sig' alongside 'pred'.
                task_type = next(
                    (t for t in sorted(_priorities, key=len, reverse=True)
                     if name.startswith(t + '_')),
                    None,
                )
                task_priority = _priorities.get(task_type, 1.0) if task_type else 1.0
                raw[name] = task_priority / (energy + 1e-8)
            mean_w = sum(raw.values()) / len(raw)
            initial_weights = {k: v / mean_w for k, v in raw.items()}
            logger.debug(f"Energy-seeded initial task weights: {initial_weights}")
        elif modality_energy_ratios is None:
            modality_energy_ratios = {'eeg': 0.02, 'fmri': 1.0}

        # Learnable weights stored in log space for numerical stability.
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.log(torch.tensor(initial_weights.get(name, 1.0))))
            for name in task_names
        })

        # Record the initial log-space weight for each task so that
        # pred_weight_floor can be applied as an absolute lower bound.
        # Stored as a plain Python dict (not a buffer/Parameter) so that it is
        # NOT included in the state_dict — the floor is always anchored to the
        # constructor's priority-weighted initial values, even after checkpoint
        # restore (which would overwrite the learned log_weights but not this).
        self._initial_log_weights: Dict[str, float] = {
            name: math.log(initial_weights.get(name, 1.0))
            for name in task_names
        }

        # Identify which task names are "prediction" tasks (pred_* / pred_sig_*).
        # These are the tasks subject to the pred_weight_floor constraint.
        # Use a set for O(1) membership testing in update_weights().
        self._pred_task_names: set = {
            name for name in task_names if name.startswith('pred')
        }

        # Pre-compute log-space weight bounds as Python floats to avoid creating
        # new GPU tensors on every update_weights() call (one per task per step).
        # Using plain float clamp_ is equivalent and allocation-free.
        self._max_log_weight: float = math.log(max_weight)
        self._min_log_weight: float = math.log(min_weight)

        # Track training dynamics
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('epoch_count', torch.tensor(0, dtype=torch.long))
        
        # Track initial loss values for per-task scale normalisation
        self.register_buffer('initial_losses', torch.zeros(len(task_names), dtype=torch.float32))
        self.register_buffer('initial_losses_set', torch.tensor(False, dtype=torch.bool))
        
    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss.

        Args:
            losses: Dictionary of task losses {task_name: loss_value}

        Returns:
            total_loss: Weighted sum of losses
            weight_dict: Current weights for logging
        """
        # Get current weights.
        # Detach from the autograd graph: log_weights are updated via update_weights()
        # (a manual, gradient-free rule) and are NOT part of the optimizer.  Including
        # them in the backward graph would cause their .grad to accumulate across steps
        # without ever being zeroed by optimizer.zero_grad(), which can trigger
        # "Trying to backward through the graph a second time" in edge cases.
        # Detaching treats the weight scalars as constants for THIS backward pass,
        # which is exactly the intended semantics.
        weights = {name: torch.exp(self.log_weights[name]).detach().clamp(self.min_weight, self.max_weight)
                   for name in self.task_names}
        
        # Capture initial losses on first call (used only by update_weights for
        # convergence-rate tracking — NOT for normalization here).
        # Energy-seeded initial weights already correct for amplitude scale; adding
        # another normalization by L0 would compound the two corrections and create
        # extreme gradient imbalances (e.g. w_eeg/L0_eeg = 50/0.001 = 50,000 vs
        # w_fmri/L0_fmri = 1/0.5 = 2 → EEG dominates by 25,000×).
        if not self.initial_losses_set:
            self.initial_losses = torch.tensor([
                losses.get(name, torch.tensor(0.0)).detach().item()
                for name in self.task_names
            ], device=self.initial_losses.device)
            self.initial_losses_set = torch.tensor(True, device=self.initial_losses.device)

        # Compute weighted losses — energy-seeded weights alone handle amplitude scale.
        weighted_losses = {}
        for name in self.task_names:
            if name in losses:
                weighted_losses[name] = weights[name] * losses[name]
            else:
                weighted_losses[name] = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        total_loss = sum(weighted_losses.values())
        
        # Return weights for logging (already detached above)
        weight_dict = {name: w.item() for name, w in weights.items()}
        
        return total_loss, weight_dict
    
    def update_weights(self, losses: Dict[str, torch.Tensor]):
        """
        Update loss weights based on relative convergence rates.

        Design note: the original GradNorm algorithm calls torch.autograd.grad()
        per task to compute per-task gradient norms.  That requires the computation
        graph to still be alive, but update_weights() is called AFTER backward()
        has already consumed and freed the graph.  Calling autograd.grad() at that
        point raises "Trying to backward through the graph a second time".

        Instead we use normalized loss ratios L(t)/L(0) as a proxy for per-task
        convergence speed (inspired by GradNorm's ˜L_i(t) metric).  Tasks that
        are converging SLOWER relative to their initial loss (higher ratio) receive
        a HIGHER weight so the optimizer allocates more capacity to them.

        Normalization by initial losses is essential here: without it, the raw
        loss magnitude of fMRI (large amplitude → large absolute loss) would always
        exceed EEG loss, causing the balancer to permanently misread fMRI as the
        "harder" task based on scale rather than convergence difficulty.

        Direction:
          rel_loss > 1  (task converging slower than average) → raise weight  ← correct
          rel_loss < 1  (task converging faster than average) → lower weight  ← correct
        """
        # Only update after warmup and at specified frequency
        if self.epoch_count < self.warmup_epochs:
            return
        
        if self.step_count % self.update_frequency != 0:
            self.step_count += 1
            return
        
        self.step_count += 1

        # Gather raw loss scalars (computation graph already freed)
        loss_values = {
            name: losses[name].item()
            for name in self.task_names
            if name in losses
        }

        if not loss_values:
            return

        # Normalize each task's loss by its initial value to get scale-independent
        # convergence ratios.  Without normalization, the raw fMRI loss (high amplitude
        # → large absolute value) is always larger than EEG loss, causing the balancer
        # to permanently misread fMRI as harder based on scale rather than convergence.
        # After normalization: ratio < 1 → converged below initial level (fast); ratio > 1
        # → still above initial level (slow).
        if self.initial_losses_set:
            loss_values_norm = {
                name: loss_val / (self.initial_losses[self.task_names.index(name)].item() + 1e-8)
                for name, loss_val in loss_values.items()
            }
        else:
            # initial_losses not captured yet (shouldn't happen after warmup, but be safe)
            loss_values_norm = loss_values

        avg_loss_norm = sum(loss_values_norm.values()) / len(loss_values_norm)

        with torch.no_grad():
            for name, loss_norm_val in loss_values_norm.items():
                if loss_norm_val < 1e-8:
                    continue

                # Relative convergence rate vs. group average.
                rel_loss = loss_norm_val / (avg_loss_norm + 1e-8)

                # GradNorm-inspired direction: RAISE weight for slower-converging tasks.
                # rel_loss > 1 → task converging slower → needs more gradient signal → +update
                # rel_loss < 1 → task converging faster → can reduce signal → −update
                weight_update = +self.learning_rate * (rel_loss - 1.0)
                weight_update = max(-0.5, min(0.5, weight_update))  # clip

                self.log_weights[name].data += weight_update

                self.log_weights[name].data.clamp_(self._min_log_weight, self._max_log_weight)

                # Pred weight floor: prevent prediction tasks from being reduced
                # below (initial_log_weight - pred_weight_floor).  This ensures
                # the GradNorm mechanism cannot starve the primary objective even
                # when latent-space pred losses converge quickly (which does NOT
                # imply good signal-space pred_r2).
                if self.pred_weight_floor > 0 and name in self._pred_task_names:
                    floor_log = self._initial_log_weights.get(name, 0.0) - self.pred_weight_floor
                    if self.log_weights[name].data.item() < floor_log:
                        self.log_weights[name].data.fill_(floor_log)
        
        if logger.isEnabledFor(logging.DEBUG):
            weights = {name: torch.exp(self.log_weights[name]).item()
                       for name in self.task_names}
            logger.debug(f"Updated loss weights: {weights}")
            logger.debug(f"Normalized loss ratios: {loss_values_norm}")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup tracking."""
        self.epoch_count = torch.tensor(epoch, dtype=torch.long, device=self.epoch_count.device)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights as dictionary."""
        return {
            name: torch.exp(self.log_weights[name]).clamp(self.min_weight, self.max_weight).item()
            for name in self.task_names
        }
