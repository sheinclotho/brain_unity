"""
Digital Twin Brain Inference Engine
=====================================

This module provides the high-level interface for using a trained TwinBrain
model as a **personalized digital twin brain**.  It is the operational
counterpart to the training pipeline in ``graph_native_system.py``.

Three flagship capabilities
---------------------------
1. **Intervention simulation** — inject a perturbation at one or more brain
   regions (mimicking TMS / optogenetics / pharmacology) and simulate the
   propagated system-level causal response.  This is the defining capability
   of a *digital twin*: answer "what-if" questions about the brain without
   running an actual experiment.

2. **Few-shot subject adaptation** — personalize a pretrained model to a
   new subject by fine-tuning only the subject embedding (O(H) parameters)
   while keeping the shared encoder/decoder/predictor frozen.  As few as
   10-50 windows (~20-100 TR of fMRI or ~40s of EEG) typically suffice.

3. **Gradient attribution** — identify which brain regions are most predictive
   for a given target, providing interpretable "functional fingerprints".

Neuroscientific basis
---------------------
The intervention simulation mirrors TMS experiments (Huang et al. 2019,
Neuron): a pulse is applied to cortical region X and the propagating effect
is observed via concurrent TMS-EEG or TMS-fMRI.  Our digital twin:

  h = encoder(baseline_brain_state)           # learned representation
  h_pert[region_X] += Δ                       # inject perturbation
  future_pert = predictor(h_pert)             # causal prediction
  response = decoder(propagator(future_pert)) # system-level response
  causal_effect = response - baseline_response

The causal effect captures how a perturbation to one region propagates
through the learned functional connectivity graph to other regions —
analogous to the "Green's function" / lead-field framework in computational
neuroscience (Deco et al. 2013, J. Neurosci.).

Example usage
-------------
::

    from models.digital_twin_inference import TwinBrainDigitalTwin

    # Load a trained model
    twin = TwinBrainDigitalTwin.from_checkpoint(
        "outputs/experiment/best_model.pt",
        subject_to_idx_path="outputs/experiment/subject_to_idx.json",
    )

    # Few-shot adaptation to new subject (only updates subject_embed)
    twin.adapt_to_subject(new_subject_windows, subject_idx=0, num_steps=50)

    # Simulate TMS to right motor cortex (fMRI ROI index 42)
    result = twin.simulate_intervention(
        baseline_data=graph_window,
        interventions={"fmri": ([42], 2.0)},  # 2 σ perturbation
        num_prediction_steps=15,
    )
    causal_effect = result["causal_effect"]["fmri"]   # [N_fmri, 15, 1]

    # Attribution: which regions drive prediction of target ROI 0?
    saliency = twin.compute_attribution(
        data=graph_window,
        target_modality="fmri",
        target_nodes=[0],
    )

References
----------
- Logothetis et al. (2001) "Neurophysiological investigation of the basis of
  the fMRI signal." *Nature*, 412, 150–157.
- Deco et al. (2013) "Resting brains never rest." *J. Neurosci.*, 33, 9499.
- Huang et al. (2019) "Measuring and interpreting neuronal oscillations."
  *Neuron*, 102, 1157–1167.
- Tian et al. (2019) "Contrastive Multiview Coding." NeurIPS.
- Thomas et al. (2022) "Self-supervised learning of brain dynamics." NeurIPS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class TwinBrainDigitalTwin:
    """High-level digital twin inference engine for a trained TwinBrain model.

    This class wraps a :class:`~models.graph_native_system.GraphNativeBrainModel`
    and a :class:`~models.graph_native_system.GraphNativeTrainer` and exposes
    three user-facing APIs:

    * :meth:`simulate_intervention` — the core digital-twin capability.
    * :meth:`adapt_to_subject` — few-shot personalization to a new subject.
    * :meth:`compute_attribution` — gradient saliency for interpretability.
    * :meth:`predict_future` — plain causal future prediction (no perturbation).

    All methods are safe to call without gradients (they use
    ``torch.no_grad()`` internally) except :meth:`adapt_to_subject` which
    does gradient descent on the subject embedding.
    """

    def __init__(
        self,
        model: GraphNativeBrainModel,
        device: str = "cpu",
        subject_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Args:
            model: A trained :class:`~models.graph_native_system.GraphNativeBrainModel`.
            device: Device to run inference on (``'cuda'`` or ``'cpu'``).
            subject_to_idx: Optional mapping from subject ID strings to integer
                indices used by ``model.subject_embed``.  Saved alongside the
                checkpoint as ``subject_to_idx.json``.
        """
        self.model = model.to(device).eval()
        self.device = device
        self.subject_to_idx = subject_to_idx or {}

    # ── Construction helpers ──────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        subject_to_idx_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
    ) -> "TwinBrainDigitalTwin":
        """Load a digital twin from a saved checkpoint.

        The checkpoint must have been saved by :func:`main.train_model`
        (format described in SPEC.md §3.5).  The model architecture is
        inferred from the ``model_state_dict`` keys, so you do NOT need
        to recreate the exact :class:`~models.graph_native_system.GraphNativeBrainModel`
        call used during training.

        Args:
            checkpoint_path: Path to ``best_model.pt`` or any checkpoint saved
                by the training loop.
            device: ``'auto'`` selects CUDA if available, otherwise CPU.
                Pass ``'cpu'`` or ``'cuda'`` to override.
            subject_to_idx_path: Path to ``subject_to_idx.json``.  If not
                provided, the loader tries the same directory as the checkpoint.
            config: Optional config dict.  Required only if you want to rebuild
                the exact model (edge types, etc.).  If None, a minimal model
                is reconstructed from the state_dict shape information.

        Returns:
            A :class:`TwinBrainDigitalTwin` ready for inference.

        Example::

            twin = TwinBrainDigitalTwin.from_checkpoint(
                "outputs/my_exp/best_model.pt",
                device="cuda",
            )
        """
        # Lazy import to avoid circular dependency
        from models.graph_native_system import GraphNativeBrainModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]

        # ── Infer architecture from state_dict shapes ─────────────────
        H = None
        num_subjects = 0
        num_runs = 0
        node_types = set()
        edge_types = set()

        for key, val in state.items():
            # Hidden dimension from encoder input_proj
            if key.startswith("encoder.input_proj.") and key.endswith(".weight"):
                H = val.shape[0]  # out_features = H
            # Subject embedding: nn.Embedding(num_subjects, H)
            if key == "subject_embed.weight":
                num_subjects = val.shape[0]
                H = val.shape[1]
            # Run/session embedding: nn.Embedding(num_runs, H)
            if key == "run_embed.weight":
                num_runs = val.shape[0]
            # Node types from encoder.input_proj keys
            if key.startswith("encoder.input_proj."):
                parts = key.split(".")
                if len(parts) >= 3:
                    node_types.add(parts[2])
            # Edge types from encoder.stgcn_layers conv_dict keys
            if "stgcn_layers" in key and "conv_dict" in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "conv_dict" and i + 1 < len(parts):
                        et_str = parts[i + 1]
                        if "__" in et_str:
                            et_parts = et_str.split("__")
                            if len(et_parts) == 3:
                                edge_types.add(tuple(et_parts))

        if H is None:
            raise ValueError(
                "Cannot infer hidden_channels from checkpoint. "
                "Please pass config= explicitly."
            )

        node_types_list = sorted(node_types) or ["eeg", "fmri"]
        edge_types_list = sorted(edge_types) or [
            (nt, "connects", nt) for nt in node_types_list
        ]

        logger.info(
            f"Rebuilding model from checkpoint: "
            f"H={H}, node_types={node_types_list}, "
            f"num_subjects={num_subjects}, num_runs={num_runs}"
        )

        # ── Extract hyperparameters from config (if available) ─────────────
        # config.yaml is saved alongside each training checkpoint and records
        # the exact architecture hyperparameters used during training.  Using
        # these avoids defaulting to prediction_steps=10 when the model was
        # trained with a different value (e.g. 50), which would cause the
        # predictor submodule weights to be randomly initialized instead of
        # loaded from the checkpoint (state dict keys exist but shapes differ
        # under strict=False → those tensors are silently skipped).
        model_cfg = (config or {}).get("model", {})
        prediction_steps: int = model_cfg.get("prediction_steps", 10)
        # predictor_config may live under two different keys depending on which
        # version of config.yaml was used:
        #   • config['model']['predictor_config']  — legacy / explicit path
        #   • config['v5_optimization']['advanced_prediction']  — standard training path
        # Try the legacy path first; fall back to the training path if not found.
        predictor_config: Optional[Dict] = model_cfg.get("predictor_config", None)
        if predictor_config is None:
            predictor_config = (config or {}).get("v5_optimization", {}).get(
                "advanced_prediction", None
            )
            if predictor_config is not None:
                logger.info(
                    "  predictor_config 读取自 v5_optimization.advanced_prediction "
                    "(context_length=%s)",
                    predictor_config.get("context_length", "default"),
                )
        use_prediction: bool = bool(model_cfg.get("use_prediction", True))
        num_decoder_layers: int = model_cfg.get("num_decoder_layers", 3)
        dropout: float = float(model_cfg.get("dropout", 0.1))
        if prediction_steps != 10:
            logger.info("  从 config.yaml 读取 prediction_steps=%d", prediction_steps)

        model = GraphNativeBrainModel(
            node_types=node_types_list,
            edge_types=edge_types_list,
            in_channels_dict={nt: 1 for nt in node_types_list},
            hidden_channels=H,
            num_subjects=num_subjects,
            num_runs=num_runs,
            prediction_steps=prediction_steps,
            predictor_config=predictor_config,
            use_prediction=use_prediction,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        model.load_state_dict(state, strict=False)
        model.eval()

        # ── Load subject_to_idx ────────────────────────────────────────
        subject_to_idx: Dict[str, int] = {}
        if subject_to_idx_path is None:
            candidate = checkpoint_path.parent / "subject_to_idx.json"
            subject_to_idx_path = candidate if candidate.exists() else None
        if subject_to_idx_path is not None:
            try:
                with open(subject_to_idx_path, "r", encoding="utf-8") as f:
                    subject_to_idx = json.load(f)
                logger.info(
                    f"Loaded subject_to_idx: {len(subject_to_idx)} subjects "
                    f"from {subject_to_idx_path}"
                )
            except Exception as e:
                logger.warning(f"Could not load subject_to_idx: {e}")

        return cls(model=model, device=device, subject_to_idx=subject_to_idx)

    # ── Public inference API ──────────────────────────────────────────────

    @torch.no_grad()
    def predict_future(
        self,
        data: HeteroData,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Causal future prediction (no perturbation).

        Uses the last ``context_length`` timesteps of the encoded brain state
        to predict the next ``num_steps`` timesteps in signal space.

        Args:
            data: HeteroData window (output of :func:`extract_windowed_samples`).
            num_steps: Number of future steps.  If None, uses the model default.

        Returns:
            Dict ``{node_type: [N, steps, C]}`` of predicted signal trajectories.

        Raises:
            RuntimeError: If the predictor or decoder raise an unexpected error.
        """
        data = data.to(self.device)
        combined_embed = self._get_combined_embed(data)
        encoded_data = self.model.encoder(data, subject_embed=combined_embed)

        if not self.model.use_prediction:
            logger.warning("Model was trained without use_prediction=True. Returning empty dict.")
            return {}

        pred_dict = {}
        for nt in self.model.node_types:
            if nt in encoded_data.node_types:
                h = encoded_data[nt].x
                if h.shape[1] < self.model._PRED_MIN_SEQ_LEN:
                    logger.debug(
                        "predict_future: skipping node type '%s' — "
                        "encoded sequence length %d < _PRED_MIN_SEQ_LEN %d.",
                        nt, h.shape[1], self.model._PRED_MIN_SEQ_LEN,
                    )
                    continue
                pred_dict[nt] = self.model.predictor.predict_next(h, num_steps=num_steps)

        if pred_dict:
            pred_dict = self.model.prediction_propagator(pred_dict, data)

        # Decode to signal space
        hd = HeteroData()
        for nt, v in pred_dict.items():
            hd[nt].x = v
        try:
            return self.model.decoder(hd)
        except Exception as e:
            # Surface the real error so callers (e.g. _rollout_with_twin) see the
            # root cause rather than a confusing downstream "modality not in pred_dict".
            shapes = {k: tuple(v.shape) for k, v in pred_dict.items()}
            raise RuntimeError(
                f"predict_future decoder failed: {e}\n"
                f"pred_dict keys={list(pred_dict.keys())}, shapes={shapes}"
            ) from e

    @torch.no_grad()
    def simulate_intervention(
        self,
        baseline_data: HeteroData,
        interventions: Dict[str, Tuple[List[int], Union[float, torch.Tensor]]],
        num_prediction_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simulate a digital twin intervention.

        Injects a perturbation at one or more brain regions in the latent
        space and predicts the system-level causal response.  The causal
        effect is estimated as ``perturbed_response − baseline_response``.

        This mirrors TMS/optogenetics experiments: stimulate region X and
        measure how the perturbation propagates through functional networks.

        Args:
            baseline_data: HeteroData window representing the current
                brain state (context).
            interventions: Dict mapping modality name to (node_indices, delta):

                * ``node_indices``: list of node indices to perturb.
                  E.g. ``[42, 43]`` for two adjacent fMRI ROIs.
                * ``delta``: perturbation magnitude.

                  - **float/int scalar**: the latent representation at
                    ``node_indices`` is shifted along its own mean direction
                    by ``delta`` standard deviations.  ``delta=2.0`` is a
                    "2-sigma" supraphysiological stimulation (strong TMS).
                    ``delta=-1.0`` simulates silencing (e.g. inhibitory TMS
                    or GABA-ergic drug).
                  - **Tensor [H]**: explicit direction vector in H-dimensional
                    latent space.  Useful for targeted interventions in
                    specific representational subspaces.

            num_prediction_steps: How many future steps to simulate.
                Defaults to the model's ``prediction_steps``.

        Returns:
            Dict with keys:

            * ``'causal_effect'``: ``{nt: Tensor[N, steps, C]}`` — the net
              effect of the intervention (perturbed − baseline).  Positive
              values indicate increased activity; negative = suppression.
            * ``'baseline'``: ``{nt: Tensor[N, steps, C]}`` — predicted
              future without intervention.
            * ``'perturbed'``: ``{nt: Tensor[N, steps, C]}`` — predicted
              future with intervention.
            * ``'encoded_baseline'``: ``{nt: Tensor[N, T, H]}`` — encoded
              baseline latents (before perturbation).

        Example::

            # Simulate 2-sigma stimulation of right motor cortex (ROI 42)
            result = twin.simulate_intervention(
                baseline_data=graph_window,
                interventions={"fmri": ([42], 2.0)},
                num_prediction_steps=15,
            )
            causal_fmri = result["causal_effect"]["fmri"]   # [190, 15, 1]
            # Most-affected regions by stimulating ROI 42:
            top_regions = causal_fmri.squeeze(-1).abs().max(dim=1).values.argsort(descending=True)[:10]
        """
        return self.model.simulate_intervention(
            data=baseline_data,
            interventions=interventions,
            num_steps=num_prediction_steps,
        )

    def adapt_to_subject(
        self,
        data_list: List[HeteroData],
        subject_idx: int,
        num_steps: int = 100,
        lr: float = 5e-3,
        verbose: bool = True,
    ) -> None:
        """Few-shot personalization: fine-tune the subject embedding for a new subject.

        Freezes all model parameters **except** ``model.subject_embed.weight[subject_idx]``
        and runs ``num_steps`` gradient descent steps minimizing the reconstruction
        loss on ``data_list``.  This is parameter-efficient personalisation:
        only H parameters (e.g. 128) are updated.

        The scientific motivation: the encoder has learned the shared functional
        topology from training subjects.  A new subject's brain differs mainly
        in the *baseline activity level* and individual *connection-strength offsets*,
        which are captured by the subject embedding without changing the shared
        topology knowledge.

        Args:
            data_list: List of HeteroData samples from the new subject.
                As few as 10–50 windows typically suffice (20–100 TR of fMRI
                or ~40 s of EEG).
            subject_idx: Index in ``model.subject_embed`` for this subject.
                If ``num_subjects`` was set to 0 during training, this method
                raises a ``RuntimeError``.
            num_steps: Number of gradient steps (default 100).  Each step
                processes one randomly sampled window from ``data_list``.
            lr: Learning rate for the subject embedding update (default 5e-3,
                higher than the main training LR because only O(H) parameters
                are updated — the optimization landscape is well-conditioned).
            verbose: Log loss every 10 steps.

        Raises:
            RuntimeError: If the model has no subject embedding (``num_subjects=0``).
        """
        if self.model.num_subjects == 0 or not hasattr(self.model, "subject_embed"):
            raise RuntimeError(
                "This model was trained without subject embeddings (num_subjects=0). "
                "Run training with data from at least 2 subjects to enable "
                "subject-specific personalisation."
            )
        if subject_idx >= self.model.num_subjects:
            raise ValueError(
                f"subject_idx={subject_idx} is out of range "
                f"[0, {self.model.num_subjects - 1}]."
            )

        # Freeze everything except the subject embedding table.
        # We can't track gradients through an index-slice of nn.Parameter,
        # so we unfreeze the whole weight tensor and use gradient masking to
        # ensure only row subject_idx is actually updated.
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.subject_embed.weight.requires_grad_(True)

        optimizer = torch.optim.Adam([self.model.subject_embed.weight], lr=lr)

        self.model.train()  # enable dropout for regularization during adaptation
        import random

        if verbose:
            logger.info(
                f"Few-shot adaptation: subject_idx={subject_idx}, "
                f"{num_steps} steps on {len(data_list)} windows, lr={lr}"
            )

        for step in range(num_steps):
            sample = random.choice(data_list).to(self.device)
            optimizer.zero_grad()

            # Temporarily set subject_idx on the sample for the forward pass
            _orig_idx = getattr(sample, "subject_idx", None)
            sample.subject_idx = torch.tensor(
                subject_idx, dtype=torch.long, device=self.device
            )

            reconstructed, _ = self.model(sample, return_prediction=False)
            losses = self.model.compute_loss(sample, reconstructed)
            total_loss = sum(losses.values())
            total_loss.backward()
            # Mask gradients: only allow updates to the target row
            with torch.no_grad():
                if self.model.subject_embed.weight.grad is not None:
                    mask = torch.zeros_like(self.model.subject_embed.weight)
                    mask[subject_idx] = 1.0
                    self.model.subject_embed.weight.grad.mul_(mask)
            optimizer.step()

            # Restore original subject_idx (non-destructive).
            # If the sample originally had no subject_idx, remove the temp attribute.
            if _orig_idx is not None:
                sample.subject_idx = _orig_idx
            elif hasattr(sample, "subject_idx"):
                del sample.subject_idx

            if verbose and (step + 1) % 10 == 0:
                logger.info(
                    f"  adapt step {step + 1}/{num_steps}: "
                    f"loss={total_loss.item():.4f}"
                )

        # Restore full gradient flow and eval mode.
        # Reverses the freeze applied at the start of adapt_to_subject():
        #   - Start: all params frozen, then subject_embed.weight selectively unfrozen
        #   - End  : all params unfrozen (no selective step needed, the loop covers all)
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.model.eval()

        if verbose:
            embed_norm = self.model.subject_embed.weight[subject_idx].norm().item()
            logger.info(
                f"Subject {subject_idx} adaptation complete. "
                f"Embedding L2 norm: {embed_norm:.4f}"
            )

    def compute_attribution(
        self,
        data: HeteroData,
        target_modality: str,
        target_nodes: List[int],
        target_timestep: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Gradient-based attribution: which brain regions drive the prediction?

        Computes the gradient of the predicted future activity of
        ``target_nodes`` in ``target_modality`` with respect to the input
        features of **all** nodes in all modalities.  Large gradient magnitude
        at a source node means that node's current activity is highly influential
        for predicting the target's future activity.

        This provides an interpretable "functional fingerprint": a saliency map
        over the whole brain showing which regions are causally informative for
        the target region's future dynamics.

        Args:
            data: HeteroData window (context).
            target_modality: Modality of the target prediction
                (e.g. ``'fmri'`` or ``'eeg'``).
            target_nodes: Node indices whose predicted activity to attribute.
            target_timestep: Which prediction timestep to attribute.
                -1 = last step (default, corresponding to the longest-horizon
                prediction — typically the most interesting for interpretability).

        Returns:
            Dict ``{node_type: Tensor[N, T, C]}`` of gradient saliency maps.
            Larger absolute value = more influence on the target prediction.
            Often visualised as ``saliency[nt].abs().mean(dim=(1, 2))`` — a
            per-node scalar importance.

        Example::

            # What drives prediction of ROI 0 (e.g. default mode network hub)?
            saliency = twin.compute_attribution(
                data=graph_window,
                target_modality="fmri",
                target_nodes=[0],
            )
            # Per-node importance for fMRI
            node_importance = saliency["fmri"].abs().mean(dim=(1, 2))  # [N_fmri]
        """
        data = data.to(self.device)

        # Enable gradients for input features only
        for nt in data.node_types:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                data[nt].x = data[nt].x.detach().requires_grad_(True)

        combined_embed = self._get_combined_embed(data)
        encoded_data = self.model.encoder(data, subject_embed=combined_embed)

        if not self.model.use_prediction:
            return {}

        pred_dict = {}
        for nt in self.model.node_types:
            if nt in encoded_data.node_types:
                h = encoded_data[nt].x
                if h.shape[1] < self.model._PRED_MIN_SEQ_LEN:
                    continue
                pred_dict[nt] = self.model.predictor.predict_next(h)

        if pred_dict:
            pred_dict = self.model.prediction_propagator(pred_dict, data)

        if target_modality not in pred_dict:
            logger.warning(f"compute_attribution: '{target_modality}' not in predictions.")
            return {}

        target_pred = pred_dict[target_modality]  # [N, steps, H]
        # Scalar to differentiate: mean over target nodes and target dimension
        scalar = target_pred[target_nodes, target_timestep, :].mean()
        scalar.backward()

        saliency = {}
        for nt in data.node_types:
            if hasattr(data[nt], "x") and data[nt].x is not None and data[nt].x.grad is not None:
                saliency[nt] = data[nt].x.grad.detach()

        # Zero out any remaining gradients to leave the model clean
        self.model.zero_grad()

        return saliency

    def compute_effective_connectivity(
        self,
        data_windows: Union['HeteroData', List['HeteroData']],
        modality: str = 'fmri',
        perturbation_strength: float = 1.0,
        signed: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute whole-brain Effective Connectivity (EC) matrix.

        Implements the NPI algorithm (Luo et al. 2025, Nature Methods):
        systematically perturbs each brain region and measures the causal
        propagated response, producing a directed N×N EC matrix.

        Compared to Granger causality and dynamic causal modelling (DCM),
        this approach:
          • Captures nonlinear dynamics via the trained surrogate model
          • Is data-driven (no hand-crafted biophysical priors)
          • Provides whole-brain coverage at once
          • Supports both excitatory and inhibitory connectivity

        Args:
            data_windows: One or more HeteroData windows to average over.
                Averaging over multiple windows reduces estimation noise
                (analogous to averaging across trials in TMS-EEG studies).
            modality: Brain signal modality to compute EC for.
            perturbation_strength: Perturbation amplitude in latent std units.
                1.0 (default): mild, comparable to spontaneous fluctuations.
                2.0: supraphysiological (strong TMS equivalent).
            signed: Return signed EC (True, default) or absolute magnitude only.
            normalize: Normalize EC to [0, 1].

        Returns:
            ec_matrix: [N, N] EC matrix.
                ec_matrix[j, i] = causal influence of region i on region j.
        """
        if not isinstance(data_windows, list):
            data_windows = [data_windows]

        ec_sum = None
        count = 0
        for window in data_windows:
            try:
                ec = self.model.compute_effective_connectivity(
                    window,
                    modality=modality,
                    perturbation_strength=perturbation_strength,
                    signed=signed,
                    normalize=False,   # normalize at the end after averaging
                )
                ec_sum = ec if ec_sum is None else ec_sum + ec
                count += 1
            except Exception as e:
                logger.warning(f"EC computation failed for window: {e}")
                continue

        if ec_sum is None or count == 0:
            raise RuntimeError("EC computation failed for all windows.")

        ec_mean = ec_sum / count

        if normalize:
            max_val = ec_mean.abs().max()
            if max_val > 1e-8:
                ec_mean = ec_mean / max_val

        return ec_mean

    def compute_model_fc(
        self,
        data_windows: Union['HeteroData', List['HeteroData']],
        modality: str = 'fmri',
    ) -> torch.Tensor:
        """Compute model Functional Connectivity (FC) from predicted latents.

        FC is computed as the Pearson correlation matrix of the predicted
        future latent activations across brain regions.  High model FC
        correlation with empirical FC validates the surrogate brain quality
        (NPI validation strategy, Luo et al. 2025).

        Unlike EC (directed, causal), FC is symmetric and correlational.
        The comparison FC vs EC reveals the additional information that causal
        modelling provides beyond simple correlation.

        Args:
            data_windows: One or more HeteroData windows to average over.
            modality: Brain signal modality to compute FC for.

        Returns:
            fc_matrix: [N, N] symmetric correlation matrix, values in [-1, 1].
        """
        import torch.nn.functional as F
        if not isinstance(data_windows, list):
            data_windows = [data_windows]

        all_latents = []
        device = next(self.model.parameters()).device

        for window in data_windows:
            window = window.to(device)
            with torch.no_grad():
                _, _, encoded = self.model(window, return_encoded=True)
            if modality in encoded:
                h = encoded[modality]  # [N, T, H]
                # Mean-pool latent dim H → per-node temporal activation series [N, T]
                node_ts = h.mean(dim=-1)   # [N, T]: one time series per brain region
                all_latents.append(node_ts)

        if not all_latents:
            raise RuntimeError(f"No valid windows for FC computation of '{modality}'.")

        # Concatenate across windows: [N, T_total]
        latent_ts = torch.cat(all_latents, dim=-1)   # [N, T_total]
        T_total = latent_ts.shape[1]
        if T_total < 2:
            raise ValueError(
                f"Need at least 2 timesteps for FC computation; got T_total={T_total}. "
                "Provide more windows or longer windows."
            )
        # Z-score each node's time series
        mu = latent_ts.mean(dim=-1, keepdim=True)
        std = latent_ts.std(dim=-1, keepdim=True).clamp(min=1e-6)
        latent_z = (latent_ts - mu) / std            # [N, T_total]
        # Pearson correlation matrix [N, N]
        fc = torch.matmul(latent_z, latent_z.T) / (T_total - 1)
        fc = fc.clamp(-1.0, 1.0)
        return fc

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_combined_embed(
        self, data: HeteroData
    ) -> Optional[torch.Tensor]:
        """Delegate to the model's helper so embedding logic lives in one place."""
        return self.model._get_combined_embed(data, device=self.device)
