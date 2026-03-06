"""
Advanced Multi-Step Prediction
===============================

Implements enhanced multi-step prediction strategies:
1. Hierarchical multi-scale prediction (coarse-to-fine)
2. Transformer-based prediction with long-range attention
3. Stratified window sampling (beginning, middle, end)
4. Uncertainty-aware prediction
5. Autoregressive with teacher forcing schedule

Key Improvements over baseline:
- Better long-range temporal modeling
- Memory-efficient sampling strategy
- Multi-scale temporal hierarchies
- Confidence estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def _transformer_seq2seq_predict(
    predictor: 'TransformerPredictor',
    context: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Seq2seq prediction with a TransformerPredictor.

    Extends the context with ``n_steps`` placeholder tokens (initialised with
    the last context step) and applies a causal mask so each future position
    can only attend to earlier positions (standard autoregressive inference).

    Used by both ``UncertaintyAwarePredictor._base_predict()`` and the
    ``elif self.use_transformer`` branches in
    ``EnhancedMultiStepPredictor.predict_next()`` and ``.forward()``.

    Args:
        predictor: TransformerPredictor instance.
        context: Context tensor ``[batch, ctx_len, input_dim]``.
        n_steps: Number of future steps to return.

    Returns:
        Predicted future tensor ``[batch, n_steps, input_dim]``.
    """
    if n_steps <= 0:
        # Guard against n_steps=0 (or negative):
        # When n_steps=0, the intended slice pred_all[:, -n_steps:, :] becomes
        # pred_all[:, 0:, :] because -0 == 0 in Python, returning the ENTIRE
        # extended context sequence instead of the expected empty tensor.
        return context.new_empty(context.shape[0], 0, context.shape[2])
    future_init = context[:, -1:, :].repeat(1, n_steps, 1)
    x_extended = torch.cat([context, future_init], dim=1)
    seq_len = x_extended.shape[1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=context.device) * float('-inf'),
        diagonal=1,
    )
    pred_all = predictor(x_extended, mask=causal_mask)
    return pred_all[:, -n_steps:, :]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (will be moved to device with module.to(device))
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor for long-range temporal modeling.
    
    Advantages over GRU:
    - Parallel processing
    - Better long-range dependencies
    - Multi-head attention for different temporal scales
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        use_gradient_checkpointing: bool = True,
    ):
        """
        Initialize transformer predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            use_gradient_checkpointing: Use gradient checkpointing for memory
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-normalization for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, input_dim]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Output [batch, seq_len, input_dim]
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            x = torch.utils.checkpoint.checkpoint(
                self.transformer,
                x,
                mask,
                use_reentrant=False
            )
        else:
            x = self.transformer(x, mask=mask)
        
        # Normalize and project output
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x


class HierarchicalPredictor(nn.Module):
    """
    Hierarchical multi-scale prediction.
    
    Predicts at multiple temporal scales:
    - Coarse: long-term trends (low frequency)
    - Medium: intermediate dynamics
    - Fine: short-term fluctuations (high frequency)
    
    Combines predictions in coarse-to-fine manner.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_scales: int = 3,
        scale_factors: Optional[List[int]] = None,
        predictor_type: str = 'transformer',  # 'transformer' or 'gru'
        dropout: float = 0.1,
    ):
        """
        Initialize hierarchical predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for each scale
            num_scales: Number of temporal scales
            scale_factors: Downsampling factors for each scale (e.g., [4, 2, 1])
            predictor_type: Type of predictor ('transformer' or 'gru')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        if scale_factors is None:
            # Default: exponentially decreasing scales
            scale_factors = [2 ** (num_scales - 1 - i) for i in range(num_scales)]
        self.scale_factors = scale_factors
        
        # Create predictor for each scale
        self.predictors = nn.ModuleList()
        for scale in range(num_scales):
            if predictor_type == 'transformer':
                predictor = TransformerPredictor(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=4,
                    num_layers=2,
                    dropout=dropout,
                )
            elif predictor_type == 'gru':
                predictor = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    dropout=dropout,
                    batch_first=True,
                )
            else:
                raise ValueError(f"Unknown predictor type: {predictor_type}")
            
            self.predictors.append(predictor)
        
        # Output projection for GRU predictors: maps hidden_dim → input_dim for
        # autoregressive feedback.  Without this projection, the GRU output
        # (shape [batch, 1, hidden_dim]) would be fed back as the next input
        # (which expects shape [batch, 1, input_dim]).  When hidden_dim ≠ input_dim
        # (the default — hidden_channels × 2 vs hidden_channels) this crashes on
        # the second autoregressive step.  A Linear projection makes the GRU
        # design consistent with the standard seq2seq decoder pattern.
        if predictor_type == 'gru':
            self.gru_output_projs = nn.ModuleList([
                nn.Linear(hidden_dim, input_dim) for _ in range(num_scales)
            ])
        else:
            self.gru_output_projs = None
        
        # Upsampling layers to reconstruct fine scale
        self.upsamplers = nn.ModuleList()
        for i, scale_factor in enumerate(scale_factors[:-1]):  # Skip finest scale
            upsampler = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=scale_factor * 2,
                    stride=scale_factor,
                    padding=scale_factor // 2,
                ),
                # Normalisation after ConvTranspose1d must work for ANY (batch, time) size.
                #
                # Rejected options and why:
                #   nn.LayerNorm(input_dim): normalises the LAST dim; after ConvTranspose1d
                #     the tensor is [batch, input_dim, T_up] so the last dim is T_up, not
                #     input_dim — raises shape mismatch when T_up != input_dim.
                #   nn.BatchNorm1d(input_dim): normalises each of the C channels across
                #     N × L.  With TwinBrain's batch_size=1 and the NPI case where
                #     future_steps_scaled=1, input is [1, input_dim, 1] → N×L = 1 element
                #     per channel → var = 0 → normalised output = 0 for ALL channels.
                #     Coarse-scale prediction becomes a zero tensor → hierarchy silently
                #     degrades to single-scale for the most scientifically relevant NPI
                #     configuration (prediction_steps=1).
                #   nn.InstanceNorm1d(input_dim): normalises per-sample per-channel across L.
                #     With L=1: same var=0 problem as BatchNorm1d.
                #
                # Chosen: nn.GroupNorm(1, input_dim)
                #   Groups the entire channel axis (all input_dim channels) into ONE group per
                #   sample, then normalises across all C×L elements in that group.
                #   For [1, input_dim=128, 1]: uses 128 elements → statistics are well-defined
                #   regardless of batch size or time length.  Equivalent to LayerNorm over
                #   the channel dim applied in (N, C, L) format.
                nn.GroupNorm(1, input_dim),
                nn.GELU(),
            )
            self.upsamplers.append(upsampler)
        
        # Fusion layer to combine multi-scale predictions
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * num_scales, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        future_steps: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Hierarchical prediction.

        Coarse scales (large scale_factor) are skipped when
        ``future_steps // scale_factor == 0`` (e.g. prediction_steps=1 with
        scale_factor=4).  Only scales that can produce at least one future step
        participate in the fusion.  This allows NPI-style "N predict 1"
        configurations without crashing.
        
        Args:
            x: Context [batch, context_len, input_dim]
            future_steps: Number of future steps to predict
            
        Returns:
            prediction: Final prediction [batch, future_steps, input_dim]
            scale_predictions: Predictions at each scale (only valid scales)
        """
        batch_size = x.shape[0]
        
        scale_predictions = []
        
        # Predict at each scale
        for i, (predictor, scale_factor) in enumerate(zip(self.predictors, self.scale_factors)):
            # Number of future steps at this (downsampled) scale.
            # Bug fix: integer division may produce 0 when prediction_steps is
            # small (e.g. prediction_steps=1, scale_factor=4 → 1//4 = 0).
            # Repeating 0 future slots creates an empty tensor and the
            # transformer / GRU have nothing to predict.
            # Fix: clamp to at least 1 so each scale always predicts ≥1 step.
            future_steps_scaled = max(1, future_steps // scale_factor)

            # Downsample context
            if scale_factor > 1:
                # Guard: avg_pool1d requires context_len >= scale_factor to
                # produce at least 1 output step.  output_len formula:
                #   floor((L - kernel) / stride) + 1
                # When L < kernel: output_len ≤ 0 → PyTorch raises "Output size
                # is too small" *inside* avg_pool1d (before we could check the
                # result shape).  This is triggered by the combination:
                #   _PRED_MIN_SEQ_LEN=4, _PRED_CONTEXT_RATIO=2/3
                #   → T_ctx = int(4 * 2/3) = 2 < scale_factor=4
                # Fix: check BEFORE calling avg_pool1d.  When the context is
                # too short for this scale, substitute a zero prediction so the
                # fusion layer always receives a full [B, future_steps, input_dim]
                # tensor.  The fusion layer learns to down-weight zero-filled
                # scales (they carry no signal) without disrupting training.
                if x.shape[1] < scale_factor:
                    logger.debug(
                        f"HierarchicalPredictor scale {i} (factor={scale_factor}): "
                        f"context length {x.shape[1]} < scale_factor → using zero "
                        f"prediction for this scale (fusion will learn to down-weight it)."
                    )
                    pred_up = torch.zeros(
                        x.shape[0], future_steps, self.input_dim, device=x.device
                    )
                    scale_predictions.append(pred_up)
                    continue

                # Average pooling to downsample
                x_down = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale_factor,
                    stride=scale_factor,
                ).transpose(1, 2)
            else:
                x_down = x

            # Predict at this scale
            if isinstance(predictor, TransformerPredictor):
                # Delegate to the module-level seq2seq helper — same logic as the
                # standalone _transformer_seq2seq_predict: extend context with
                # future placeholders, apply causal mask, slice last n steps.
                pred = _transformer_seq2seq_predict(predictor, x_down, future_steps_scaled)
            else:
                # For GRU, predict autoregressively.
                # Pass the per-scale output projection so the GRU hidden output
                # (hidden_dim) is projected back to input_dim before being fed
                # as the next-step input (standard seq2seq decoder pattern).
                output_proj = (
                    self.gru_output_projs[i]
                    if self.gru_output_projs is not None else None
                )
                pred = self._autoregressive_predict(
                    predictor, x_down, future_steps_scaled, output_proj=output_proj
                )
            
            # Upsample to finest resolution if needed
            if i < len(self.upsamplers):
                # Transpose for Conv1d: [batch, channels, time]
                pred_up = self.upsamplers[i](pred.transpose(1, 2)).transpose(1, 2)
                
                # Adjust length if needed
                if pred_up.shape[1] != future_steps:
                    pred_up = F.interpolate(
                        pred_up.transpose(1, 2),
                        size=future_steps,
                        mode='linear',
                        align_corners=False,
                    ).transpose(1, 2)
            else:
                # Finest scale: ensure output has exactly future_steps steps.
                # When future_steps_scaled was clamped from 0→1 for a scale where
                # scale_factor == 1 (no upsampling), pred already has 1 step; if
                # future_steps > 1 we repeat the last predicted step.
                if pred.shape[1] < future_steps:
                    repeat_count = future_steps - pred.shape[1]
                    pred_up = torch.cat(
                        [pred, pred[:, -1:, :].repeat(1, repeat_count, 1)], dim=1
                    )
                else:
                    pred_up = pred[:, :future_steps, :]
            
            scale_predictions.append(pred_up)
        
        # Fuse multi-scale predictions
        # Stack along feature dimension
        multi_scale = torch.cat(scale_predictions, dim=-1)  # [batch, future_steps, input_dim * num_scales]
        
        # Fusion
        final_prediction = self.fusion(multi_scale)
        
        return final_prediction, scale_predictions
    
    def _autoregressive_predict(
        self,
        predictor: nn.GRU,
        context: torch.Tensor,
        num_steps: int,
        output_proj: Optional[nn.Linear] = None,
    ) -> torch.Tensor:
        """Autoregressive prediction with GRU.

        Args:
            predictor: GRU module (input_size=input_dim, hidden_size=hidden_dim).
            context: Context sequence [batch, T, input_dim].
            num_steps: Number of autoregressive steps to generate.
            output_proj: Optional Linear(hidden_dim → input_dim) projection.
                When hidden_dim ≠ input_dim this projection is *required*:
                without it, the GRU output (shape [B, 1, hidden_dim]) is fed
                back as the next input (which expects [B, 1, input_dim]),
                causing a shape mismatch crash on step 2.
        """
        # Get initial hidden state
        _, hidden = predictor(context)
        
        # Start with last context value
        current = context[:, -1:, :]
        
        predictions = []
        for _ in range(num_steps):
            # Predict next step
            output, hidden = predictor(current, hidden)
            
            # Project GRU hidden output (hidden_dim) back to input_dim.
            # The projected tensor serves BOTH purposes:
            #   1. Accumulated into predictions (must be input_dim for upsamplers/fusion)
            #   2. Fed back as next-step input (must be input_dim for GRU input_size)
            # Round 3 fixed (2) but left (1) accumulating raw hidden_dim output,
            # causing crashes in upsamplers (ConvTranspose1d expects input_dim) and
            # in the fusion layer (Linear(input_dim * num_scales, ...) expects input_dim).
            projected = output_proj(output) if output_proj is not None else output
            predictions.append(projected)
            current = projected
        
        # Concatenate predictions
        return torch.cat(predictions, dim=1)


class StratifiedWindowSampler:
    """
    Stratified window sampling for memory-efficient training.
    
    Instead of using only beginning and end windows (baseline),
    samples windows from:
    - Beginning (early dynamics)
    - Middle (intermediate patterns)
    - End (recent context)
    
    This provides better coverage of temporal patterns.
    """
    
    def __init__(
        self,
        context_length: int = 50,
        prediction_steps: int = 10,
        num_windows: int = 3,
        sampling_strategy: str = 'uniform',  # 'uniform', 'random', or 'adaptive'
    ):
        """
        Initialize stratified window sampler.
        
        Args:
            context_length: Length of context window
            prediction_steps: Number of steps to predict
            num_windows: Number of windows to sample
            sampling_strategy: How to sample windows
        """
        self.context_length = context_length
        self.prediction_steps = prediction_steps
        self.num_windows = num_windows
        self.sampling_strategy = sampling_strategy
        
        self.window_length = context_length + prediction_steps
    
    @staticmethod
    def _uniform_start_indices(num_windows: int, max_start: int) -> List[int]:
        """Return ``num_windows`` evenly-spaced start indices in ``[0, max_start]``."""
        if num_windows == 1:
            return [max_start // 2]
        return [int(i * max_start / (num_windows - 1)) for i in range(num_windows)]

    def sample_windows(
        self,
        sequence: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Sample windows from sequence.
        
        Args:
            sequence: Full sequence [batch, time, features]
            importance_weights: Optional importance weights for adaptive sampling
            
        Returns:
            windows: List of (context, target, start_idx) tuples
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Check if sequence is long enough
        if seq_len < self.window_length:
            # Use full sequence
            context = sequence
            target = sequence[:, -self.prediction_steps:, :]
            return [(context, target, 0)]
        
        # Compute possible start indices
        max_start = seq_len - self.window_length
        
        if self.sampling_strategy == 'uniform':
            start_indices = self._uniform_start_indices(self.num_windows, max_start)
        
        elif self.sampling_strategy == 'random':
            # Random sampling
            start_indices = torch.randint(0, max_start + 1, (self.num_windows,)).tolist()
        
        elif self.sampling_strategy == 'adaptive':
            # Adaptive sampling based on importance weights
            if importance_weights is None:
                # Fall back to uniform
                start_indices = self._uniform_start_indices(self.num_windows, max_start)
            else:
                # Sample based on importance
                # importance_weights: [time]
                # Compute importance for each possible window
                window_importance = []
                for start in range(max_start + 1):
                    end = start + self.window_length
                    importance = importance_weights[start:end].mean().item()
                    window_importance.append(importance)
                
                # Sample windows proportional to importance (ensure same device as sequence)
                window_importance = torch.tensor(window_importance, dtype=torch.float32, device=sequence.device)
                probs = F.softmax(window_importance, dim=0)
                start_indices = torch.multinomial(probs, self.num_windows, replacement=False).tolist()
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Extract windows
        windows = []
        for start_idx in start_indices:
            end_idx = start_idx + self.window_length
            window = sequence[:, start_idx:end_idx, :]
            
            # Split into context and target
            context = window[:, :self.context_length, :]
            target = window[:, self.context_length:, :]
            
            windows.append((context, target, start_idx))
        
        return windows


class UncertaintyAwarePredictor(nn.Module):
    """
    Uncertainty-aware prediction with confidence estimation.
    
    Estimates both:
    - Prediction mean (expected value)
    - Prediction uncertainty (confidence)
    
    Uses uncertainty for:
    - Loss weighting (downweight uncertain predictions)
    - Confidence-based ensembling
    - Uncertainty quantification for downstream tasks
    """
    
    def __init__(
        self,
        base_predictor: nn.Module,
        input_dim: int,
        uncertainty_method: str = 'gaussian',  # 'gaussian' or 'dropout'
        num_mc_samples: int = 10,
    ):
        """
        Initialize uncertainty-aware predictor.
        
        Args:
            base_predictor: Base prediction model
            input_dim: Input dimension
            uncertainty_method: Method for uncertainty estimation
            num_mc_samples: Number of MC samples for dropout-based uncertainty
        """
        super().__init__()
        
        self.base_predictor = base_predictor
        self.input_dim = input_dim
        self.uncertainty_method = uncertainty_method
        self.num_mc_samples = num_mc_samples
        
        if uncertainty_method == 'gaussian':
            # Predict mean and log variance
            self.uncertainty_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, input_dim),
            )
    
    def _base_predict(
        self,
        x: torch.Tensor,
        future_steps: Optional[int],
    ) -> torch.Tensor:
        """Dispatch a single forward call to the base predictor.

        Each base predictor type has a different calling convention:
        - HierarchicalPredictor: forward(x, future_steps) → (pred, scales)
        - TransformerPredictor:  forward(x_extended, mask) → full_seq
          Requires seq2seq extension (context + future placeholders + causal mask).
          Simply calling forward(x) returns the same-length output [B, ctx_len, H],
          NOT the desired [B, future_steps, H].
        - nn.GRU and others: forward(x) → output (used directly)

        This helper is called from the gaussian and dropout paths in forward()
        to avoid duplicating the dispatch logic at every call site.
        """
        if isinstance(self.base_predictor, HierarchicalPredictor):
            pred, _ = self.base_predictor(x, future_steps)
            return pred
        elif isinstance(self.base_predictor, TransformerPredictor):
            # Delegate to the module-level helper that handles seq2seq extension.
            # future_steps=None is only possible when called without a valid step
            # count; we fall back to the context length as a safe default.
            n_steps = future_steps if future_steps is not None else x.shape[1]
            return _transformer_seq2seq_predict(self.base_predictor, x, n_steps)
        else:
            # nn.GRU (and any future predictor type that is not HierarchicalPredictor
            # or TransformerPredictor): direct call.
            #
            # Defensive tuple-unpack: nn.GRU.forward(x) returns (output, h_n).
            # Returning the tuple directly would crash every downstream consumer
            # that expects a Tensor.  UncertaintyAwarePredictor can be instantiated
            # directly (outside of EnhancedMultiStepPredictor) in unit tests or
            # future extensions, so we unpack here regardless of the outer guard.
            # (Note: EnhancedMultiStepPredictor.__init__ already raises ValueError
            # for the specific combination that would put a raw GRU here, so this
            # guard handles direct-instantiation scenarios only.)
            result = self.base_predictor(x)
            if isinstance(result, tuple):
                return result[0]  # GRU: (output, h_n) → output tensor only
            return result

    def forward(
        self,
        x: torch.Tensor,
        future_steps: Optional[int] = None,
        return_uncertainty: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Context [batch, context_len, input_dim]
            future_steps: Number of future steps to predict
            return_uncertainty: Whether to return uncertainty
            
        Returns:
            prediction_mean: Mean prediction [batch, future_steps, input_dim]
            prediction_std: Uncertainty (None if not requested)
        """
        if self.uncertainty_method == 'gaussian':
            # Forward through base predictor
            prediction_mean = self._base_predict(x, future_steps)
            
            if not return_uncertainty:
                return prediction_mean, None
            
            # Predict log variance.
            # Clamp before exponentiation to prevent overflow to Inf:
            # exp(0.5 * 10) ≈ 148, exp(0.5 * -10) ≈ 0.007 — both are reasonable
            # std bounds for normalised neural signals.  Without this clamp, large
            # positive log_var produces Inf std → NaN in the NLL loss → NaN gradients.
            log_var = torch.clamp(self.uncertainty_head(prediction_mean), min=-10.0, max=10.0)
            prediction_std = torch.exp(0.5 * log_var)
            
            return prediction_mean, prediction_std
        
        elif self.uncertainty_method == 'dropout':
            # MC dropout for uncertainty estimation
            if not return_uncertainty:
                # Single forward pass
                return self._base_predict(x, future_steps), None
            
            # Multiple forward passes with dropout
            # Save training state so we don't corrupt eval mode when this is
            # called during validation.  .train() enables Dropout and switches
            # BatchNorm to use mini-batch statistics; if we leave the module in
            # training mode after the MC samples the subsequent eval-mode forward
            # passes in validate() will use wrong normalisation statistics.
            was_training = self.base_predictor.training
            self.base_predictor.train()  # Enable dropout for MC sampling
            try:
                predictions = []
                for _ in range(self.num_mc_samples):
                    predictions.append(self._base_predict(x, future_steps))
            finally:
                self.base_predictor.train(was_training)  # restore original state
            
            # Compute mean and std
            predictions = torch.stack(predictions, dim=0)  # [num_samples, batch, time, dim]
            prediction_mean = predictions.mean(dim=0)
            prediction_std = predictions.std(dim=0)
            
            return prediction_mean, prediction_std
        
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.
        
        Args:
            predictions: Predicted values [batch, time, dim]
            targets: Target values [batch, time, dim]
            uncertainty: Predicted uncertainty (std) [batch, time, dim]
            
        Returns:
            loss: Uncertainty-weighted loss
        """
        if uncertainty is None or self.uncertainty_method == 'dropout':
            # Standard MSE loss
            return F.mse_loss(predictions, targets)
        
        # Gaussian negative log likelihood
        # Loss = 0.5 * (log(2π) + log(σ²) + (y - μ)² / σ²)
        # Simplified: 0.5 * (log(σ²) + (y - μ)² / σ²)
        
        squared_error = (predictions - targets) ** 2
        log_var = 2.0 * torch.log(uncertainty + 1e-8)
        
        # NLL loss
        nll_loss = 0.5 * (log_var + squared_error / (uncertainty ** 2 + 1e-8))
        
        return nll_loss.mean()


class EnhancedMultiStepPredictor(nn.Module):
    """
    Complete enhanced multi-step prediction system.
    
    Combines:
    - Hierarchical multi-scale prediction
    - Transformer-based modeling
    - Stratified window sampling
    - Uncertainty estimation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        context_length: int = 50,
        prediction_steps: int = 10,
        use_hierarchical: bool = True,
        use_transformer: bool = True,
        use_uncertainty: bool = True,
        num_scales: int = 3,
        num_windows: int = 3,
        sampling_strategy: str = 'uniform',
        dropout: float = 0.1,
    ):
        """
        Initialize enhanced multi-step predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            context_length: Length of context window
            prediction_steps: Number of steps to predict
            use_hierarchical: Use hierarchical multi-scale prediction
            use_transformer: Use transformer (else GRU)
            use_uncertainty: Enable uncertainty estimation
            num_scales: Number of scales for hierarchical prediction
            num_windows: Number of windows for stratified sampling
            sampling_strategy: Window sampling strategy
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.prediction_steps = prediction_steps
        self.use_hierarchical = use_hierarchical
        self.use_transformer = use_transformer  # needed to dispatch predict_next/forward correctly
        self.use_uncertainty = use_uncertainty

        # Validate unsupported configuration early — before allocating any parameters.
        #
        # use_hierarchical=False, use_transformer=False, use_uncertainty=True is broken
        # because:
        #   1. UncertaintyAwarePredictor._base_predict calls self.base_predictor(x) on
        #      nn.GRU, which returns (output, h_n) tuple, not a tensor.
        #   2. Even if unpacked, nn.GRU returns [N, ctx_len, hidden_dim] — the wrong
        #      temporal length (ctx_len, not future_steps) and wrong feature dimension
        #      (hidden_dim=256, not input_dim=128).
        #   3. uncertainty_head expects input_dim; it receives hidden_dim → crash.
        #
        # Supported alternatives:
        #   • Set use_uncertainty=False  (simplest — uncertainty estimation is disabled
        #     by default and its head does not receive training gradients anyway).
        #   • Set use_transformer=True   (flat TransformerPredictor + UAP — fixed R6).
        #   • Set use_hierarchical=True  (HierarchicalPredictor + UAP — fully supported).
        if not use_hierarchical and not use_transformer and use_uncertainty:
            raise ValueError(
                "Unsupported predictor configuration: "
                "use_hierarchical=False + use_transformer=False + use_uncertainty=True. "
                "The raw nn.GRU base predictor is incompatible with "
                "UncertaintyAwarePredictor (wrong return type, temporal length and "
                "feature dimension).  Choose one of: "
                "(a) use_uncertainty=False, "
                "(b) use_transformer=True, "
                "(c) use_hierarchical=True."
            )

        # Create base predictor
        if use_hierarchical:
            predictor_type = 'transformer' if use_transformer else 'gru'
            base_predictor = HierarchicalPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_scales=num_scales,
                predictor_type=predictor_type,
                dropout=dropout,
            )
        elif use_transformer:
            base_predictor = TransformerPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=4,
                dropout=dropout,
            )
        else:
            # Simple GRU baseline
            base_predictor = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                dropout=dropout,
                batch_first=True,
            )
        
        # Wrap with uncertainty estimation if enabled
        if use_uncertainty:
            self.predictor = UncertaintyAwarePredictor(
                base_predictor=base_predictor,
                input_dim=input_dim,
                uncertainty_method='gaussian',
            )
        else:
            self.predictor = base_predictor
        
        # Output projection for the simple-GRU (not use_hierarchical) case.
        # HierarchicalPredictor has its own per-scale projections (gru_output_projs).
        # Here we need a single projection for the top-level autoregressive loop
        # in predict_next() / forward() when the predictor IS the raw nn.GRU.
        # Condition: self.predictor == nn.GRU iff not hierarchical, not transformer,
        # and not wrapped in UncertaintyAwarePredictor.
        if not use_hierarchical and not use_transformer and not use_uncertainty:
            self.gru_output_proj = nn.Linear(hidden_dim, input_dim)
        else:
            self.gru_output_proj = None
        
        # Window sampler
        self.window_sampler = StratifiedWindowSampler(
            context_length=context_length,
            prediction_steps=prediction_steps,
            num_windows=num_windows,
            sampling_strategy=sampling_strategy,
        )

    # ── Private prediction helpers ──────────────────────────────────────────

    def _gru_rollout(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressive GRU rollout: context → ``n_steps`` future steps.

        Encapsulates the per-step loop shared by ``_predict_from_context()`` and
        the ``forward()`` method, eliminating the copy-paste.

        Args:
            context: ``[batch, T, input_dim]``
            n_steps: Number of autoregressive steps.

        Returns:
            ``[batch, n_steps, input_dim]``
        """
        _, hidden = self.predictor(context)
        current = context[:, -1:, :]
        preds = []
        for _ in range(n_steps):
            output, hidden = self.predictor(current, hidden)
            projected = self.gru_output_proj(output) if self.gru_output_proj is not None else output
            preds.append(projected)
            current = projected
        return torch.cat(preds, dim=1)

    def _predict_from_context(self, context: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Core prediction dispatch: context ``[B, T, H]`` → ``[B, pred_steps, H]``.

        Dispatches to the appropriate predictor based on the configuration flags
        set at construction time.  Does **not** return uncertainty estimates —
        call the ``UncertaintyAwarePredictor`` directly when std is needed.

        This helper is called by both ``predict_next()`` and the non-uncertainty
        branch of ``forward()``, eliminating the duplicated 4-branch dispatch.

        Args:
            context: Context tensor ``[batch, T, H]``.
            num_steps: Override the number of future steps to generate.
                When ``None`` (default), uses ``self.prediction_steps``.
                Pass a smaller value (e.g. ``aligned_steps = min(prediction_steps,
                T_fut)``) to avoid generating unsupervised prediction steps.
        """
        steps = num_steps if num_steps is not None else self.prediction_steps
        if self.use_hierarchical:
            pred, _ = self.predictor(context, steps)
        elif self.use_transformer:
            pred = _transformer_seq2seq_predict(self.predictor, context, steps)
        else:
            pred = self._gru_rollout(context, steps)
        return pred

    def forward(
        self,
        sequences: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with stratified sampling.
        
        Args:
            sequences: Full sequences [batch, time, input_dim]
            importance_weights: Optional importance for adaptive sampling [time]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            predictions: Predictions for all sampled windows
            targets: Targets for all sampled windows
            uncertainties: Uncertainty estimates (None if not requested)
        """
        # Sample windows
        windows = self.window_sampler.sample_windows(sequences, importance_weights)
        
        # Predict for each window
        all_predictions = []
        all_targets = []
        all_uncertainties = [] if return_uncertainty else None
        
        for context, target, _ in windows:
            # uncertainty path needs std; all other paths delegate to _predict_from_context
            if self.use_uncertainty:
                pred_mean, pred_std = self.predictor(
                    context,
                    future_steps=self.prediction_steps,
                    return_uncertainty=return_uncertainty,
                )
            else:
                pred_mean = self._predict_from_context(context)
                pred_std = None
            
            all_predictions.append(pred_mean)
            all_targets.append(target)
            if pred_std is not None:
                all_uncertainties.append(pred_std)
        
        # Stack predictions
        predictions = torch.stack(all_predictions, dim=0)  # [num_windows, batch, time, dim]
        targets = torch.stack(all_targets, dim=0)
        uncertainties = torch.stack(all_uncertainties, dim=0) if all_uncertainties else None
        
        return predictions, targets, uncertainties
    
    def predict_next(self, h: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Causal one-shot prediction: context → future.

        Uses the last ``min(context_length, T)`` timesteps as context and
        predicts the next ``prediction_steps`` timesteps.  Unlike ``forward()``
        (which samples multiple windows for self-supervised training), this
        method produces a single causally-aligned prediction — the correct
        entry point for training loss, inference, and evaluation.

        Scientific analogy (NPI-style): the model receives the last
        ``context_length`` brain-state vectors and produces the next
        ``prediction_steps`` brain-state vectors.  With
        ``context_length=70, prediction_steps=1`` this is exactly the
        "70 predict 1" paradigm; with
        ``context_length=200, prediction_steps=10`` it predicts 10 future
        steps from 200 past steps.

        Args:
            h: Latent sequence [batch, T, H].  Only the last
               ``min(context_length, T)`` steps are used.
            num_steps: Override number of predicted steps.  When ``None``
               (default) uses ``self.prediction_steps``.  Pass
               ``aligned_steps = min(prediction_steps, T_fut)`` from the
               caller to avoid generating unsupervised prediction steps for
               modalities whose future window is shorter than
               ``prediction_steps`` (e.g. fMRI T_fut=17 with
               prediction_steps=50 → pass num_steps=17 to save 66% of
               predictor computation and activation memory).

        Returns:
            pred: Predicted latent [batch, num_steps, H]
        """
        steps = num_steps if num_steps is not None else self.prediction_steps
        ctx_len = min(self.context_length, h.shape[1])
        if self.context_length < steps:
            # Only warn when the configured context_length cap is shorter than
            # the requested steps — that is a genuine configuration mismatch.
            # Having h.shape[1] < steps is expected and by-design
            # in windowed-sampling mode (T_ctx = window × 2/3 < prediction_steps).
            logger.warning(
                f"predict_next: context_length ({self.context_length}) is configured "
                f"shorter than prediction steps ({steps}). "
                f"Consider increasing context_length or reducing prediction_steps."
            )
        context = h[:, -ctx_len:, :]   # [batch, ctx_len, H] — causally bounded

        if self.use_uncertainty:
            # UncertaintyAwarePredictor: we don't need std here, request mean only.
            pred, _ = self.predictor(
                context,
                future_steps=steps,
                return_uncertainty=False,
            )
        else:
            pred = self._predict_from_context(context, num_steps=steps)

        return pred

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prediction loss.
        
        Args:
            predictions: Predictions [num_windows, batch, time, dim]
            targets: Targets [num_windows, batch, time, dim]
            uncertainties: Uncertainty estimates (optional)
            
        Returns:
            loss: Mean loss across all windows
        """
        if self.use_uncertainty and uncertainties is not None:
            # Uncertainty-aware loss for each window
            losses = [
                self.predictor.compute_loss(predictions[i], targets[i], uncertainties[i])
                for i in range(predictions.shape[0])
            ]
            return torch.stack(losses).mean()
        else:
            # Standard MSE loss
            return F.mse_loss(predictions, targets)
