"""
Complete Graph-Native Training System
=====================================

Full reimagination of TwinBrain training pipeline:
1. Graph-native from data to prediction
2. No unnecessary conversions
3. Spatial-temporal modeling on graphs
4. Efficient and interpretable

This is a COMPLETE standalone system, not just optimization modules.
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import random
from torch_geometric.data import HeteroData
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

# Import AMP components if available (for mixed precision training)
AMP_AVAILABLE = False
USE_NEW_AMP_API = False
try:
    # Try new API: autocast from torch.amp (requires device_type parameter)
    from torch.amp import autocast
    # GradScaler might still be in torch.cuda.amp in some PyTorch versions
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler
    AMP_AVAILABLE = True
    USE_NEW_AMP_API = True
except ImportError:
    try:
        # Fallback to old API if new one not available
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
        USE_NEW_AMP_API = False
    except ImportError:
        AMP_AVAILABLE = False
        autocast = None
        GradScaler = None

from .graph_native_mapper import GraphNativeBrainMapper
from .graph_native_encoder import GraphNativeEncoder, SpatialTemporalGraphConv
from .adaptive_loss_balancer import AdaptiveLossBalancer
from .eeg_channel_handler import EnhancedEEGHandler
from .advanced_prediction import EnhancedMultiStepPredictor

logger = logging.getLogger(__name__)


class GraphNativeDecoder(nn.Module):
    """
    Graph-native decoder that reconstructs temporal signals.
    
    Decodes from latent graph representations back to temporal signals
    WITHOUT breaking graph structure.
    """
    
    def __init__(
        self,
        node_types: List[str],
        hidden_channels: int,
        out_channels_dict: Dict[str, int],
        num_layers: int = 3,
        temporal_upsample: Optional[int] = None,
    ):
        """
        Initialize decoder.
        
        Args:
            node_types: Node types to decode
            hidden_channels: Hidden dimension from encoder
            out_channels_dict: Output channels per node type
            num_layers: Number of decoding layers
            temporal_upsample: Upsample factor for temporal dimension
        """
        super().__init__()
        
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.out_channels_dict = out_channels_dict
        self.temporal_upsample = temporal_upsample
        
        # Temporal deconvolution per node type
        self.temporal_deconv = nn.ModuleDict()
        
        for node_type in node_types:
            layers = []
            current_dim = hidden_channels
            
            # Stack of convolution layers
            for i in range(num_layers):
                out_dim = hidden_channels // (2 ** i) if i < num_layers - 1 else out_channels_dict[node_type]
                use_stride2 = bool(temporal_upsample and i == 0)
                
                if use_stride2:
                    # Temporal upsampling: ConvTranspose1d doubles T (stride=2 is correct here)
                    layers.append(nn.ConvTranspose1d(
                        current_dim, out_dim, kernel_size=4, stride=2, padding=1,
                    ))
                else:
                    # Feature transform at same temporal resolution.
                    # Conv1d(kernel_size=3, padding=1) preserves T exactly.
                    # ConvTranspose1d(kernel_size=4, stride=1, padding=1) silently adds 1
                    # to T per layer — causing shape mismatch in compute_loss.
                    layers.append(nn.Conv1d(
                        current_dim, out_dim, kernel_size=3, padding=1,
                    ))
                
                if i < num_layers - 1:
                    layers.append(nn.GroupNorm(1, out_dim))
                    layers.append(nn.ReLU())
                
                current_dim = out_dim
            
            self.temporal_deconv[node_type] = nn.Sequential(*layers)
    
    def forward(self, encoded_data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Decode to temporal signals.
        
        Args:
            encoded_data: Encoded HeteroData with features [N, T, H]
            
        Returns:
            Reconstructed signals per node type [N, T', C_out]
        """
        reconstructed = {}
        
        for node_type in self.node_types:
            if node_type in encoded_data.node_types:
                x = encoded_data[node_type].x  # [N, T, H]
                N, T, H = x.shape
                
                # Reshape for conv: [N, T, H] -> [N, H, T]
                x = x.permute(0, 2, 1)
                
                # Apply temporal deconvolution
                x_recon = self.temporal_deconv[node_type](x)  # [N, C_out, T']
                
                # Reshape back: [N, C_out, T'] -> [N, T', C_out]
                x_recon = x_recon.permute(0, 2, 1)
                
                reconstructed[node_type] = x_recon
        
        return reconstructed


class GraphPredictionPropagator(nn.Module):
    """
    System-level graph propagation for predictions.

    After the per-node temporal predictor generates initial predictions
    ``{node_type: [N, pred_steps, H]}``, this module propagates them
    through the brain's connectivity graph so that a change in one brain
    region's predicted activity influences all its connected neighbours.

    Neuroscientific motivation
    --------------------------
    The brain operates as a **coupled dynamical system**.  Stimulating
    region A (or altering its activity) does not affect A in isolation —
    the perturbation propagates via white-matter tracts (structural
    connectivity) and correlated activity (functional connectivity) to
    regions B, C, D, …  Without this module the predictor treats every
    node independently, so stimulating a single ROI only alters that
    ROI's trajectory.  With this module the prediction becomes a
    system-level forecast: changed activity at A ripples through the
    graph to its neighbours in proportion to their edge weights.

    Architecture
    ------------
    ``num_prop_layers`` rounds of ST-GCN message passing applied to the
    prediction tensor ``[N, pred_steps, H]``, treating ``pred_steps`` as
    the temporal dimension (analogous to ``T`` in the encoder).  The
    same edge connectivity used during encoding is reused here, so the
    propagator is consistent with the learned representations.
    Residual connections + LayerNorm keep gradients healthy.
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_channels: int,
        num_prop_layers: int = 2,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
        temporal_chunk_size: Optional[int] = None,
    ):
        """
        Args:
            node_types: Node types present in the graph.
            edge_types: Edge types (same list as the encoder).
            hidden_channels: Feature dimension H (must match predictor output).
            num_prop_layers: Number of graph-propagation rounds.  2 is
                sufficient to reach 2-hop neighbours (e.g. A→B→C), which
                covers the typical cortical relay distance.
            dropout: Dropout rate.
            use_gradient_checkpointing: Free intermediate activations inside
                each ST-GCN propagate() call to reduce peak GPU memory during
                backward.  Mirrors the same flag on the main encoder.
            temporal_chunk_size: Passed to SpatialTemporalGraphConv; bounds
                peak GPU memory during backward recomputation per chunk.
        """
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types

        # Stack of ST-GCN layers (temporal_kernel_size=1 ≡ 1×1-conv so
        # consecutive prediction steps are NOT mixed — each step is
        # propagated independently, preserving temporal structure).
        self.prop_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_prop_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict['__'.join(edge_type)] = SpatialTemporalGraphConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    temporal_kernel_size=1,
                    use_attention=True,
                    use_spectral_norm=True,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    dropout=dropout,
                    temporal_chunk_size=temporal_chunk_size,
                )
            self.prop_layers.append(nn.ModuleDict(conv_dict))
            self.layer_norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_channels)
                for nt in node_types
            }))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """
        Propagate predictions through graph connectivity.

        Args:
            pred_dict: Initial per-node predictions
                ``{node_type: Tensor[N, pred_steps, H]}``.
            data: HeteroData providing ``edge_index`` (and optionally
                ``edge_attr``) for each edge type.  The node features are
                NOT read from here — only the topology is used.

        Returns:
            Propagated predictions ``{node_type: Tensor[N, pred_steps, H]}``.
            Nodes with no inbound edges in a given layer pass through
            unchanged (same as the encoder residual convention).
        """
        x_dict = {k: v for k, v in pred_dict.items()}

        # Build edge index / attr dicts once (same across layers)
        edge_index_dict: Dict[Tuple, torch.Tensor] = {}
        edge_attr_dict: Dict[Tuple, torch.Tensor] = {}
        for edge_type in self.edge_types:
            if edge_type in data.edge_types:
                edge_index_dict[edge_type] = data[edge_type].edge_index
                if hasattr(data[edge_type], 'edge_attr'):
                    edge_attr_dict[edge_type] = data[edge_type].edge_attr

        for conv_dict, layer_norm_dict in zip(self.prop_layers, self.layer_norms):
            x_dict_new: Dict[str, torch.Tensor] = {}

            for node_type, x in x_dict.items():
                messages = []

                for edge_type in self.edge_types:
                    src, _, dst = edge_type
                    if dst != node_type:
                        continue
                    if edge_type not in edge_index_dict:
                        continue
                    if src not in x_dict:
                        continue

                    x_src = x_dict[src]
                    edge_index = edge_index_dict[edge_type]
                    edge_attr = edge_attr_dict.get(edge_type, None)

                    conv = conv_dict['__'.join(edge_type)]
                    N_src, N_dst = x_src.shape[0], x.shape[0]
                    msg = conv(x_src, edge_index, edge_attr, size=(N_src, N_dst))

                    # Align pred_steps dimension if source and destination
                    # modalities have different prediction lengths.
                    T_dst = x.shape[1]
                    if msg.shape[1] != T_dst:
                        msg = F.interpolate(
                            msg.permute(0, 2, 1),   # [N, H, T_src]
                            size=T_dst,
                            mode='linear',
                            align_corners=False,
                        ).permute(0, 2, 1)           # [N, T_dst, H]

                    messages.append(msg)

                # Residual aggregation — same convention as the encoder:
                # average incoming messages and add as a residual.
                if messages:
                    x_new = x + self.dropout(sum(messages) / len(messages))
                else:
                    x_new = x  # passthrough when no inbound edges

                # Per-timestep LayerNorm
                N, T, H = x_new.shape
                x_new = layer_norm_dict[node_type](x_new.view(N * T, H)).view(N, T, H)
                x_dict_new[node_type] = x_new

            x_dict = x_dict_new

        return x_dict


class GraphNativeBrainModel(nn.Module):
    """
    Complete graph-native brain model.

    End-to-end architecture:
    1. Mapper: Build graph from brain data
    2. Encoder: Spatial-temporal encoding on graph
    3. Predictor: Per-node future prediction (temporal)
    4. Propagator: System-level graph propagation of predictions
    5. Decoder: Reconstruct signals

    NO sequence conversions - pure graph operations throughout.

    System-level prediction
    -----------------------
    Step 3 runs ``EnhancedMultiStepPredictor`` independently on each
    node's latent time series.  Step 4 (``GraphPredictionPropagator``)
    then propagates those predictions through the brain's connectivity
    graph.  This means stimulating a single brain region affects all
    its neighbours in proportion to their edge weights, producing a
    scientifically meaningful whole-brain forecast rather than an
    isolated single-region forecast.
    """
    
    # 潜空间预测切分比例：前 CONTEXT_RATIO 作为 context，余下部分为 future target。
    # context ≥ 2/3 确保 StratifiedWindowSampler 能在 context 内取到至少 1 个完整窗口。
    _PRED_CONTEXT_RATIO: float = 2 / 3
    # 潜空间预测所需最小序列长度（= 保证 T_ctx ≥ 1 且 T_fut ≥ 1）
    _PRED_MIN_SEQ_LEN: int = 4

    # ── Loss helper static methods ──────────────────────────────────────────

    @staticmethod
    def _spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Frequency-domain magnitude spectrum loss (scale-normalised).

        Scientific basis: EEG signals have characteristic spectral bands
        (theta 4-7 Hz, alpha 8-12 Hz, beta 13-30 Hz) and fMRI has slow
        BOLD fluctuations (~0.01-0.1 Hz).  Matching magnitude spectra
        encourages the model to preserve these rhythms, complementing the
        time-domain Huber/MSE loss which only penalises pointwise amplitude.

        Implementation: Real FFT along the time axis (dim=1), MSE between
        magnitude spectra (phase is NOT penalised — phases of stochastic
        neural signals are noisy and hard to match precisely).

        Scale normalisation: divides by T so the spectral MSE has the same
        order-of-magnitude as Huber/MSE on the raw signal.  Without this,
        by Parseval's theorem (sum |X[k]|² = T·σ²) the spectral MSE grows
        as O(T), dwarfing the time-domain loss for long sequences.

        Args:
            pred:   [N, T, C]
            target: [N, T, C]  (must have T ≥ 2)
        Returns:
            Scalar spectral MSE (scale-normalised by T).
        """
        T = pred.shape[1]
        # cuFFT in half precision (float16/AMP) only supports power-of-two
        # signal sizes.  Cast to float32 to support arbitrary lengths such
        # as T=300 used by default.  The resulting scalar loss is then
        # compatible with the float16 computation graph via autocast.
        pred_f = pred.float()
        # Detach target: tgt_f is the ground-truth signal (data[node_type].x),
        # used as a fixed reference for the magnitude spectrum comparison.
        # Gradients should flow only through pred_f (the model reconstruction).
        # Detaching also avoids creating an unnecessary float32 copy of the
        # target in the autograd graph, reducing peak memory during backward.
        tgt_f  = target.detach().float()
        pred_fft = torch.fft.rfft(pred_f, dim=1)   # [N, T//2+1, C] complex
        tgt_fft  = torch.fft.rfft(tgt_f,  dim=1)   # [N, T//2+1, C] complex
        return F.mse_loss(pred_fft.abs(), tgt_fft.abs()) / T

    @staticmethod
    def _pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pearson correlation loss: mean(1 − r) across nodes and channels.

        Directly optimises the temporal *pattern* matching component of R².
        Unlike MSE/Huber (which penalise absolute deviations), this loss is
        scale-invariant: it only cares that peaks and troughs in pred align
        with those in target.

        R² = 1 − SS_res/SS_tot ≈ r² for linear predictions, so maximising
        Pearson correlation is equivalent to maximising R².

        Args:
            pred, target: [N, T, C]  (T ≥ 2 required)
        Returns:
            Scalar loss in [0, 2]  (0 = perfect positive correlation).
        """
        N, T, C = pred.shape
        # Cast to float32 before computing Pearson correlation.
        # In AMP training (float16), the clamp(min=1e-8) guard below is
        # ineffective: 1e-8 is below the float16 minimum representable
        # positive normal value (~6.1e-5) and rounds to 0, so the clamp
        # becomes clamp(min=0) and does not prevent division by zero.
        # Near-zero norms (common in early training when the model outputs
        # almost-zero predictions) then produce NaN → training divergence.
        # float32 has sufficient precision so 1e-8 is correctly enforced.
        p = pred.float().reshape(N * C, T)    # [N*C, T]
        # Detach target: the reference signal is supervision only; gradients
        # should flow through *pred* alone.  Detaching also prevents the
        # float32 reshape of the target from being retained in the autograd
        # graph as a saved activation, reducing peak memory during backward.
        t = target.detach().float().reshape(N * C, T)  # [N*C, T]
        # Centre
        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)
        # Pearson r per (node, channel)
        numer = (p * t).sum(dim=1)
        denom = (p.norm(dim=1) * t.norm(dim=1)).clamp(min=1e-8)
        r = (numer / denom).clamp(-1.0, 1.0)  # [N*C]
        return (1.0 - r).mean()

    @staticmethod
    def _cross_modal_align_loss(
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-modal latent alignment via cosine similarity.

        Forces the global mean representation of the source modality (EEG)
        and destination modality (fMRI) to be close in the shared latent
        space.  This directly optimises the neurovascular coupling (NVC)
        alignment: EEG high-gamma power and fMRI BOLD share an underlying
        neural signal source (Logothetis et al. 2001), so their mean
        latent codes should point in similar directions.

        Reference:
            Tian et al. (2019) Contrastive Multiview Coding (CMC).
            Thomas et al. (2022) Self-supervised learning of brain dynamics
              from broad neuroimaging data.

        Args:
            h_src: Source modality latent [N_src, T, H] (e.g. EEG).
            h_dst: Destination modality latent [N_dst, T, H] (e.g. fMRI).

        Returns:
            Scalar loss in [0, 2]  (0 = perfect alignment).
        """
        # Mean-pool over nodes and time to obtain a global representation [H].
        # Cast to float32: cosine similarity with near-zero norms in float16
        # produces NaN (same issue as _pearson_loss).
        # Cast back to the original dtype at the end so the returned scalar is
        # compatible with the loss computation graph in AMP (float16) training.
        z_src = h_src.float().mean(dim=(0, 1))   # [H]
        z_dst = h_dst.float().mean(dim=(0, 1))   # [H]
        z_src = F.normalize(z_src, dim=0)
        z_dst = F.normalize(z_dst, dim=0)
        loss = 1.0 - (z_src * z_dst).sum()
        return loss.to(h_src.dtype)

    @staticmethod
    def _info_nce_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """InfoNCE / Contrastive Predictive Coding (CPC) loss.

        Forces the predictor to produce *discriminative* latent representations
        that uniquely identify the correct future timestep, rather than predicting
        the unconditional mean of the future distribution.

        Why this matters for pred_r2:
          Pure regression (Huber / MSE) can be minimised by predicting a smooth
          average of plausible futures, which yields low MSE but negative R² (worse
          than predicting the global mean).  InfoNCE requires the predictor to
          distinguish the correct future (positive) from all other timesteps
          (negatives), which is only possible if the predicted representation
          carries unique information about that specific future window.
          This is the same principle as CPC (Oord 2018), wav2vec 2.0 (Baevski 2020),
          and CLIP (Radford 2021): contrastive discrimination forces the model to
          capture the *what* of future states, not just the *expected average*.

        Scientific basis:
          The brain exhibits a rich structure of temporal sequences; even brief
          windows of EEG / fMRI carry unique spectro-spatial fingerprints that
          distinguish them from other windows in the same session (Stringer et al.
          2019, Nature; Engemann et al. 2022, NeuroImage).  InfoNCE exploits this
          structure as a training signal.

        Implementation:
          Flatten (node, step) pairs to [N*S, H]; L2-normalise; compute cosine
          similarity matrix [N*S, N*S] / τ; positives are on the diagonal.
          Symmetric bidirectional cross-entropy (pred→target and target→pred) as
          in CLIP for additional stability.

        References:
            Oord et al. (2018) "Representation Learning with CPC." NeurIPS wrkshp.
            Baevski et al. (2020) "wav2vec 2.0." NeurIPS.
            Radford et al. (2021) "CLIP." ICML.

        Args:
            pred:        [N, S, H] predicted latent representations.
            target:      [N, S, H] actual future latent representations.
                         Stop-gradient applied internally (target is supervision).
            temperature: Softmax temperature τ.  Smaller → more discriminative but
                         harder to optimise.  0.1 is standard for brain imaging data.
        Returns:
            Scalar InfoNCE loss.  Lower = predictor is more discriminative.
        """
        N, S, H = pred.shape
        n_items = N * S
        if n_items < 2:
            # Degenerate case: cannot form positive/negative pairs.
            return pred.new_zeros(1).squeeze()
        # Flatten and L2-normalise for cosine similarity.
        # Cast to float32: cosine similarity with float16 and near-zero norms
        # can produce NaN (same issue as _pearson_loss, _cross_modal_align_loss).
        pred_flat   = F.normalize(pred.float().reshape(n_items, H), dim=-1)   # [n_items, H]
        # Detach target: it is fixed supervision (analogous to a label).
        # Gradients flow only through pred, not through target.
        target_flat = F.normalize(target.detach().float().reshape(n_items, H), dim=-1)  # [n_items, H]
        # Cosine similarity matrix, scaled by temperature.
        logits = torch.matmul(pred_flat, target_flat.T) / temperature  # [n_items, n_items]
        # Positive pairs lie on the diagonal: pred[i] ↔ target[i].
        labels = torch.arange(n_items, device=pred.device)
        # Symmetric InfoNCE (CLIP-style): average both directions for stability.
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
        return loss.to(pred.dtype)

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        use_prediction: bool = True,
        prediction_steps: int = 10,
        dropout: float = 0.1,
        loss_type: str = 'mse',
        use_gradient_checkpointing: bool = False,
        predictor_config: Optional[Dict] = None,
        use_dynamic_graph: bool = False,
        k_dynamic_neighbors: int = 10,
        num_subjects: int = 0,
        use_spectral_loss: bool = False,
        temporal_chunk_size: Optional[int] = None,
        use_cross_modal_align: bool = True,
        pred_step_weight_gamma: float = 1.0,
        num_runs: int = 0,
        use_info_nce: bool = False,
        info_nce_temperature: float = 1.0,
        use_reconstruction_loss: bool = True,
    ):
        """
        Initialize complete model.
        
        Args:
            node_types: Node types (e.g., ['fmri', 'eeg'])
            edge_types: Edge types
            in_channels_dict: Input channels per modality
            hidden_channels: Hidden dimension
            num_encoder_layers: Encoder depth
            num_decoder_layers: Decoder depth
            use_prediction: Enable future prediction
            prediction_steps: Steps to predict ahead
            dropout: Dropout rate
            loss_type: Loss function type ('mse', 'huber', 'smooth_l1')
            use_gradient_checkpointing: Free intermediate activations per timestep
                to avoid MemoryError on long sequences (trades memory for compute)
            predictor_config: Optional dict from config['v5_optimization']['advanced_prediction'].
                Keys: context_length, use_hierarchical, use_transformer, use_uncertainty,
                num_scales, num_windows, sampling_strategy.  Defaults used when None.
                context_length (default 200): number of past timesteps the predictor
                uses as input.  Set context_length=70 + prediction_steps=1 for the
                NPI "70-predict-1" paradigm.
            use_dynamic_graph: Enable self-iterating graph structure learning
                (DynamicGraphConstructor per ST-GCN layer, intra-modal edges only).
            k_dynamic_neighbors: k-nearest neighbours for the dynamic adjacency.
            num_subjects: Total number of subjects in the dataset.
                > 0 creates nn.Embedding(num_subjects, hidden_channels) for
                per-subject personalization (AGENTS.md §九 Gap 2).
                Each subject learns a unique latent offset added to all node
                features after input projection, enabling the model to capture
                individual brain differences without a separate model per subject.
                At inference time, fine-tune only the subject embedding (frozen
                encoder) for few-shot personalization.
                0 = disabled (default, backward-compatible).
            use_spectral_loss: Add a frequency-domain magnitude spectrum loss
                (FFT-based) in addition to the time-domain Huber/MSE loss.
                Encourages the model to preserve neural spectral structure
                (EEG alpha/beta/theta; fMRI slow BOLD fluctuations), improving
                both recon R² and pred R² for oscillatory brain signals.
                Default False for backward compatibility; True recommended for
                EEG-heavy experiments.
            temporal_chunk_size: Number of timesteps processed per propagate()
                call in SpatialTemporalGraphConv.  Bounds peak GPU memory during
                backward recomputation.  None = full T (original behaviour).
                Set to 64 to cut peak message-tensor memory by ~4× vs T=300.
            use_cross_modal_align: Add a cross-modal latent alignment loss
                (cosine similarity between mean EEG and fMRI representations).
                Encourages the shared latent space to reflect neurovascular
                coupling (Logothetis 2001): EEG and fMRI encode the same
                underlying neural activity and should be close in latent space.
                Reference: CMC (Tian et al. 2019), Thomas et al. 2022.
                Default True (V5.43); set False to disable.
            pred_step_weight_gamma: Exponential weight for later prediction
                steps.  step t gets weight exp(gamma * t / T_fut), normalised
                to mean 1.  Larger gamma → more emphasis on far-future
                steps (harder to predict).  0.0 = uniform (original behaviour).
                Default 1.0 (V5.43).
            num_runs: Total number of recording sessions/runs in the dataset.
                > 0 creates nn.Embedding(num_runs, hidden_channels) for
                per-run session embedding.  Each run (recording session) learns
                a unique latent offset added to node features alongside the
                subject embedding, capturing session-level drift such as scanner
                noise, fatigue, and cognitive state variation between sessions.
                This enables cross-session knowledge transfer and improves
                generalization across recording sessions.
                Initialized to zero so it has no effect at the start of training
                and only diverges from zero as session-specific patterns emerge.
                0 = disabled (default, backward-compatible).
            use_info_nce: Add InfoNCE contrastive prediction loss alongside the
                regression prediction losses (pred_*, pred_sig_*).  Prevents
                mean-prediction collapse (predicting the unconditional mean of the
                future distribution), which is the primary cause of negative pred_r2
                in pure MSE/Huber training.  InfoNCE forces the predictor to produce
                *discriminative* representations that uniquely identify each future
                time window.  Default True (V5.47).
                Reference: Oord et al. (2018) CPC; Baevski et al. (2020) wav2vec 2.0.
            info_nce_temperature: Softmax temperature τ for InfoNCE loss.  Smaller τ
                is more discriminative (sharper similarity distribution) but harder
                to optimise on small datasets.  0.1 = standard value for large
                datasets (SimCLR, wav2vec 2.0), but too aggressive for brain signals
                with small datasets (n_items ≈ 3230 → loss_scale ≈ 81 → 50× gradient
                suppression of pred_sig).  1.0 = V5.50 recommended value for
                EEG/fMRI with low BOLD autocorrelation (ρ ≈ 0.23), balancing
                InfoNCE and pred_sig gradients at ≈1.3:1.  Default 1.0 (V5.50).
            use_reconstruction_loss: Include the signal reconstruction loss
                (recon_{nt}) in the training objective.  Default True.
                Set False to train on prediction only (pred_* + pred_sig_* +
                pred_nce_*), freeing the entire gradient budget for prediction.
                The encoder and decoder are still trained through the prediction
                path (pred_sig_loss → decoder → predictor), so disabling
                reconstruction does NOT orphan any module weights.
                Recommended setting: True (default) for general use; False when
                pred_r2 is the sole objective and the model is too small to
                simultaneously learn both tasks well.
        """
        super().__init__()
        
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.use_prediction = use_prediction
        self.prediction_steps = prediction_steps
        self.loss_type = loss_type
        self.num_subjects = num_subjects
        self.num_runs = num_runs
        self.use_spectral_loss = use_spectral_loss
        self.use_cross_modal_align = use_cross_modal_align
        self.pred_step_weight_gamma = pred_step_weight_gamma
        self.use_info_nce = use_info_nce
        self.info_nce_temperature = info_nce_temperature
        self.use_reconstruction_loss = use_reconstruction_loss

        # 被试特异性嵌入 (AGENTS.md §九 Gap 2)
        # num_subjects > 0: each subject gets a learnable [H] offset added to
        # all node features after input projection, capturing individual differences.
        # Initialized with small Gaussian noise (std=0.02) to avoid disrupting
        # the shared pre-training signal in early epochs.
        if num_subjects > 0:
            self.subject_embed = nn.Embedding(num_subjects, hidden_channels)
            nn.init.normal_(self.subject_embed.weight, std=0.02)

        # 会话/run 特异性嵌入 (V5.44, 跨会话预测支持)
        # num_runs > 0: each recording session gets a learnable [H] offset,
        # capturing session-level drift (scanner noise, fatigue, cognitive state
        # variation).  Initialized to zero so the first epoch is identical to
        # num_runs=0 — the offset only emerges as the optimizer finds session
        # patterns.  Added to node features AFTER subject_embed, so the two
        # form an additive decomposition:
        #   x_proj += subject_embed   (who: stable individual identity)
        #   x_proj += run_embed       (when: transient session state)
        if num_runs > 0:
            self.run_embed = nn.Embedding(num_runs, hidden_channels)
            nn.init.zeros_(self.run_embed.weight)  # zero-init: no session bias initially
        
        # Encoder: Graph-native spatial-temporal encoding
        self.encoder = GraphNativeEncoder(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_layers=num_encoder_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dropout=dropout,
            use_dynamic_graph=use_dynamic_graph,
            k_dynamic_neighbors=k_dynamic_neighbors,
            temporal_chunk_size=temporal_chunk_size,
        )
        
        # Decoder: Reconstruct temporal signals
        self.decoder = GraphNativeDecoder(
            node_types=node_types,
            hidden_channels=hidden_channels,
            out_channels_dict=in_channels_dict,
            num_layers=num_decoder_layers,
        )
        
        # Predictor: per-node temporal future prediction (optional)
        if use_prediction:
            pred_cfg = predictor_config or {}
            self.predictor = EnhancedMultiStepPredictor(
                input_dim=hidden_channels,
                hidden_dim=hidden_channels * 2,
                context_length=pred_cfg.get('context_length', 200),
                prediction_steps=prediction_steps,
                use_hierarchical=pred_cfg.get('use_hierarchical', True),
                use_transformer=pred_cfg.get('use_transformer', True),
                use_uncertainty=pred_cfg.get('use_uncertainty', False),
                num_scales=pred_cfg.get('num_scales', 3),
                num_windows=pred_cfg.get('num_windows', 3),
                sampling_strategy=pred_cfg.get('sampling_strategy', 'uniform'),
            )

            # Prediction propagator: system-level graph diffusion of predictions.
            # After EnhancedMultiStepPredictor generates independent per-node
            # trajectories, GraphPredictionPropagator runs num_prop_layers rounds
            # of graph message-passing so that stimulating one brain region
            # propagates its predicted activity to connected regions.
            # num_prop_layers=2 lets signals travel ≥2 hops (A→B→C), covering
            # the typical cortical relay distance.
            self.prediction_propagator = GraphPredictionPropagator(
                node_types=node_types,
                edge_types=edge_types,
                hidden_channels=hidden_channels,
                num_prop_layers=pred_cfg.get('num_prop_layers', 2),
                dropout=dropout,
                use_gradient_checkpointing=use_gradient_checkpointing,
                temporal_chunk_size=temporal_chunk_size,
            )
    
    def _get_combined_embed(
        self,
        data: HeteroData,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Return the combined subject+run embedding for *data*, or None.

        Centralises the three formerly-duplicated subject/run embedding lookup
        blocks that existed in ``forward()``, ``simulate_intervention()``, and
        ``compute_effective_connectivity()``.

        Design rules:
        * Index tensors are created directly on *device* to avoid a CPU→GPU
          transfer that occurred when ``torch.tensor()`` was called without
          a ``device`` argument and then moved with ``.to(device)`` afterwards.
        * If *device* is None the device is inferred from the embedding weights
          (i.e. wherever the model lives).
        * Out-of-range subject indices emit a one-time warning and are clamped
          (stale-cache protection: see AGENTS.md §三 "缓存命中路径 continue 绕过必要副作用").

        Args:
            data: HeteroData window carrying optional ``subject_idx`` / ``run_idx``.
            device: Target device for index tensors.  Defaults to the device of
                ``self.subject_embed`` or ``self.run_embed`` weights.

        Returns:
            Combined [H] embedding tensor, or ``None`` when both embeddings are
            disabled (backward-compatible — encoder handles ``None`` gracefully).
        """
        if device is None:
            if self.num_subjects > 0:
                device = self.subject_embed.weight.device
            elif self.num_runs > 0:
                device = self.run_embed.weight.device

        subject_embed: Optional[torch.Tensor] = None
        if self.num_subjects > 0 and hasattr(data, 'subject_idx') and data.subject_idx is not None:
            s_idx = data.subject_idx
            if not isinstance(s_idx, torch.Tensor):
                s_idx = torch.tensor(s_idx, dtype=torch.long, device=device)
            else:
                s_idx = s_idx.to(device)
            s_val = s_idx.item()  # one CPU sync; reused for check and warning message
            if s_val < 0 or s_val >= self.num_subjects:
                logger.warning(
                    f"subject_idx={s_val} out of range [0, {self.num_subjects - 1}]. "
                    f"Likely a stale graph cache (num_subjects changed).  "
                    f"Clear the cache and re-run to fix.  Falling back to subject 0."
                )
                s_idx = s_idx.clamp(0, self.num_subjects - 1)
            subject_embed = self.subject_embed(s_idx)  # [H]

        run_embed: Optional[torch.Tensor] = None
        if self.num_runs > 0 and hasattr(data, 'run_idx') and data.run_idx is not None:
            r_idx = data.run_idx
            if not isinstance(r_idx, torch.Tensor):
                r_idx = torch.tensor(r_idx, dtype=torch.long, device=device)
            else:
                r_idx = r_idx.to(device)
            r_idx = r_idx.clamp(0, self.num_runs - 1)
            run_embed = self.run_embed(r_idx)  # [H]

        # Additive decomposition: subject (who) + run (when).
        # Either or both may be None; encoder handles None gracefully.
        if subject_embed is not None and run_embed is not None:
            return subject_embed + run_embed
        if run_embed is not None:
            return run_embed
        return subject_embed  # may be None

    def _decode_latents_to_signal(
        self,
        pred_dict: Dict[str, torch.Tensor],
        error_context: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Decode a dict of latent tensors {node_type: [N, steps, H]} to signal space.

        Shared helper used by ``simulate_intervention``,
        ``compute_effective_connectivity``, and ``compute_loss``.  Avoids the
        formerly-triplicated HeteroData construction + try/except wrapping.

        Args:
            pred_dict: Mapping from node type to latent tensor [N, steps, H].
            error_context: Optional prefix for debug log messages on failure.

        Returns:
            Mapping from node type to decoded signal [N, steps, C], or an empty
            dict if ``pred_dict`` is empty or the decoder raises an exception.
        """
        if not pred_dict:
            return {}
        hd = HeteroData()
        for nt, v in pred_dict.items():
            hd[nt].x = v
        try:
            return self.decoder(hd)
        except Exception as e:
            if error_context:
                logger.debug(f"{error_context}: {e}")
            return {}

    def forward(
        self,
        data: HeteroData,
        return_prediction: bool = False,
        return_encoded: bool = False,
    ) -> Tuple:
        """
        Forward pass with input validation.
        
        Args:
            data: Input HeteroData with temporal features
            return_prediction: Whether to return future predictions
            return_encoded: Whether to return latent encoded representations
                {node_type: h[N, T, H]} — needed by compute_loss for
                the latent-space self-supervised prediction loss.
            
        Returns:
            When return_encoded=False (default):
                (reconstructed, predictions) — 2-tuple
            When return_encoded=True:
                (reconstructed, predictions, encoded_dict) — 3-tuple
            reconstructed: {node_type: tensor[N, T, C]}
            predictions: {node_type: tensor[N, steps, H]} or None
            encoded_dict: {node_type: tensor[N, T, H]}
        """
        # Input validation (use explicit checks, not assertions)
        for node_type in self.node_types:
            if node_type in data.node_types and hasattr(data[node_type], 'x'):
                x = data[node_type].x
                if x.ndim != 3:
                    raise ValueError(f"Expected [N, T, C] for {node_type}, got {x.shape}")
                if torch.isnan(x).any():
                    raise ValueError(f"NaN detected in {node_type} input")
                if torch.isinf(x).any():
                    raise ValueError(f"Inf detected in {node_type} input")
        
        # 1. Encode: Graph-native spatial-temporal encoding
        # Combined subject+run embedding (personalisation, V5.19/V5.44).
        # Extracted into _get_combined_embed() to eliminate three formerly-
        # duplicated ~20-line blocks across forward/simulate/EC methods.
        combined_embed = self._get_combined_embed(data)

        encoded_data = self.encoder(data, subject_embed=combined_embed)
        
        # 2. Decode: Reconstruct signals
        reconstructed = self.decoder(encoded_data)
        
        # 3. Predict: Future steps (optional)
        predictions = None
        if return_prediction and self.use_prediction:
            # Per-node causal prediction: use last context_length timesteps
            # to predict the next prediction_steps timesteps (NPI paradigm).
            # predict_next() is the single correct entry point — it uses only
            # the last min(context_length, T) steps (no future leakage) and
            # produces a single coherent [N, pred_steps, H] prediction.
            predictions = {}
            for node_type in self.node_types:
                if node_type in encoded_data.node_types:
                    h = encoded_data[node_type].x  # [N, T, H]
                    predictions[node_type] = self.predictor.predict_next(h)

            # System-level graph propagation.
            if predictions:
                predictions = self.prediction_propagator(predictions, data)

        # 4. Optionally return latent encoded dict for compute_loss prediction loss
        if return_encoded:
            encoded_dict = {
                nt: encoded_data[nt].x
                for nt in self.node_types
                if nt in encoded_data.node_types
            }
            return reconstructed, predictions, encoded_dict

        return reconstructed, predictions
    
    @torch.no_grad()
    def simulate_intervention(
        self,
        data: HeteroData,
        interventions: Dict[str, Tuple],
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simulate a digital twin intervention: inject a perturbation at one or
        more brain regions and observe the propagated system-level causal response.

        This is the core **digital twin** capability: ask "what-if" questions
        about the brain without running an actual experiment.  It mirrors TMS
        (Transcranial Magnetic Stimulation) experiments:
          1. Encode the baseline brain state → h [N, T, H]
          2. Perturb h at the target nodes: h_pert[idx] += delta
          3. Predict both perturbed and baseline future trajectories
          4. Propagate through the brain connectivity graph (system-level coupling)
          5. Decode to signal space
          6. Causal effect = perturbed_response − baseline_response

        The causal effect tells us: "what does stimulating region X do to the
        rest of the brain?"  This is scientifically analogous to the Green's
        function / lead-field framework (Deco et al. 2013) and the perturbational
        complexity index (PCI) used in consciousness research.

        Args:
            data: HeteroData window representing the current brain state.
            interventions: Dict mapping modality → (node_indices, delta) where:

                * ``node_indices`` (List[int]): brain region indices to perturb.
                * ``delta`` (float | Tensor[H]):

                  - **float**: perturbation in units of the node's latent std.
                    ``delta=2.0`` ≈ supraphysiological stimulation (strong TMS).
                    ``delta=-1.0`` ≈ inhibitory TMS / GABA-ergic drug.
                  - **Tensor[H]**: explicit direction vector in latent H-space.
            num_steps: Future steps to predict.  Defaults to model prediction_steps.

        Returns:
            Dict with keys:

            * ``'causal_effect'``: ``{nt: Tensor[N, steps, C]}`` — net effect
              (perturbed − baseline).  Positive = increased activity.
            * ``'baseline'``: ``{nt: Tensor[N, steps, C]}`` — prediction without
              intervention (what would happen anyway).
            * ``'perturbed'``: ``{nt: Tensor[N, steps, C]}`` — prediction with
              intervention.
            * ``'encoded_baseline'``: ``{nt: Tensor[N, T, H]}`` — encoded latents
              of the baseline state (useful for inspection and further analysis).

        Example::

            # Simulate 2σ TMS to right motor cortex (fMRI ROI index 42)
            result = model.simulate_intervention(
                data=brain_window,
                interventions={"fmri": ([42], 2.0)},
                num_steps=15,
            )
            causal = result["causal_effect"]["fmri"]  # [N_fmri, 15, 1]
            # Top-10 most-affected regions:
            top = causal.squeeze(-1).abs().max(1).values.argsort(descending=True)[:10]
        """
        self.eval()
        device = next(self.parameters()).device
        data = data.to(device)

        # Combined subject+run embedding (personalisation) — uses helper to
        # eliminate the formerly-duplicated ~15-line block.
        combined_embed = self._get_combined_embed(data, device=device)

        # 1. Encode baseline
        encoded_data = self.encoder(data, subject_embed=combined_embed)
        h_baseline: Dict[str, torch.Tensor] = {
            nt: encoded_data[nt].x.clone()
            for nt in self.node_types
            if nt in encoded_data.node_types and hasattr(encoded_data[nt], "x")
        }

        # 2. Build perturbed latents
        h_perturbed = {nt: h.clone() for nt, h in h_baseline.items()}
        for nt, (node_indices, delta) in interventions.items():
            if nt not in h_perturbed:
                logger.warning(f"simulate_intervention: '{nt}' not in encoded nodes, skipping")
                continue
            H = h_perturbed[nt].shape[-1]
            if isinstance(delta, (int, float)):
                # Scale perturbation by the std of the target nodes' latent activations.
                # This makes delta=1.0 always correspond to "1 standard deviation
                # of the node's natural activity range", regardless of model scale.
                target_h = h_perturbed[nt][node_indices]     # [k, T, H]
                std_h = target_h.std(dim=(0, 1)).clamp(min=1e-6)  # [H]
                mean_dir = target_h.mean(dim=(0, 1))         # [H]
                # Use mean direction normalized; fall back to ones if near-zero
                norm = mean_dir.norm()
                direction = (mean_dir / norm) if norm > 1e-6 else torch.ones(H, device=device) / (H ** 0.5)
                delta_vec = direction * float(delta) * std_h.mean()
            else:
                delta_vec = delta.to(device)
                if delta_vec.shape != (H,):
                    raise ValueError(f"delta must be scalar or [H={H}], got {delta_vec.shape}")
            # Perturb: add delta to all time steps of specified nodes
            h_perturbed[nt][node_indices, :, :] += delta_vec.view(1, H)

        # 3. Run predictor on both contexts
        pred_baseline: Dict[str, torch.Tensor] = {}
        pred_perturbed: Dict[str, torch.Tensor] = {}
        if self.use_prediction:
            for nt in h_baseline:
                T = h_baseline[nt].shape[1]
                if T < self._PRED_MIN_SEQ_LEN:
                    continue
                pred_baseline[nt] = self.predictor.predict_next(h_baseline[nt])
                pred_perturbed[nt] = self.predictor.predict_next(h_perturbed[nt])

            # 4. System-level graph propagation
            if pred_baseline:
                pred_baseline = self.prediction_propagator(pred_baseline, data)
            if pred_perturbed:
                pred_perturbed = self.prediction_propagator(pred_perturbed, data)

        # 5. Decode predictions to signal space
        sig_baseline = self._decode_latents_to_signal(
            pred_baseline, error_context="simulate_intervention decoder"
        )
        sig_perturbed = self._decode_latents_to_signal(
            pred_perturbed, error_context="simulate_intervention decoder"
        )

        # 6. Causal effect = perturbed − baseline
        causal_effect: Dict[str, torch.Tensor] = {}
        for nt in sig_baseline:
            if nt in sig_perturbed:
                causal_effect[nt] = sig_perturbed[nt] - sig_baseline[nt]

        return {
            'causal_effect': causal_effect,
            'baseline': sig_baseline,
            'perturbed': sig_perturbed,
            'encoded_baseline': h_baseline,
        }

    @torch.no_grad()
    def compute_effective_connectivity(
        self,
        data: HeteroData,
        modality: str = 'fmri',
        perturbation_strength: float = 1.0,
        signed: bool = True,
        normalize: bool = True,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Compute whole-brain Effective Connectivity (EC) matrix via the NPI paradigm.

        Systematically perturbs each brain region and measures the causal propagated
        response across all other regions, producing a directed N×N EC matrix.

        This implements the core algorithm from:
            Luo et al. (2025) "Mapping effective connectivity by virtually perturbing
            a surrogate brain." *Nature Methods*.

        Algorithm:
            For each source region i ∈ [0, N):
                1. Encode baseline brain state → h [N, T, H]
                2. Apply unit perturbation at region i:
                   h_pert[i] += direction_i × strength × std(h[i])
                3. Predict perturbed and baseline futures; propagate through graph
                4. Decode to signal space; compute causal effect = perturbed − baseline
                5. EC[:, i] = mean-abs causal effect across time at each region

        Compared to NPI (fMRI-only RNN), TwinBrain's EC mapping offers:
          • Graph-native encoding (preserves small-world topology)
          • Multi-modal: can compute EEG-EC, fMRI-EC, and cross-modal EEG→fMRI EC
          • Subject-specific: personalized via subject_embed

        Args:
            data: HeteroData brain state window (a single time window).
            modality: Node type to perturb and measure ('fmri' or 'eeg').
            perturbation_strength: Perturbation amplitude in units of the node's
                latent standard deviation.  1.0 = 1 σ (physiologically mild),
                2.0 = 2 σ (supraphysiological, as in strong TMS).
            signed: If True, return signed EC (positive = excitatory, negative =
                inhibitory, based on net mean causal effect direction).
                If False, return absolute EC magnitude only.
            normalize: Normalize EC matrix to [0, 1] by dividing by max absolute value.
            batch_size: Number of regions to perturb in parallel.  Higher values
                use more memory but are faster on GPU.

        Returns:
            ec_matrix: Float tensor [N, N].
                ec_matrix[j, i] = causal influence of region i on region j.
                Row j = "who influences j?"; Column i = "where does i project to?".
                Diagonal represents self-loops (auto-modulation); typically small.
        """
        self.eval()
        device = next(self.parameters()).device
        data = data.to(device)

        if modality not in data.node_types:
            raise ValueError(f"Modality '{modality}' not found in data. "
                             f"Available: {list(data.node_types)}")

        N = data[modality].x.shape[0]
        T = data[modality].x.shape[1]
        if T < self._PRED_MIN_SEQ_LEN:
            raise ValueError(
                f"Sequence too short for EC computation: T={T} < {self._PRED_MIN_SEQ_LEN}. "
                f"Use a longer time window."
            )

        # Combined subject+run embedding (shared across all N perturbations) —
        # uses helper to eliminate the formerly-duplicated ~15-line block.
        combined_embed = self._get_combined_embed(data, device=device)

        # Encode baseline once
        encoded_data = self.encoder(data, subject_embed=combined_embed)
        h_baseline: Dict[str, torch.Tensor] = {
            nt: encoded_data[nt].x.clone()
            for nt in self.node_types
            if nt in encoded_data.node_types and hasattr(encoded_data[nt], "x")
        }

        # Predict baseline future
        pred_baseline: Dict[str, torch.Tensor] = {}
        if self.use_prediction:
            for nt, h in h_baseline.items():
                if h.shape[1] >= self._PRED_MIN_SEQ_LEN:
                    pred_baseline[nt] = self.predictor.predict_next(h)
            if pred_baseline:
                pred_baseline = self.prediction_propagator(pred_baseline, data)

        sig_baseline = self._decode_latents_to_signal(pred_baseline)

        # Accumulate EC column by column (each column = one perturbed region)
        ec_matrix = torch.zeros(N, N, device=device)

        # Compute per-node perturbation direction: unit vector along mean latent
        h_mod = h_baseline[modality]                   # [N, T, H]
        node_stds = h_mod.std(dim=1).clamp(min=1e-6)        # [N, H]
        node_means = h_mod.mean(dim=1)                   # [N, H]
        norms = node_means.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [N, 1]
        node_directions = node_means / norms             # [N, H]  unit direction

        for i in range(N):
            # Perturb region i: add delta in the node's principal direction
            h_pert = {nt: h.clone() for nt, h in h_baseline.items()}
            delta_vec = node_directions[i] * perturbation_strength * node_stds[i].mean()
            h_pert[modality][i, :, :] += delta_vec.view(1, 1, -1)  # [1, 1, H] → broadcast over [T, H]

            # Predict perturbed future
            pred_pert: Dict[str, torch.Tensor] = {}
            if self.use_prediction:
                for nt, h in h_pert.items():
                    if h.shape[1] >= self._PRED_MIN_SEQ_LEN:
                        pred_pert[nt] = self.predictor.predict_next(h)
                if pred_pert:
                    pred_pert = self.prediction_propagator(pred_pert, data)

            sig_pert = self._decode_latents_to_signal(pred_pert)

            # Causal effect at modality: perturbed - baseline
            if modality in sig_pert and modality in sig_baseline:
                effect = sig_pert[modality] - sig_baseline[modality]  # [N, steps, C]
                # Mean causal effect over time and channels → scalar per region [N]
                ec_column = effect.mean(dim=(1, 2))  # [N]: averaged over steps and C
                if signed:
                    ec_matrix[:, i] = ec_column
                else:
                    ec_matrix[:, i] = ec_column.abs()
            elif modality in sig_pert:
                # Baseline was empty (e.g. use_prediction=False); use raw encoded diff
                h_pert_mod = h_pert[modality].mean(dim=(1, 2))   # [N]
                h_base_mod = h_baseline[modality].mean(dim=(1, 2))
                ec_column = h_pert_mod - h_base_mod
                ec_matrix[:, i] = ec_column.abs()

        if normalize:
            max_val = ec_matrix.abs().max()
            if max_val > 1e-8:
                ec_matrix = ec_matrix / max_val

        return ec_matrix

    def compute_loss(
        self,
        data: HeteroData,
        reconstructed: Dict[str, torch.Tensor],
        predictions: Optional[Dict[str, torch.Tensor]] = None,
        encoded: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            data: Original data
            reconstructed: Reconstructed signals
            predictions: Unused (kept for API compatibility)
            encoded: Latent representations {node_type: h[N, T, H]}.
                When provided together with use_prediction=True, a
                self-supervised prediction loss is computed entirely in
                latent space (first 2/3 → predict last 1/3), giving the
                predictor a real training signal for the first time.
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Reconstruction loss per modality
        # Gated by use_reconstruction_loss (default True).  When False the
        # encoder and decoder are still trained through the signal-space
        # prediction path (pred_sig_loss → decoder → predictor), so no
        # parameters become orphaned.
        if self.use_reconstruction_loss:
            for node_type in self.node_types:
                if node_type not in data.node_types or node_type not in reconstructed:
                    continue
                # Detach target: raw signal is fixed supervision (like a label).
                # When EEG enhancement is active, data[node_type].x is the EEG
                # handler output (requires_grad=True).  Detaching prevents an
                # unintended gradient path through the handler via the target.
                target = data[node_type].x.detach()  # [N, T, C]
                recon = reconstructed[node_type]  # [N, T', C_out]
                
                # Align temporal dimensions: the decoder may produce T' ≠ T when
                # temporal_upsample is set, or (defensively) if any upstream change
                # shifts T.  Truncate to the shorter of the two.
                T_min = min(target.shape[1], recon.shape[1])
                if target.shape[1] != recon.shape[1]:
                    logger.warning(
                        f"Decoder output T={recon.shape[1]} ≠ target T={target.shape[1]} "
                        f"for {node_type}; truncating to T={T_min}."
                    )
                    recon = recon[:, :T_min, :]
                    target = target[:, :T_min, :]
                
                # Guard against N-mismatch: this signals the cross-modal encoder bug
                # (propagate() called without size=(N_src, N_dst)).  Raise clearly
                # rather than letting broadcasting silently corrupt gradients.
                if recon.shape[0] != target.shape[0]:
                    raise RuntimeError(
                        f"Node count mismatch for '{node_type}': "
                        f"recon has {recon.shape[0]} nodes but target has {target.shape[0]}. "
                        f"This usually means propagate() was called without size=(N_src, N_dst) "
                        f"for a cross-modal edge, causing N_src to bleed into N_dst."
                    )
                
                # Choose loss function
                if self.loss_type == 'huber':
                    recon_loss = F.huber_loss(recon, target, delta=1.0)
                elif self.loss_type == 'smooth_l1':
                    recon_loss = F.smooth_l1_loss(recon, target)
                else:
                    recon_loss = F.mse_loss(recon, target)
                
                losses[f'recon_{node_type}'] = recon_loss

                # ── Spectral reconstruction loss ─────────────────────────────
                # Frequency-domain complement to the time-domain reconstruction
                # loss.  For EEG especially, matching the magnitude spectrum
                # (power in alpha/beta/theta bands) produces better pred R².
                # Requires T_min ≥ 4 for a meaningful FFT.
                if self.use_spectral_loss and T_min >= 4:
                    losses[f'spectral_{node_type}'] = self._spectral_loss(
                        recon[:, :T_min, :], target[:, :T_min, :]
                    )

        # ── Cross-modal latent alignment loss (V5.43) ────────────────────────
        # Cosine similarity between mean EEG and fMRI global latent
        # representations.  Encourages the shared H-dimensional space to
        # reflect neurovascular coupling: EEG and fMRI are two views of the
        # same underlying neural activity and should be close in latent space.
        #
        # Scientific basis: Logothetis et al. 2001 (Nature), CMC Tian 2019,
        #   Thomas et al. 2022 "Self-supervised learning of brain dynamics".
        #
        # Implementation: mean-pool encoded[nt] over nodes and time → [H];
        # loss = 1 − cosine_similarity(z_eeg, z_fmri).
        # Gradients flow through BOTH modalities' encoder paths, creating
        # an explicit cross-modal alignment signal that the graph edge alone
        # cannot provide (edges are topological, not embedding-space aware).
        if self.use_cross_modal_align and encoded is not None:
            # Node types are always lowercase throughout the codebase (see
            # GraphNativeBrainMapper and map_*_to_graph helpers).  Use the
            # canonical lowercase names directly.
            _eeg_key  = 'eeg'  if 'eeg'  in encoded else None
            _fmri_key = 'fmri' if 'fmri' in encoded else None
            if _eeg_key is not None and _fmri_key is not None:
                losses['cross_modal_align'] = self._cross_modal_align_loss(
                    encoded[_eeg_key], encoded[_fmri_key]
                )

        # ── 潜空间自监督预测预标记任务（系统级）──────────────────────────────
        #
        # 设计说明（V5.42 因果注意力修复后）：
        #   编码器使用对称填充 Conv1d（±1 步边界泄漏）和因果时序注意力
        #   (TemporalAttention: is_causal=True，V5.42 修复)。
        #   is_causal=True 确保 h[:, t, :] 仅包含 signal[0..t] 的信息，
        #   训练时不再有全局未来信息泄漏。Conv1d 的 ±1 步边界泄漏仅影响
        #   序列首尾各 1 步（约 T 的 0.7%），可接受的工程近似。
        #   • 训练：编码器因果 → 预测器收到无污染的上下文
        #   • 验证：validate() 重新编码仅 T_ctx 步 → 消除 Conv1d 边界泄漏
        #   训练/验证的监督目标完全一致，pred_r2 可信。
        #
        # 流程：
        #   1. 切分 h → context（前 2/3）+ future_target（后 1/3）
        #   2. predict_next(context)：最后 context_length 步 → 下一 pred_steps 步
        #   3. GraphPredictionPropagator：系统级传播（EEG→fMRI 耦合动态）
        #   4. 计算传播后预测 vs. actual future latent 的损失
        if encoded is not None and self.use_prediction:
            pred_means: Dict[str, torch.Tensor] = {}
            future_targets: Dict[str, torch.Tensor] = {}
            # Track T_ctx per modality so the signal-space loss block below can
            # slice the correct future window from the raw signal.
            T_ctx_dict: Dict[str, int] = {}

            for node_type in self.node_types:
                if node_type not in encoded:
                    continue
                h = encoded[node_type]  # [N, T, H]
                T = h.shape[1]
                if T < self._PRED_MIN_SEQ_LEN:
                    continue
                T_ctx = int(T * self._PRED_CONTEXT_RATIO)
                context = h[:, :T_ctx, :]               # [N, T_ctx, H]
                # Stop-gradient on the prediction target: future_target is
                # supervision (analogous to a label in supervised learning),
                # so gradients should flow only through the *prediction*
                # (pred_mean / predictor / context) and NOT back through
                # future_target into the encoder.
                # Benefits:
                #   1. Semantically correct: the target should be treated as
                #      a fixed reference, not as a model output to be optimised.
                #   2. Reduces peak GPU memory during backward() by removing
                #      the gradient path through h[:, T_ctx:, :] → encoder.
                future_target = h[:, T_ctx:, :].detach() # [N, T_fut, H]

                # Causal prediction: last context_length steps → next prediction_steps.
                # This is the NPI paradigm: "N past steps → K future steps."
                #
                # Efficiency: generate only as many steps as can be supervised.
                # With windowed sampling, T_fut = T - T_ctx = T × (1/3).
                # Examples (default config):
                #   fMRI (T=50):  T_ctx=33, T_fut=17 → effective_steps=min(50,17)=17
                #   EEG  (T=500): T_ctx=333,T_fut=167→ effective_steps=min(50,167)=50
                # Generating prediction_steps=50 for fMRI then truncating to 17
                # would waste 66% of predictor computation and activation memory.
                T_fut = T - T_ctx
                effective_steps = min(self.prediction_steps, T_fut)
                pred = self.predictor.predict_next(context, num_steps=effective_steps)
                pred_means[node_type] = pred
                future_targets[node_type] = future_target
                T_ctx_dict[node_type] = T_ctx

            # System-level graph propagation of predictions.
            if pred_means:
                pred_means = self.prediction_propagator(pred_means, data)

            # ── Latent-space prediction loss ─────────────────────────────────
            # Propagated latent prediction vs. actual future latent.
            for node_type, pred_mean in pred_means.items():
                future_target = future_targets[node_type]
                # Guard N-mismatch (same defensive contract as recon_loss):
                # propagator is expected to preserve N, but raise explicitly if
                # not rather than silently broadcasting or producing wrong gradients.
                if pred_mean.shape[0] != future_target.shape[0]:
                    logger.warning(
                        f"pred_loss skipped for '{node_type}': "
                        f"pred has {pred_mean.shape[0]} nodes but target has "
                        f"{future_target.shape[0]}.  This may indicate a bug in "
                        f"GraphPredictionPropagator."
                    )
                    continue
                aligned_steps = min(pred_mean.shape[1], future_target.shape[1])
                if aligned_steps > 0:
                    pm_slice = pred_mean[:, :aligned_steps, :]      # [N, S, H]
                    ft_slice = future_target[:, :aligned_steps, :]  # [N, S, H]
                    if self.pred_step_weight_gamma > 0.0 and aligned_steps > 1:
                        # Exponentially increasing weight for later prediction steps.
                        # Step t gets weight exp(gamma * t / (aligned_steps - 1)),
                        # normalised so the mean weight = 1 (preserving loss scale).
                        # This focuses gradient signal on far-future steps that are
                        # harder to predict (curriculum difficulty weighting).
                        # Reference: Bengio et al. 2015 "Scheduled Sampling".
                        #
                        # Implementation: fully vectorised — NO Python loop.
                        # The previous implementation used a for-loop over aligned_steps
                        # (e.g. 17 for fMRI, 50 for EEG), issuing individual
                        # F.huber_loss CUDA kernel calls per step.
                        # Per epoch: (17_fmri_steps + 50_eeg_steps) × 80 grad-accum
                        # = 5360 F.huber_loss kernel launches — significant overhead.
                        # The vectorised approach uses 3-4 CUDA ops per modality per
                        # sample regardless of aligned_steps.
                        step_w = torch.exp(
                            torch.linspace(
                                0.0, self.pred_step_weight_gamma, aligned_steps,
                                device=pred_mean.device, dtype=torch.float32,
                            )
                        )
                        step_w = (step_w / step_w.mean()).detach()  # [aligned_steps]
                        # Compute per-element loss: [N, aligned_steps, H]
                        diff = (pm_slice - ft_slice)
                        if self.loss_type == 'huber':
                            delta = 1.0
                            abs_diff = diff.abs()
                            per_element = torch.where(
                                abs_diff < delta,
                                0.5 * diff.pow(2),
                                delta * (abs_diff - 0.5 * delta),
                            )
                        else:
                            per_element = diff.pow(2)
                        # Mean over N and H → per-step scalar [aligned_steps]
                        per_step_loss = per_element.mean(dim=(0, 2))
                        pred_loss = (per_step_loss * step_w).mean()
                    elif self.loss_type == 'huber':
                        pred_loss = F.huber_loss(pm_slice, ft_slice, delta=1.0)
                    else:
                        pred_loss = F.mse_loss(pm_slice, ft_slice)
                    losses[f'pred_{node_type}'] = pred_loss

            # ── Signal-space prediction loss ─────────────────────────────────
            # Decode the predicted latents to raw signal space and compare with
            # the actual future raw signal.
            #
            # Rationale: the latent-space loss above teaches the predictor to
            # predict abstract latent dynamics.  However, the pred_r2 metric in
            # validate() is measured in signal space (decoder(pred_latent) vs
            # future raw signal).  If the decoder was never trained to decode
            # *predicted* latents (only *encoded* latents), the round-trip quality
            # can be poor even when the latent prediction itself is reasonable.
            # Adding this end-to-end signal-space loss directly optimises the
            # metric that is reported to the user.
            #
            # The gradient flows:  pred_sig_loss → decoder → prediction_propagator
            #                       → predictor.predict_next
            # Training both the decoder (for predicted-latent decoding) and the
            # predictor (for signal-quality alignment) in a single backward pass.
            if pred_means and T_ctx_dict:
                # The decoder (GraphNativeDecoder) only needs encoded_data[nt].x
                # and runs 1-D temporal convolutions; no edge_index/edge_attr needed.
                _ctx = (
                    f"pred_sig decoder failed for "
                    f"node_types={list(pred_means.keys())} — "
                    f"pred shapes={[tuple(v.shape) for v in pred_means.values()]}"
                )
                _pred_sigs = self._decode_latents_to_signal(pred_means, error_context=_ctx)
                for _nt, _pred_sig in _pred_sigs.items():
                    _T_ctx = T_ctx_dict.get(_nt)
                    if _T_ctx is None or _nt not in data.node_types:
                        continue
                    # Detach future signal: it is fixed supervision, not a
                    # differentiable target.  When EEG enhancement is active,
                    # data[_nt].x has requires_grad=True; detaching prevents an
                    # unintended gradient path through the EEG handler.
                    _future_sig = data[_nt].x[
                        :, _T_ctx:_T_ctx + _pred_sig.shape[1], :
                    ].detach()
                    _n = min(_pred_sig.shape[1], _future_sig.shape[1])
                    if _n < 1:
                        logger.debug(
                            f"pred_sig_{_nt}: skipped (aligned_steps=0; "
                            f"pred_sig.shape={tuple(_pred_sig.shape)}, "
                            f"future_sig.shape={tuple(_future_sig.shape)})"
                        )
                        continue
                    if _pred_sig.shape[0] != _future_sig.shape[0]:
                        logger.debug(
                            f"pred_sig_{_nt}: skipped (N mismatch; "
                            f"pred N={_pred_sig.shape[0]}, future N={_future_sig.shape[0]})"
                        )
                        continue
                    if self.loss_type == 'huber':
                        _sig_loss = F.huber_loss(
                            _pred_sig[:, :_n, :],
                            _future_sig[:, :_n, :],
                            delta=1.0,
                        )
                    else:
                        _sig_loss = F.mse_loss(
                            _pred_sig[:, :_n, :],
                            _future_sig[:, :_n, :],
                        )
                    # Pearson correlation component: optimises temporal pattern
                    # matching (the shape of the predicted trajectory).  R² ≈ r²
                    # for linear predictions, so this loss directly targets what
                    # validate() reports as pred_r2.  Weight 0.5 (up from 0.2)
                    # makes the pattern-matching signal strong enough to overcome
                    # the amplitude-only Huber/MSE term that can be minimised by
                    # predicting a flat (mean-like) signal, which yields negative
                    # pred_r2 even at low MSE.
                    if _n >= 2:
                        _corr = self._pearson_loss(
                            _pred_sig[:, :_n, :], _future_sig[:, :_n, :]
                        )
                        _sig_loss = _sig_loss + 0.5 * _corr
                    losses[f'pred_sig_{_nt}'] = _sig_loss

        # ── InfoNCE contrastive prediction loss (V5.47) ──────────────────────────
        # Complements the MSE/Huber regression losses by forcing the predictor to
        # produce *discriminative* representations.  Pure regression can be minimised
        # by predicting the unconditional mean (flat output), which yields low MSE
        # but negative pred_r2.  InfoNCE requires identifying the correct future
        # timestep from all (node × step) negative examples, which is only possible
        # if the predicted representation carries unique information about that window.
        # Reference: Oord et al. (2018) CPC; Baevski et al. (2020) wav2vec 2.0.
        if self.use_info_nce and encoded is not None and self.use_prediction and pred_means:
            for _nce_nt, _nce_pred in pred_means.items():
                if _nce_nt not in future_targets:
                    continue
                _nce_target = future_targets[_nce_nt]
                _nce_aligned = min(_nce_pred.shape[1], _nce_target.shape[1])
                # Need at least 2 (node×step) pairs for contrastive loss.
                if _nce_aligned < 1 or _nce_pred.shape[0] * _nce_aligned < 2:
                    continue
                if _nce_pred.shape[0] != _nce_target.shape[0]:
                    continue
                losses[f'pred_nce_{_nce_nt}'] = self._info_nce_loss(
                    _nce_pred[:, :_nce_aligned, :],
                    _nce_target[:, :_nce_aligned, :],
                    temperature=self.info_nce_temperature,
                )

        return losses


class GraphNativeTrainer:
    """
    Complete training system for graph-native brain model.
    
    Integrates:
    - Graph-native model
    - Adaptive loss balancing
    - EEG channel enhancement
    - Advanced prediction
    
    This is a STANDALONE system, not dependent on old trainer.
    """
    
    def __init__(
        self,
        model: GraphNativeBrainModel,
        node_types: List[str],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_adaptive_loss: bool = True,
        use_eeg_enhancement: bool = True,
        use_amp: bool = True,
        use_gradient_checkpointing: bool = False,
        use_scheduler: bool = True,
        scheduler_type: str = 'cosine',
        use_torch_compile: bool = True,
        compile_mode: str = 'reduce-overhead',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        optimization_config: Optional[Dict] = None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        augmentation_config: Optional[Dict] = None,
        cuda_clear_interval: int = 50,
    ):
        """
        Initialize trainer.
        
        Args:
            model: GraphNativeBrainModel
            node_types: Node types in data
            learning_rate: Learning rate
            weight_decay: Weight decay
            use_adaptive_loss: Use adaptive loss balancing
            use_eeg_enhancement: Use EEG channel enhancement
            use_amp: Use automatic mixed precision (AMP) for 2-3x speedup
            use_gradient_checkpointing: Use gradient checkpointing to save memory
            use_scheduler: Use learning rate scheduling (10-20% faster convergence)
            scheduler_type: Type of scheduler ('cosine', 'onecycle', 'plateau')
            use_torch_compile: Use torch.compile() for 20-40% speedup (PyTorch 2.0+)
            compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            device: Device to train on
            optimization_config: Optional dict from config['v5_optimization'].
                Contains sub-dicts 'adaptive_loss' and 'eeg_enhancement' with
                fine-grained hyperparameters.  Defaults used when None.
            max_grad_norm: Max gradient norm for clipping (config['training']['max_grad_norm']).
                Hardcoding this to 1.0 previously caused the config value to be silently ignored.
            gradient_accumulation_steps: Accumulate gradients over this many steps before
                calling optimizer.step().  Default 1 = standard single-step update (backward
                compatible).  Set to 4 for an effective batch size of 4 with batch_size=1
                data, stabilising gradient estimates on small datasets without extra memory.
                The loss is scaled by 1/gradient_accumulation_steps before backward() to
                keep gradient magnitudes consistent regardless of the accumulation count.
            augmentation_config: Optional dict from config['training']['augmentation'].
                Supported keys:
                  enabled (bool): master switch (default False — backward compatible).
                  noise_std (float): std of Gaussian noise added to node features
                      (default 0.01; relative to z-scored signals so 1% amplitude).
                  scale_range ([min, max]): random amplitude scaling per sample
                      (default [0.9, 1.1]; None to disable).
            cuda_clear_interval: Call gc.collect() + torch.cuda.empty_cache() every
                this many training steps within an epoch (in addition to the per-epoch
                clears that already happen at epoch start/end).  This prevents CUDA
                memory fragmentation from accumulating when an epoch has many steps
                (e.g. 8 subjects × 10 windows = 80 steps/epoch).  Default 50.
                Set to 0 to disable intra-epoch clearing (original behaviour).
        """
        self._optimization_config = optimization_config or {}
        self.model = model.to(device)
        self.device = device
        # Cache device type string (e.g. 'cuda', 'cpu') once so all methods can
        # use it without re-deriving it from self.device each time.
        self._device_type = getattr(
            torch.device(device) if isinstance(device, str) else device,
            'type',
            str(device).split(':')[0],
        )
        self.node_types = node_types
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        # Intra-epoch CUDA fragmentation control.
        # With many subjects + windowed_sampling enabled, a single epoch can have
        # 40-80+ steps.  Each step allocates and frees [chunk_size×E×H] message
        # tensors; without periodic clearing the CUDA free-list fragments and a
        # large contiguous allocation later in the epoch fails (OOM) even though
        # reserved >> allocated.  Clearing every cuda_clear_interval steps keeps
        # fragmentation bounded throughout the epoch.
        self._cuda_clear_interval = max(0, int(cuda_clear_interval))

        # Temporal augmentation config (applied only during training, not validation)
        _aug = augmentation_config or {}
        self._aug_enabled    = bool(_aug.get('enabled', False))
        self._aug_noise_std  = float(_aug.get('noise_std', 0.01))
        _sr = _aug.get('scale_range', [0.9, 1.1])
        self._aug_scale_min  = float(_sr[0]) if _sr else 1.0
        self._aug_scale_max  = float(_sr[1]) if _sr else 1.0
        # Time masking: randomly zero out a contiguous time segment per node type.
        # Inspired by SpecAugment (Park et al. 2019).  Ratio is the maximum fraction
        # of T that can be masked (uniform draw in [0, ratio] each step).
        # 0.0 = disabled (backward-compatible default).
        self._aug_time_mask_ratio = float(_aug.get('time_mask_max_ratio', 0.0))
        if self._aug_enabled:
            logger.info(
                f"时序数据增强已启用: "
                f"noise_std={self._aug_noise_std}, "
                f"scale_range=[{self._aug_scale_min}, {self._aug_scale_max}], "
                f"time_mask_max_ratio={self._aug_time_mask_ratio}"
            )
        if self._cuda_clear_interval > 0:
            logger.info(
                f"CUDA 碎片防护已启用: "
                f"每 {self._cuda_clear_interval} 步清理一次显存空闲列表 "
                f"（支持 4-8 被试多窗口训练而不 OOM）"
            )
        
        # Verify CUDA availability
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"CUDA requested but not available. Falling back to CPU.")
            self.device = 'cpu'
            self.model = self.model.to('cpu')
        
        # torch.compile() for PyTorch 2.0+ (20-40% speedup)
        # NOTE: torch.compile() uses the 'inductor' backend which requires Triton.
        # Triton is unavailable on Windows and some other platforms. The compilation
        # is *lazy* – torch.compile() itself doesn't raise; the crash happens at the
        # first forward pass. We therefore probe Triton availability upfront and skip
        # compilation gracefully rather than relying on the try/except below.
        if use_torch_compile and hasattr(torch, 'compile'):
            _triton_ok = False
            try:
                import triton  # noqa: F401
                _triton_ok = True
            except ImportError:
                pass

            if not _triton_ok:
                logger.warning(
                    "torch.compile() skipped: Triton is not installed or not supported "
                    "on this platform (e.g. Windows). Running in eager mode."
                )
            else:
                logger.info(f"Enabling torch.compile() with mode={compile_mode}")
                try:
                    self.model = torch.compile(
                        self.model,
                        mode=compile_mode,
                        fullgraph=False  # Allow graph breaks for flexibility
                    )
                    logger.info("torch.compile() enabled successfully")
                except Exception as e:
                    logger.warning(f"torch.compile() failed, continuing without it: {e}")
        elif use_torch_compile:
            logger.warning("torch.compile() requested but not available (requires PyTorch >= 2.0)")
        
        # Mixed precision training
        self.use_amp = use_amp and device != 'cpu' and AMP_AVAILABLE
        if self.use_amp:
            # Use new API if available (torch.amp.GradScaler), otherwise old API (torch.cuda.amp.GradScaler)
            if USE_NEW_AMP_API:
                # New API expects device type as string (e.g., 'cuda', not 'cuda:0')
                # Extract device type: handle both string and torch.device inputs
                device_type = getattr(self.device, 'type', str(self.device).split(':')[0])
                self.device_type = device_type  # Store for use with autocast()
                self.scaler = GradScaler(device=device_type)
            else:
                # Old API doesn't take device parameter
                self.device_type = 'cuda'  # Old API only supports CUDA
                self.scaler = GradScaler()
            logger.info("Mixed precision training (AMP) enabled")
        elif use_amp and not AMP_AVAILABLE:
            logger.warning("AMP requested but not available. Training without mixed precision.")
        
        # Gradient checkpointing
        # NOTE: checkpointing is applied inside SpatialTemporalGraphConv's
        # temporal loop (via use_gradient_checkpointing on the model).
        # The trainer just logs its status here.
        if use_gradient_checkpointing:
            logger.info("Gradient checkpointing enabled (applied per ST-GCN timestep)")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.use_scheduler = use_scheduler
        self.scheduler = None
        if use_scheduler:
            if scheduler_type == 'cosine':
                # Linear warm-up for the first `warmup_epochs` epochs, then
                # CosineAnnealingWarmRestarts.  Without warm-up, the full LR
                # is applied from epoch 1, which often causes large gradient
                # steps on a freshly initialised model — particularly harmful
                # when training on small neuroimaging datasets (N < 100 samples).
                #
                # SequentialLR chains two schedulers: LinearLR ramps from
                # start_factor × lr up to lr over warmup_epochs steps, then
                # hands off to CosineAnnealingWarmRestarts.
                warmup_epochs = self._optimization_config.get(
                    'warmup_epochs',
                    3,  # safe minimum; v5_optimization.warmup_epochs in default.yaml is 5
                )
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,   # start at 10% of target LR
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                # cosine_T0: first restart after this many post-warmup epochs.
                # Read from v5_optimization.cosine_T0 (default 20 for backward compat).
                # Larger T_0 delays the LR spike past the adaptive loss balancer's
                # warmup stabilisation period, preventing pred_r2 oscillations.
                # Example: warmup=5, T_0=20 → first restart at epoch 25.
                #          warmup=5, T_0=50 → first restart at epoch 55.
                cosine_T0 = self._optimization_config.get('cosine_T0', 20)
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=cosine_T0,
                    T_mult=2,  # double period after each restart
                    eta_min=learning_rate * 0.01,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_epochs],
                )
                logger.info(
                    f"LR scheduler: Linear warmup ({warmup_epochs} epochs) "
                    f"→ CosineAnnealingWarmRestarts(T_0={cosine_T0}, T_mult=2)"
                )
            elif scheduler_type == 'onecycle':
                # OneCycle (will need total_steps, set in train_epoch)
                self.scheduler_type = 'onecycle'
                self.scheduler = None  # Will be created when we know total steps
                logger.info(f"Learning rate scheduler: OneCycle (will be initialized with total steps)")
            elif scheduler_type == 'plateau':
                # Reduce on plateau
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                )
                logger.info(f"Learning rate scheduler enabled: ReduceLROnPlateau")
            else:
                logger.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
                self.use_scheduler = False
        
        # Adaptive loss balancing
        self.use_adaptive_loss = use_adaptive_loss
        if use_adaptive_loss:
            # Create task names from node types.
            # pred_sig_{nt} is the signal-space prediction loss added in V5.39
            # (compute_loss decodes predicted latents and compares with future raw
            # signal).  It must be registered here so the balancer weights it
            # appropriately; without registration the balancer ignores it and
            # the total_loss computation would not include it when use_adaptive_loss=True.
            # spectral_{nt} is the frequency-domain reconstruction loss added in V5.40
            # (FFT magnitude comparison).  Registered when use_spectral_loss=True.
            task_names = []
            for node_type in node_types:
                # recon tasks only if reconstruction loss is enabled
                if getattr(model, 'use_reconstruction_loss', True):
                    task_names.append(f'recon_{node_type}')
                    if getattr(model, 'use_spectral_loss', False):
                        task_names.append(f'spectral_{node_type}')
                if model.use_prediction:
                    task_names.append(f'pred_{node_type}')
                    task_names.append(f'pred_sig_{node_type}')
                # getattr fallback: supports loading pre-V5.47 checkpoints where
                # model.use_info_nce may not yet exist as an instance attribute.
                if getattr(model, 'use_info_nce', False):
                    task_names.append(f'pred_nce_{node_type}')
            # cross_modal_align is added in V5.43: aligns mean EEG and fMRI
            # latent representations via cosine similarity.  Registered only
            # when the model flag is on AND both modalities are present.
            if getattr(model, 'use_cross_modal_align', False) and len(node_types) >= 2:
                task_names.append('cross_modal_align')
            
            al_cfg = self._optimization_config.get('adaptive_loss', {})
            self.loss_balancer = AdaptiveLossBalancer(
                task_names=task_names,
                modality_names=node_types,
                alpha=al_cfg.get('alpha', 1.5),
                update_frequency=al_cfg.get('update_frequency', 10),
                learning_rate=al_cfg.get('learning_rate', 0.025),
                warmup_epochs=al_cfg.get('warmup_epochs', 5),
                modality_energy_ratios=al_cfg.get('modality_energy_ratios', {'eeg': 1.0, 'fmri': 1.0}),
                task_priorities=al_cfg.get('task_priorities'),
                pred_weight_floor=al_cfg.get('pred_weight_floor', 0.5),
            )
        
        # EEG enhancement
        # DESIGN NOTE: EEG graph data has shape [N_eeg, T, 1] — each graph node is
        # one electrode, with 1-dimensional feature (signal amplitude).
        # EnhancedEEGHandler was designed for [batch, time, channels] where 'channels'
        # are individual EEG electrodes.  The correct mapping is:
        #   graph format: [N_eeg, T, 1]  →  handler format: [1, T, N_eeg]
        # N_eeg is only known at data-loading time (varies per dataset/subject), so
        # we use lazy initialisation: the handler is created on the first training step.
        self.use_eeg_enhancement = use_eeg_enhancement and 'eeg' in node_types
        self.eeg_handler = None          # created lazily in train_step()
        self._eeg_n_channels: Optional[int] = None   # recorded in _ensure_eeg_handler; used to detect channel-count mismatch across subjects
        self._eeg_handler_cfg = self._optimization_config.get('eeg_enhancement', {})
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def _ensure_eeg_handler(self, n_eeg_channels: int) -> None:
        """Lazily initialise EnhancedEEGHandler once N_eeg is known from real data.

        Called from train_step() on the first forward pass so that the handler
        is built with the correct num_channels = N_eeg (number of electrodes),
        not the graph feature dimension (always 1).
        """
        if self.eeg_handler is not None:
            return  # already initialised
        cfg = self._eeg_handler_cfg
        try:
            self.eeg_handler = EnhancedEEGHandler(
                num_channels=n_eeg_channels,
                enable_monitoring=cfg.get('enable_monitoring', True),
                enable_dropout=cfg.get('enable_dropout', True),
                enable_attention=cfg.get('enable_attention', True),
                enable_regularization=cfg.get('enable_regularization', True),
                dropout_rate=cfg.get('dropout_rate', 0.1),
                attention_hidden_dim=cfg.get('attention_hidden_dim', 64),
                entropy_weight=cfg.get('entropy_weight', 0.01),
                diversity_weight=cfg.get('diversity_weight', 0.01),
                activity_weight=cfg.get('activity_weight', 0.01),
            ).to(self.device)
            self._eeg_n_channels = n_eeg_channels
            logger.info(f"EEG handler initialised for {n_eeg_channels} channels.")
        except Exception as e:
            logger.warning(f"Failed to initialise EEG handler: {e}. Disabling EEG enhancement.")
            self.use_eeg_enhancement = False
    
    @staticmethod
    def _graph_to_handler_format(eeg_x: torch.Tensor) -> torch.Tensor:
        """Reshape EEG graph tensor for EnhancedEEGHandler.

        EnhancedEEGHandler expects ``[batch, time, channels]`` where *channels*
        are individual EEG electrodes.  Graph node features are stored as
        ``[N_eeg, T, 1]``.  This helper converts between the two formats.

        ``[N_eeg, T, 1]`` → ``[1, T, N_eeg]``
        """
        return eeg_x.squeeze(-1).permute(1, 0).unsqueeze(0)

    @staticmethod
    def _handler_to_graph_format(eeg_x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_graph_to_handler_format`.

        ``[1, T, N_eeg]`` → ``[N_eeg, T, 1]``
        """
        return eeg_x.squeeze(0).permute(1, 0).unsqueeze(-1)

    def train_step(
        self,
        data: HeteroData,
        do_zero_grad: bool = True,
        do_optimizer_step: bool = True,
        loss_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Single training step with optional mixed precision.
        
        Args:
            data: Input HeteroData
            do_zero_grad: Whether to zero gradients before this step.
                Set False when accumulating gradients across multiple steps.
            do_optimizer_step: Whether to call optimizer.step() after backward.
                Set False for all but the last step in gradient accumulation.
            loss_scale: Multiply total_loss by this factor before backward().
                Use 1/gradient_accumulation_steps to normalise gradient magnitude
                when accumulating, keeping effective gradient scale constant
                regardless of accumulation count.
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        if do_zero_grad:
            self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)

        # ── 时序数据增强（仅训练模式）────────────────────────────────────
        # 在每个训练步随机向节点特征添加小量高斯噪声和/或随机幅度缩放，
        # 提升模型对信号噪声和个体差异的鲁棒性（类似神经影像信号增强文献中
        # 的 jitter 和 scaling 方法）。验证时不增强，保证评估一致性。
        # 实现为纯 in-place-free 加法/乘法，不影响 autograd 图。
        #
        # IMPORTANT: Save pre-augmentation tensors and restore in finally.
        # If HeteroData.to(device) returns the SAME object (in-place move, which
        # is the default PyG behaviour when data is already on the target device
        # or in some PyG versions), `data` is the SAME object as data_list[i].
        # Without restoration, noise+scale augmentation permanently corrupts the
        # cached graph tensors: each epoch augments the already-augmented signal
        # from the previous epoch, causing unbounded signal drift.
        # Storing originals in _orig_aug and restoring in finally gives the same
        # safety guarantee that EEG enhancement and time masking already have.
        _orig_aug: Dict[str, torch.Tensor] = {}
        if self._aug_enabled:
            for _nt in data.node_types:
                if hasattr(data[_nt], 'x') and data[_nt].x is not None:
                    # .clone() makes an independent copy of the original tensor.
                    # Without clone(), _orig_aug[_nt] would be an alias of the
                    # same storage; any in-place operation performed later on
                    # data[_nt].x (e.g. by HeteroData internals or PyG operators)
                    # would silently corrupt the saved "original".
                    _orig_aug[_nt] = data[_nt].x.clone()
                    _x = data[_nt].x
                    if self._aug_noise_std > 0:
                        _x = _x + torch.randn_like(_x) * self._aug_noise_std
                    if self._aug_scale_min != 1.0 or self._aug_scale_max != 1.0:
                        _scale = (
                            torch.empty(1, device=_x.device, dtype=_x.dtype)
                            .uniform_(self._aug_scale_min, self._aug_scale_max)
                        )
                        _x = _x * _scale
                    data[_nt].x = _x
        
        # Apply EEG enhancement if enabled.
        # Graph data shape: [N_eeg, T, 1] (N_eeg nodes, each with a 1-dim feature).
        # EnhancedEEGHandler expects [batch, time, channels] where 'channels' are
        # individual EEG electrodes.  We therefore reshape:
        #   graph format: [N_eeg, T, 1]  →  handler format: [1, T, N_eeg]
        # After processing we reshape back.
        # IMPORTANT: Save original eeg_x and restore it in a finally block.
        # eeg_x_enhanced is part of the current computation graph (requires_grad=True).
        # If we leave data['eeg'].x pointing to it, the next epoch's forward pass
        # will build a new graph ON TOP of last epoch's freed graph, triggering:
        # "Trying to backward through the graph a second time".
        # Using try-finally guarantees restoration even if an exception is raised
        # anywhere inside the forward/backward path.
        eeg_info: dict = {}  # 初始为空；仅当 EEG handler 激活时（下方 if 块）被填充
        original_eeg_x = None
        if self.use_eeg_enhancement and 'eeg' in data.node_types:
            N_eeg = data['eeg'].x.shape[0]
            # Guard: if the handler was already initialised for a different channel
            # count (e.g. this subject has 64 channels but another had 63), skip
            # enhancement rather than feeding a wrong-shape tensor.  This silently
            # fails in the original code and produces garbage gradients or shape
            # errors that are hard to diagnose.
            # original_eeg_x is only set here (when we're about to modify), keeping
            # the finally-block restoration logic clean: non-None ↔ modified.
            if self.eeg_handler is not None and self._eeg_n_channels != N_eeg:
                logger.debug(
                    f"EEG channel count mismatch: handler built for "
                    f"{self._eeg_n_channels} channels, this sample has {N_eeg}. "
                    f"Skipping EEG enhancement for this sample."
                )
                # No modification made — original_eeg_x stays None, nothing to restore
            else:
                # Lazy-initialise handler with true electrode count (N_eeg).
                # Previous approach used in_features=1 (graph feature dim), which
                # made all channel-specific processing trivially useless.
                self._ensure_eeg_handler(N_eeg)
                if self.use_eeg_enhancement and self.eeg_handler is not None:
                    original_eeg_x = data['eeg'].x  # save before modifying
                    eeg_x_t, eeg_info = self.eeg_handler(
                        self._graph_to_handler_format(original_eeg_x), training=True
                    )
                    eeg_x_enhanced = self._handler_to_graph_format(eeg_x_t)
                    data['eeg'].x = eeg_x_enhanced
        
        # _orig_tm stores original node features before time masking for restoration
        # in the finally block.  Defined here so finally can always access it.
        _orig_tm: Dict[str, torch.Tensor] = {}
        try:
            # ── Time masking augmentation (SpecAugment-style) ─────────────────
            # Randomly zero out a contiguous time segment per node type.
            # Applied inside try so the finally block always restores originals.
            # Saves originals in _orig_tm so data is not permanently modified.
            if self._aug_enabled and self._aug_time_mask_ratio > 0:
                for _nt in data.node_types:
                    if hasattr(data[_nt], 'x') and data[_nt].x is not None:
                        _x = data[_nt].x
                        _T = _x.shape[1]
                        if _T < 4:
                            continue
                        # Clamp mask length to [1, _T] so it never exceeds the
                        # sequence length (avoids _mask_start always being 0 when
                        # the random draw produces _mask_len = _T).
                        _mask_len = min(_T, max(1, int(_T * random.uniform(0, self._aug_time_mask_ratio))))
                        _mask_start = random.randint(0, max(0, _T - _mask_len))
                        _x_masked = _x.clone()
                        _x_masked[:, _mask_start:_mask_start + _mask_len, :] = 0.0
                        _orig_tm[_nt] = _x  # save original tensor reference before masking
                        data[_nt].x = _x_masked

            # ── Forward and backward pass ─────────────────────────────────────
            # Use autocast when AMP is enabled, contextlib.nullcontext otherwise.
            # This eliminates the duplicated forward+loss block that previously
            # existed for the AMP and non-AMP paths respectively.
            if self.use_amp:
                amp_ctx = (
                    autocast(device_type=self.device_type)
                    if USE_NEW_AMP_API else autocast()
                )
            else:
                amp_ctx = contextlib.nullcontext()

            with amp_ctx:
                # Forward pass.
                # When use_prediction=True, retrieve encoded latent representations
                # so compute_loss can train the predictor in latent space.
                if self.model.use_prediction:
                    reconstructed, _, encoded = self.model(
                        data, return_prediction=False, return_encoded=True
                    )
                else:
                    reconstructed, _ = self.model(data, return_prediction=False)
                    encoded = None

                # Compute losses (reconstruction + optional latent prediction)
                losses = self.model.compute_loss(data, reconstructed, encoded=encoded)

                # Adaptive loss balancing
                if self.use_adaptive_loss:
                    total_loss, weights = self.loss_balancer(losses)
                else:
                    total_loss = sum(losses.values())

                # ── EEG 防零崩塌正则化 ───────────────────────────────
                # eeg_handler 计算的熵+多样性+活动损失之前从未加入总损失
                # （eeg_info 被静默丢弃）。此处补全，确保其梯度信号生效。
                eeg_reg = eeg_info.get('regularization_loss')
                if eeg_reg is not None:
                    total_loss = total_loss + eeg_reg
                    losses['eeg_reg'] = eeg_reg

            # Backward pass — AMP uses scaler, non-AMP uses plain backward.
            if self.use_amp:
                self.scaler.scale(total_loss * loss_scale).backward()
                if do_optimizer_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                (total_loss * loss_scale).backward()
                if do_optimizer_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
            
            # Update loss balancer with detached scalar values.
            # backward() has already freed the computation graph by this point;
            # update_weights() uses .item() internally, so passing detached losses
            # makes the post-backward contract explicit and avoids any accidental
            # graph access that would raise "backward through the graph a second time".
            if self.use_adaptive_loss:
                detached_losses = {k: v.detach() for k, v in losses.items()}
                self.loss_balancer.update_weights(detached_losses)
            
            # Return loss values.
            # IMPORTANT: total_raw must be computed BEFORE total is set, because
            # loss_dict at this point contains only the individual per-task losses
            # from compute_loss() (e.g. recon_eeg, pred_fmri, eeg_reg …).  The
            # weighted aggregate 'total' is not yet in loss_dict, so the sum is
            # the true unweighted task total — not inflated by adaptive weights.
            # total_raw = unweighted sum → interpretable train_loss metric.
            # total     = weighted sum  → used for backward; grows when balancer
            #             increases task weights even if raw losses are stable.
            loss_dict = {k: v.item() for k, v in losses.items()}
            loss_dict['total_raw'] = sum(loss_dict.values())   # unweighted; used as train_loss
            loss_dict['total'] = total_loss.item()             # weighted; used for backward only
            
            return loss_dict
        
        finally:
            # Always restore original EEG data so subsequent epochs start from the
            # raw (detached) tensor rather than this step's enhanced (grad-bearing)
            # tensor — regardless of whether an exception was raised.
            if original_eeg_x is not None:
                data['eeg'].x = original_eeg_x
            # Restore time-masked node features to avoid permanently zeroing out
            # segments in the cached graph objects.
            # Restoration logic for EEG:
            # - original_eeg_x is set (EEG handler was active): restoring from
            #   original_eeg_x gives the pre-enhancement, pre-mask signal — which
            #   IS correct.  We always restore to the original cached signal, not
            #   to any intermediate enhanced version, because EEG enhancement is
            #   meant to only affect the current forward pass (not to accumulate
            #   across epochs in the cached graph).
            # - original_eeg_x is None (EEG handler inactive): _orig_tm['eeg']
            #   restores the unmasked original signal below.
            for _nt_r, _x_orig_r in _orig_tm.items():
                if _nt_r == 'eeg' and original_eeg_x is not None:
                    continue  # EEG already restored to original_eeg_x above
                data[_nt_r].x = _x_orig_r
            # Restore noise+scale augmented features to the pre-augmentation originals.
            # This must happen LAST (after EEG and time-mask restorations) because:
            # - _orig_aug[nt] holds the tensor that existed BEFORE any augmentation
            # - _orig_tm[nt] holds the tensor AFTER augmentation but BEFORE masking
            # - Restoring _orig_aug unconditionally gives the cleanest, deterministic
            #   result: every data[nt].x ends up at the original cached signal.
            #
            # Why this is necessary: if HeteroData.to(device) moves tensors in-place
            # (the default PyG behaviour when data is already on the target device),
            # `data` IS the same Python object as data_list[i].  Without this
            # restoration, the noise+scale-augmented tensor persists in the cached
            # graph and is re-augmented on every subsequent epoch, causing
            # cumulative signal drift across epochs.
            for _nt_a, _x_orig_a in _orig_aug.items():
                data[_nt_a].x = _x_orig_a
    
    def train_epoch(self, data_list: List[HeteroData], epoch: int = None, total_epochs: int = None) -> float:
        """
        Train for one epoch with learning rate scheduling and progress logging.
        
        Args:
            data_list: List of training data
            epoch: Current epoch number (for logging)
            total_epochs: Total number of epochs (for logging)
            
        Returns:
            Average loss for epoch
        """
        if len(data_list) == 0:
            raise ValueError("Cannot train on empty data_list")
        
        # ── 逐 epoch 打乱训练样本 ────────────────────────────────────────────
        # 使用 epoch 作为随机种子：每个 epoch 的顺序不同（保证 SGD 随机性），
        # 但相同 epoch 编号时顺序可复现（方便调试）。
        #
        # 为什么必须打乱：
        # 1. `train_model` 在训练开始时只打乱一次；之后每个 epoch 都使用相同顺序。
        # 2. 不打乱时，排在列表末尾的被试/任务每个 epoch 总是得到最后一次权重更新，
        #    模型会对它们产生隐式偏好（optimizer 的动量使最后几步梯度影响最大）。
        # 3. 窗口模式下，同一 run 的 11 个窗口若连续处理，局部 loss 景观过于平滑，
        #    学习曲线呈"锯齿"型而非平稳下降。
        epoch_data = data_list.copy()   # 浅拷贝：不修改原始列表，仅改变本 epoch 的遍历顺序
        random.Random(epoch or 0).shuffle(epoch_data)
        total_loss = 0.0
        num_batches = len(epoch_data)
        ga = self.gradient_accumulation_steps  # shorthand
        
        # ── 显存清理（epoch 开始时）────────────────────────────────────────
        # 在每个 epoch 的训练步骤开始前清理 CUDA 分配器的碎片化缓存，
        # 确保本 epoch 的 backward() 能分配到连续显存块。
        # 仅在 CUDA 设备上执行；对 CPU/MPS 无影响。
        # 注意：此处清理缓存（释放碎片化 reserved 块），而非清理活跃内存
        # （模型参数/优化器状态等活跃张量不受影响）。
        # gc.collect() must run BEFORE empty_cache(): Python's reference
        # counting normally frees tensors immediately, but reference cycles
        # (common in complex module graphs) are only broken by the cyclic
        # garbage collector.  Running gc.collect() first ensures all
        # unreachable CUDA tensors are released so empty_cache() can
        # return their memory to the CUDA allocator pool.
        if self._device_type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
        
        # Advance epoch counter in the adaptive loss balancer so that the warmup
        # period expires correctly and weight adaptation becomes active.
        if self.use_adaptive_loss:
            self.loss_balancer.set_epoch(epoch or 0)
        
        # Log start of epoch
        if epoch is not None:
            if epoch == 1:
                logger.info("🚀 开始训练... (首个epoch可能因模型编译而较慢)")
            elif epoch <= 3:
                logger.info(f"📊 Epoch {epoch}/{total_epochs or '?'} 训练中...")
        
        for i, data in enumerate(epoch_data):
            # ── 梯度累积控制 ────────────────────────────────────────────────────
            # 每 ga 步执行一次 optimizer.step()，等效 batch size = ga × 1。
            # loss 除以 ga 以保持梯度期望不随 ga 变化（梯度期望 = Σ∇L_i / ga）。
            # do_zero_grad=True 仅在每组累积的第一步清零，避免丢失已累积梯度。
            is_accum_boundary = (i + 1) % ga == 0 or i == num_batches - 1
            loss_dict = self.train_step(
                data,
                do_zero_grad=(i % ga == 0),
                do_optimizer_step=is_accum_boundary,
                loss_scale=1.0 / ga,
            )
            total_loss += loss_dict.get('total_raw', loss_dict['total'])
            
            # ── 精细碎片控制（epoch 内周期清理）─────────────────────────────
            # 每 _cuda_clear_interval 步执行一次 gc.collect() + empty_cache()。
            # 动机：启用 windowed_sampling 且被试数较多时（如 8 被试 × 10 窗口 =
            # 80 步/epoch），每步分配/释放约 44-88MB 消息张量（temporal_chunk_size=32-64）。
            # 这些块被释放后留在 CUDA 分配器的空闲列表中，多步累积后形成碎片；
            # epoch 末尾的某一步请求较大连续块时可能 OOM，即使 reserved>>allocated。
            # 每 50 步清理一次可将空闲列表碎片控制在 50 步的积累量以内，
            # 代价约 2-5ms/次，对总训练时间影响可忽略不计。
            #
            # ⚠ 注意：不要加 is_accum_boundary 条件。
            # 原始设计曾用 `is_accum_boundary AND (i+1)%interval==0`，
            # 但这使得实际触发步为 LCM(ga, interval) 而非 interval：
            # ga=4, interval=50 → LCM=100，80步/epoch 从不触发（完全无效）。
            # gc.collect() + empty_cache() 仅回收**已释放**的 CUDA 块，
            # 不会影响仍在 autograd 图中的梯度张量，故在非边界步也安全。
            if (
                self._device_type == 'cuda'
                and self._cuda_clear_interval > 0
                and (i + 1) % self._cuda_clear_interval == 0
            ):
                gc.collect()
                torch.cuda.empty_cache()

            # Log progress for longer training runs (every 10 batches or at 25%, 50%, 75%)
            if num_batches > 10 and i > 0 and (i % 10 == 0 or i == num_batches // 4 or i == num_batches // 2 or i == 3 * num_batches // 4):
                progress_pct = (i + 1) / num_batches * 100
                avg_loss_so_far = total_loss / (i + 1)
                logger.info(f"  进度: {i+1}/{num_batches} batches ({progress_pct:.0f}%) - 当前平均loss: {avg_loss_so_far:.4f}")
        
        avg_loss = total_loss / len(data_list)
        self.history['train_loss'].append(avg_loss)
        
        # Release fragmented GPU memory blocks once per epoch (end of epoch).
        # Complements the cache clear at the start of next epoch so that both
        # the tail of this epoch and the head of the next start from a low-
        # fragmentation state.
        if self._device_type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
        
        # Step scheduler (if not ReduceLROnPlateau)
        if self.use_scheduler and self.scheduler is not None:
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
        
        return avg_loss
    
    def step_scheduler_on_validation(self, val_loss: float):
        """
        Step scheduler based on validation loss (for ReduceLROnPlateau).
        
        Args:
            val_loss: Validation loss
        """
        if self.use_scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
    
    @staticmethod
    def _r2_from_accum(
        ss_res: float, ss_raw: float, ss_sum: float, ss_cnt: int
    ) -> float:
        """Compute R² from online accumulators (single-pass, global mean).

        Uses the algebraic identity ``SS_tot = Σy² - n·ȳ²`` so no second pass
        over the data is needed and the global mean is exact (not per-sample).

        Args:
            ss_res: Σ(y − ŷ)² accumulated over all samples.
            ss_raw: Σy²        accumulated over all samples.
            ss_sum: Σy         accumulated over all samples.
            ss_cnt: n          total number of elements.

        Returns:
            R² ∈ (−∞, 1].  Returns 0.0 when SS_tot ≤ 1e-12 (constant signal).
        """
        if ss_cnt > 0:
            global_mean = ss_sum / ss_cnt
            ss_tot = ss_raw - ss_cnt * global_mean ** 2
        else:
            ss_tot = 0.0
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    @torch.no_grad()
    def validate(self, data_list: List[HeteroData]) -> Tuple[float, Dict[str, float]]:
        """
        Validation pass.

        Computes the same loss terms as training (reconstruction + prediction)
        so that early stopping is driven by the actual optimisation objective.

        Metrics computed
        ----------------
        1. Reconstruction R² (r2_<node_type>):
           How well the model reconstructs the input signal.
           R² = 1 − SS_res/SS_tot.  R² = 1 → perfect; R² = 0 → mean baseline;
           R² < 0 → worse than mean baseline (failure).

        2. Signal-space prediction R² (pred_r2_<node_type>):
           **Primary quality indicator** — how well the model predicts future
           brain signals from past signals.
           Procedure:
           a) Slice raw signal to T_ctx steps (context only).
           b) Re-encode the context-only slice through the full encoder.
              This is the TRULY CAUSAL encoding: the encoder cannot see any
              signal beyond T_ctx, regardless of its bidirectional structure.
           c) Run predictor on causal context latent → predicted latent.
           d) Decode to signal space → predicted signal.
           e) Compare against raw future signal [T_ctx:T_ctx+pred_steps].
           This is the real "digital-twin" capability metric — measured without
           any future information leakage from the bidirectional encoder.

        Args:
            data_list: List of validation data

        Returns:
            Tuple of:
                avg_loss: Average validation loss (scalar)
                r2_dict: Per-modality metrics dict, keys:
                    'r2_<nt>'      — reconstruction R²
                    'pred_r2_<nt>' — signal-space prediction R² (★ primary)
        """
        self.model.eval()
        total_loss = 0.0
        # Accumulators for reconstruction R² using global mean (V5.38).
        # Using per-sample means (the old approach) systematically inflates R²
        # because SS_tot only captures within-sample variance (excludes between-sample
        # variance), making the denominator artificially small.  A smaller SS_tot
        # means R² = 1 - SS_res/SS_tot is closer to 1 — giving an overly optimistic
        # metric.  We instead accumulate Σy, Σy², and count so the global mean can
        # be recovered in a single pass without storing all y:
        #
        #   SS_tot_correct = Σy² - n*ȳ²   (algebraic identity: Σ(y-ȳ)² = Σy² - n*ȳ²)
        #
        # This gives the same result as computing SS_tot with the true global mean,
        # while requiring only O(1) extra state (sum and count per modality).
        ss_res: Dict[str, float] = {nt: 0.0 for nt in self.node_types}  # Σ(y-ŷ)²
        ss_raw: Dict[str, float] = {nt: 0.0 for nt in self.node_types}  # Σy²
        ss_sum: Dict[str, float] = {nt: 0.0 for nt in self.node_types}  # Σy
        ss_cnt: Dict[str, int]   = {nt: 0   for nt in self.node_types}  # n
        # Same four accumulators for signal-space prediction R²
        pred_ss_res: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        pred_ss_raw: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        pred_ss_sum: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        pred_ss_cnt: Dict[str, int]   = {nt: 0   for nt in self.node_types}

        # ── AR(1) baseline and h=1 horizon accumulators (V5.47) ────────────
        # Two separate AR(1) baselines serve different purposes:
        #
        # ar1_ss_res  — MULTI-STEP constant-last-value baseline.
        #   Predicts y_{T_ctx+h} = y_{T_ctx} for ALL h in [1, pred_steps].
        #   Shares target statistics (ss_raw/sum/cnt) with the full pred_r2.
        #   NOTE: for z-scored signals this baseline can yield ar1_r2 < 0 because
        #   holding the last value constant over many steps is worse than predicting
        #   the global mean.  BOLD ρ ≈ 0.85-0.95 only guarantees high R² at h=1;
        #   over 15 TRs (30 s) the constant prediction degrades severely.
        #   decorr = (pred_r2 - ar1_r2) / (1 - ar1_r2) is therefore the skill score
        #   of the MULTI-STEP model relative to the MULTI-STEP constant baseline.
        #
        # ar1_h1_ss_res  — H=1 ONLY constant-last-value baseline (★ NPI-comparable).
        #   Predicts y_{T_ctx+1} = y_{T_ctx} — the single next step only.
        #   For BOLD at TR=2s this should yield ar1_r2_h1 ≈ 0.7-0.9 (the "free"
        #   autocorrelation R²).  decorr_h1 = (pred_r2_h1 - ar1_r2_h1) / (1 - ar1_r2_h1)
        #   is the direct, apples-to-apples NPI skill score: > 0 means TwinBrain's
        #   h=1 prediction beats trivial autocorrelation.
        ar1_ss_res: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        # h=1 horizon: next-step-only prediction (apples-to-apples vs NPI 3→1).
        pred_h1_ss_res: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        pred_h1_ss_raw: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        pred_h1_ss_sum: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        pred_h1_ss_cnt: Dict[str, int]   = {nt: 0   for nt in self.node_types}
        # h=1 AR(1) baseline (same target statistics as pred_h1_ss_*)
        ar1_h1_ss_res: Dict[str, float] = {nt: 0.0 for nt in self.node_types}

        for data in data_list:
            data = data.to(self.device)

            # Forward pass — always request encoded representations so that:
            # a) compute_loss() can include the latent prediction loss, and
            # b) we can compute signal-space prediction R² below.
            if self.model.use_prediction:
                reconstructed, _, encoded = self.model(
                    data, return_prediction=False, return_encoded=True
                )
            else:
                reconstructed, _ = self.model(data, return_prediction=False)
                encoded = None

            losses = self.model.compute_loss(data, reconstructed, encoded=encoded)
            total_loss += sum(losses.values()).item()

            # ── Reconstruction R² per modality ──────────────────────────
            for node_type in self.node_types:
                if node_type in data.node_types and node_type in reconstructed:
                    target = data[node_type].x       # [N, T, C]
                    recon  = reconstructed[node_type]  # [N, T', C]
                    T_min  = min(target.shape[1], recon.shape[1])
                    target = target[:, :T_min, :]
                    recon  = recon[:, :T_min, :]
                    if recon.shape[0] != target.shape[0]:
                        continue
                    ss_res[node_type] += ((target - recon) ** 2).sum().item()
                    ss_raw[node_type] += (target ** 2).sum().item()
                    ss_sum[node_type] += target.sum().item()
                    ss_cnt[node_type] += target.numel()

            # ── Signal-space prediction R² per modality ─────────────────
            # TRULY CAUSAL evaluation:
            #
            # The encoder uses symmetric Conv1d padding (±1-step boundary
            # leakage) and CAUSAL TemporalAttention (is_causal=True, V5.42
            # fix).  With causal attention the h[:,t,:] latent contains ONLY
            # information from signal[0..t], eliminating global future leakage.
            #
            # For the strictest possible metric we still re-encode with ONLY
            # the first T_ctx raw signal timesteps.  This removes even the
            # ±1-step Conv1d boundary leakage at position T_ctx-1 (which
            # affects fewer than 1% of timesteps and is otherwise negligible).
            # Cost: one extra encoder forward pass without backprop (acceptable
            # in a validation loop).
            #
            # Result: pred_R² here is a conservative lower bound that neither
            # the causal-attention nor the boundary-Conv1d approximation can
            # inflate.  This is the authoritative benchmark.
            if self.model.use_prediction:
                pred_latents: Dict[str, torch.Tensor] = {}
                pred_T_ctx: Dict[str, int] = {}
                # h_ctx_dict is assigned inside `if ctx_T_map:` below and
                # referenced inside `if pred_latents:`.  The current logic
                # guarantees pred_latents non-empty → ctx_T_map non-empty →
                # h_ctx_dict assigned, but initializing here makes this
                # invariant explicit and prevents UnboundLocalError if the
                # code is extended in the future.
                h_ctx_dict: Dict[str, torch.Tensor] = {}

                # ── Step 1: Determine context split per modality ──────────
                # Use the full-sequence latent length (from encoded above,
                # if available) to compute T_ctx; otherwise fall back to the
                # raw-signal length.
                ctx_T_map: Dict[str, int] = {}
                for node_type in self.node_types:
                    if node_type not in data.node_types:
                        continue
                    if encoded is not None and node_type in encoded:
                        T = encoded[node_type].shape[1]
                    else:
                        T = data[node_type].x.shape[1]
                    if T < self.model._PRED_MIN_SEQ_LEN:
                        continue
                    ctx_T_map[node_type] = int(T * self.model._PRED_CONTEXT_RATIO)

                if ctx_T_map:
                    # ── Step 2: Build context-only data and encode ────────
                    # Each node type is sliced to its T_ctx steps so the
                    # encoder only sees past data — no future leakage.
                    context_data = data.clone()
                    for node_type, T_ctx in ctx_T_map.items():
                        context_data[node_type].x = (
                            data[node_type].x[:, :T_ctx, :]
                        )
                    _, _, h_ctx_dict = self.model(
                        context_data,
                        return_prediction=False,
                        return_encoded=True,
                    )

                    # ── Step 3: Predict from causal context ───────────────
                    for node_type, T_ctx in ctx_T_map.items():
                        if node_type not in h_ctx_dict:
                            continue
                        h_ctx = h_ctx_dict[node_type]  # [N, T_ctx, H]
                        # Efficiency: only generate steps that can be compared
                        # against the actual future signal.  For fMRI (T_fut=17
                        # with prediction_steps=50), generating all 50 steps
                        # wastes 66% of predictor computation for no gain.
                        T_total = data[node_type].x.shape[1]
                        T_fut = T_total - T_ctx
                        effective_steps = min(
                            self.model.prediction_steps, max(1, T_fut)
                        )
                        pred_latents[node_type] = (
                            self.model.predictor.predict_next(
                                h_ctx, num_steps=effective_steps
                            )
                        )
                        pred_T_ctx[node_type] = T_ctx

                # ── Step 4: System-level graph propagation ────────────────
                if pred_latents:
                    pred_latents = self.model.prediction_propagator(
                        pred_latents, data
                    )

                if pred_latents:
                    # ── Step 5: Decode latent predictions to signal space ─
                    # Seed pred_enc from h_ctx_dict (encoded context latents,
                    # shape [N, T_ctx, H=hidden_channels]) rather than raw data
                    # (shape [N, T, C=1]).  The decoder's first Conv1d expects
                    # in_channels=hidden_channels, so starting from the raw data
                    # would crash for any modality not overridden by pred_latents
                    # (e.g. modalities skipped because T < _PRED_MIN_SEQ_LEN).
                    # Starting from h_ctx_dict ensures every modality has the
                    # correct feature dimension regardless of whether it was
                    # predicted; the pred_T_ctx guard in Step 6 prevents those
                    # modalities from contributing to the pred_R² metric.
                    #
                    # Tensor aliasing: data.clone() performs a deep copy of all
                    # node/edge tensors (PyG HeteroData.clone() calls .clone() on
                    # each stored tensor).  The subsequent .x assignments create
                    # new references in pred_enc only; data is not modified.
                    # This validate() function runs under @torch.no_grad() so
                    # there are no gradient graphs that could be affected.
                    pred_enc = data.clone()
                    for nt, h_ctx in h_ctx_dict.items():
                        pred_enc[nt].x = h_ctx   # [N, T_ctx, H] — correct H for decoder
                    for nt, pred_lat in pred_latents.items():
                        pred_enc[nt].x = pred_lat  # override with predicted latent
                    # Guard: node types NOT in h_ctx_dict (T < _PRED_MIN_SEQ_LEN=4)
                    # still carry the raw signal [N, T, C=1] from data.clone().
                    # The decoder's Conv1d(in_channels=hidden_channels) would fail
                    # with a channel mismatch on [N, 1, T].
                    # This is extremely rare in practice (typical T ≈ 300 >> 4)
                    # but could occur in short-sequence edge cases or unit tests.
                    # Fix: remove those node types from pred_enc so the decoder
                    # only processes nodes that have correct latent features.
                    for nt in list(pred_enc.node_types):
                        if nt not in h_ctx_dict:
                            logger.debug(
                                f"validate: removing '{nt}' from pred_enc "
                                f"(T={data[nt].x.shape[1]} < _PRED_MIN_SEQ_LEN="
                                f"{self.model._PRED_MIN_SEQ_LEN}; raw signal "
                                f"would cause decoder channel mismatch)"
                            )
                            del pred_enc[nt]
                    pred_signals = self.model.decoder(pred_enc)  # {nt: [N, pred_steps', C]}

                    for node_type, pred_sig in pred_signals.items():
                        if node_type not in data.node_types:
                            continue
                        T_ctx = pred_T_ctx.get(node_type)
                        if T_ctx is None:
                            continue
                        # ── Step 6: Compare against raw future signal ─────
                        # Uses the RAW signal (not the latent) so the metric
                        # is interpretable in physical units (z-scored signal).
                        future_sig = data[node_type].x[:, T_ctx:, :]  # [N, T_fut, C]
                        n_steps = min(pred_sig.shape[1], future_sig.shape[1])
                        if n_steps < 1:
                            continue
                        if pred_sig.shape[0] != future_sig.shape[0]:
                            continue
                        pred_aligned   = pred_sig[:, :n_steps, :]
                        future_aligned = future_sig[:, :n_steps, :]
                        pred_ss_res[node_type] += ((future_aligned - pred_aligned) ** 2).sum().item()
                        pred_ss_raw[node_type] += (future_aligned ** 2).sum().item()
                        pred_ss_sum[node_type] += future_aligned.sum().item()
                        pred_ss_cnt[node_type] += future_aligned.numel()
                        # AR(1) baseline (multi-step): constant "last context step"
                        # prediction expanded over ALL future steps.  Shares target
                        # statistics with pred_r2 accumulators.
                        _last_obs = data[node_type].x[:, T_ctx - 1 : T_ctx, :]  # [N, 1, C]
                        _ar1_pred = _last_obs.expand_as(future_aligned)           # [N, n_steps, C]
                        ar1_ss_res[node_type] += ((future_aligned - _ar1_pred) ** 2).sum().item()
                        # h=1 horizon: first predicted step only (comparable to NPI 3→1).
                        _f1 = future_aligned[:, :1, :]
                        _p1 = pred_aligned[:, :1, :]
                        pred_h1_ss_res[node_type] += ((_f1 - _p1) ** 2).sum().item()
                        pred_h1_ss_raw[node_type] += (_f1 ** 2).sum().item()
                        pred_h1_ss_sum[node_type] += _f1.sum().item()
                        pred_h1_ss_cnt[node_type] += _f1.numel()
                        # AR(1) h=1 baseline (★ NPI-comparable): constant last value vs
                        # the single next step.  For BOLD at TR=2s this should yield
                        # ar1_r2_h1 ≈ 0.7-0.9.  Shares target statistics with pred_h1.
                        ar1_h1_ss_res[node_type] += ((_f1 - _last_obs) ** 2).sum().item()

        avg_loss = total_loss / len(data_list)
        self.history['val_loss'].append(avg_loss)

        # ── Assemble R² dict via _r2_from_accum() helper ────────────────────
        r2_dict: Dict[str, float] = {}

        # Reconstruction R²
        for node_type in self.node_types:
            r2 = self._r2_from_accum(
                ss_res[node_type], ss_raw[node_type],
                ss_sum[node_type], ss_cnt[node_type],
            )
            r2_dict[f'r2_{node_type}'] = r2
            key = f'val_r2_{node_type}'
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(r2)

        # Signal-space prediction R² (★ primary metric)
        if self.model.use_prediction:
            for node_type in self.node_types:
                pred_r2 = self._r2_from_accum(
                    pred_ss_res[node_type], pred_ss_raw[node_type],
                    pred_ss_sum[node_type], pred_ss_cnt[node_type],
                )
                r2_dict[f'pred_r2_{node_type}'] = pred_r2
                key = f'val_pred_r2_{node_type}'
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(pred_r2)

            # AR(1) baseline R², decorrelation scores, h=1 R² (V5.47 / V5.50)
            # ─────────────────────────────────────────────────────────────────────
            # Four derived metrics provide scientifically rigorous NPI comparison:
            #
            # 1. ar1_r2_{nt}: MULTI-STEP constant baseline R².
            #    Predicts y_{T_ctx+h} = y_{T_ctx} for h in [1, pred_steps].
            #    Can be NEGATIVE for fast-changing signals (EEG) or long horizons
            #    (fMRI > 10 TRs) because holding the last value constant over many
            #    steps becomes worse than predicting the global mean.
            #    Note: "AR(1) ≈ 0.7-0.9 for h=1" does NOT apply here — that claim
            #    holds only for the single next step, not averaged over all steps.
            #
            # 2. decorr_{nt}: MULTI-STEP skill score.
            #    = (pred_r2 − ar1_r2) / (1 − ar1_r2)
            #    Measures how much of the gap between AR(1) and perfect prediction
            #    TwinBrain closes when predicting ALL future steps.
            #    > 0 → outperforms constant-last-value multi-step baseline.
            #
            # 3. ar1_r2_h1_{nt}: H=1 ONLY constant baseline R² (★ NPI-comparable).
            #    Predicts y_{T_ctx+1} = y_{T_ctx} — the SINGLE next step only.
            #    For BOLD at TR=2s this should yield ar1_r2_h1 ≈ 0.7-0.9 because
            #    BOLD is strongly autocorrelated (ρ ≈ 0.85-0.95) at lag 1.
            #    For EEG at 4ms this may also be high (ρ is large at 4ms).
            #    This is the "free" R² any trivial predictor captures at h=1.
            #
            # 4. decorr_h1_{nt}: H=1 ONLY skill score (★ NPI-comparable).
            #    = (pred_r2_h1 − ar1_r2_h1) / (1 − ar1_r2_h1)
            #    > 0 → TwinBrain's h=1 prediction beats trivial autocorrelation.
            #    < 0 → model does NOT beat trivial autocorrelation at h=1 (weak).
            #    This is the direct, apples-to-apples NPI comparison metric.
            for node_type in self.node_types:
                # AR(1) multi-step R² (shares target stats with pred_r2 accumulators)
                ar1_r2 = self._r2_from_accum(
                    ar1_ss_res[node_type],
                    pred_ss_raw[node_type],
                    pred_ss_sum[node_type],
                    pred_ss_cnt[node_type],
                )
                r2_dict[f'ar1_r2_{node_type}'] = ar1_r2

                # Multi-step decorrelation score: skill relative to constant baseline
                pred_r2_val = r2_dict.get(f'pred_r2_{node_type}', 0.0)
                decorr = (pred_r2_val - ar1_r2) / max(1e-3, 1.0 - ar1_r2)
                r2_dict[f'decorr_{node_type}'] = float(decorr)

                # h=1 horizon R² (next-step only, comparable to NPI's 3→1)
                pred_r2_h1 = self._r2_from_accum(
                    pred_h1_ss_res[node_type],
                    pred_h1_ss_raw[node_type],
                    pred_h1_ss_sum[node_type],
                    pred_h1_ss_cnt[node_type],
                )
                r2_dict[f'pred_r2_h1_{node_type}'] = pred_r2_h1

                # h=1 AR(1) R² (★ NPI-comparable: shares target stats with pred_h1)
                ar1_r2_h1 = self._r2_from_accum(
                    ar1_h1_ss_res[node_type],
                    pred_h1_ss_raw[node_type],
                    pred_h1_ss_sum[node_type],
                    pred_h1_ss_cnt[node_type],
                )
                r2_dict[f'ar1_r2_h1_{node_type}'] = ar1_r2_h1

                # h=1 decorrelation score (★ NPI-comparable skill score)
                decorr_h1 = (pred_r2_h1 - ar1_r2_h1) / max(1e-3, 1.0 - ar1_r2_h1)
                r2_dict[f'decorr_h1_{node_type}'] = float(decorr_h1)

        return avg_loss, r2_dict
    
    def save_checkpoint(self, path: Path, epoch: int):
        """Save training checkpoint with atomic write for safety."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        if self.use_adaptive_loss:
            checkpoint['loss_balancer_state'] = self.loss_balancer.state_dict()
        
        # Save scheduler state so that LR schedule resumes correctly after loading.
        # Without this, load_checkpoint() would restart the LR from the initial
        # value + warmup phase, disrupting cosine-annealing restarts.
        if self.use_scheduler and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Atomic save: write to temp file then rename
        temp_path = path.parent / f"{path.name}.tmp"
        
        try:
            torch.save(checkpoint, temp_path)
            # Atomic rename (on most filesystems)
            temp_path.replace(path)
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")
            raise RuntimeError(f"Checkpoint save failed: {e}")
    
    def adapt_to_subject(
        self,
        data_list: List[HeteroData],
        subject_idx: int,
        num_steps: int = 100,
        lr: float = 5e-3,
        verbose: bool = True,
    ) -> None:
        """Few-shot personalization: fine-tune only the subject embedding for a new subject.

        Freezes all model parameters **except** ``model.subject_embed.weight[subject_idx]``
        and runs ``num_steps`` gradient descent steps minimising the reconstruction loss.
        This is extremely parameter-efficient: only H parameters (e.g. 128) are updated,
        making personalisation feasible with as few as 10–50 training windows.

        Scientific motivation: the encoder has already learned the shared functional
        connectivity structure from training subjects.  A new subject's brain differs
        mainly in its baseline activity level and connection-strength offsets — captured
        by the subject embedding without altering shared topology knowledge.

        Use this after training on multiple subjects to quickly adapt the model to a
        new subject at inference time, without re-training the full model.

        Args:
            data_list: Windows from the new subject.  As few as 10–50 windows
                (~20–100 TR of fMRI, or ~40 s of EEG) typically suffice.
            subject_idx: Integer index in ``model.subject_embed``.
                Must be < ``model.num_subjects``.  Allocate an unused index
                (e.g. ``num_subjects`` + 1) for a genuinely new subject,
                or reuse an existing index to override that subject's embedding.
            num_steps: Gradient descent steps (default 100).
            lr: Learning rate for the embedding update.  Default 5e-3 is
                higher than the main training LR (1e-4) because only H
                parameters are updated — the optimisation landscape is well-
                conditioned and converges quickly.
            verbose: Log loss every 10 steps.

        Raises:
            RuntimeError: If the model has ``num_subjects == 0`` (no embedding table).
            ValueError:   If ``subject_idx >= model.num_subjects``.
        """
        if self.model.num_subjects == 0 or not hasattr(self.model, 'subject_embed'):
            raise RuntimeError(
                "adapt_to_subject requires num_subjects > 0 during training. "
                "Re-train the model with at least 2 subjects."
            )
        if subject_idx >= self.model.num_subjects:
            raise ValueError(
                f"subject_idx={subject_idx} exceeds model.num_subjects="
                f"{self.model.num_subjects}."
            )

        # Freeze all parameters except the subject embedding table.
        # We freeze at the whole-parameter level (requires_grad=False) for
        # all params, then unfreeze only subject_embed.weight.  This is correct
        # because PyTorch's autograd cannot track gradients through a tensor
        # obtained by index-slicing a nn.Parameter (the slice is not a leaf
        # variable).  By unfreezing the whole weight tensor, autograd computes
        # gradients for all subject slots, but only subject_idx has non-zero
        # input gradient because no other row participates in the loss.
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.subject_embed.weight.requires_grad_(True)

        adapt_optimizer = torch.optim.Adam([self.model.subject_embed.weight], lr=lr)
        self.model.train()

        if verbose:
            logger.info(
                f"Adapting to subject_idx={subject_idx}: "
                f"{num_steps} steps, {len(data_list)} windows, lr={lr}"
            )

        for step in range(num_steps):
            sample = random.choice(data_list).to(self.device)
            adapt_optimizer.zero_grad()

            _orig_idx = getattr(sample, 'subject_idx', None)
            sample.subject_idx = torch.tensor(
                subject_idx, dtype=torch.long, device=self.device
            )
            try:
                reconstructed, _ = self.model(sample, return_prediction=False)
                losses = self.model.compute_loss(sample, reconstructed)
                total_loss = sum(losses.values())
                total_loss.backward()
                # Zero out gradients for ALL rows EXCEPT subject_idx before updating.
                # This ensures the optimizer only updates the target subject's row,
                # leaving other subjects' embeddings untouched.
                # Convention: keep_mask[subject_idx] = 1.0, all others = 0.0.
                with torch.no_grad():
                    if self.model.subject_embed.weight.grad is not None:
                        keep_mask = torch.zeros(
                            self.model.num_subjects, 1,
                            device=self.model.subject_embed.weight.device,
                        )
                        keep_mask[subject_idx] = 1.0
                        self.model.subject_embed.weight.grad.mul_(keep_mask)
                adapt_optimizer.step()
            finally:
                # Always restore the original subject_idx (non-destructive).
                # If the sample didn't have subject_idx originally, remove it to
                # avoid polluting the sample with a spurious attribute.
                if _orig_idx is not None:
                    sample.subject_idx = _orig_idx
                elif hasattr(sample, 'subject_idx'):
                    del sample.subject_idx

            if verbose and (step + 1) % 10 == 0:
                logger.info(
                    f"  adapt step {step+1}/{num_steps}: loss={total_loss.item():.4f}"
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
                f"Adaptation done: subject_idx={subject_idx}, "
                f"embedding L2={embed_norm:.4f}"
            )

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        # weights_only=False is required for loading HeteroData and custom objects
        # stored in checkpoints.  Omitting it triggers a FutureWarning in recent
        # PyTorch versions and will become an error in a future release.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.use_adaptive_loss and 'loss_balancer_state' in checkpoint:
            self.loss_balancer.load_state_dict(checkpoint['loss_balancer_state'])
        
        # Restore scheduler state to continue LR annealing from where it left off.
        # Old checkpoints (saved before this fix) will not have 'scheduler_state_dict';
        # in that case the scheduler restarts from epoch 0 (pre-fix behaviour).
        if self.use_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif self.use_scheduler and self.scheduler is not None:
            logger.warning(
                "Checkpoint does not contain scheduler_state_dict "
                "(checkpoint was saved before V5.22). "
                "LR schedule will restart from epoch 0."
            )
        
        logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint['epoch']
