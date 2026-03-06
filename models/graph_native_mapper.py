"""
Graph-Native Brain Mapper
=========================

Complete reimagination of brain data mapping that:
1. Builds and MAINTAINS graph structure throughout pipeline
2. No graph → sequence → graph conversions
3. Native small-world network representation
4. Efficient spatial-temporal modeling on graphs

Philosophy:
- Brain = Graph (small-world network)
- Keep it as graph from data loading to training
- Leverage graph structure for interpretability
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops, to_undirected
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum number of time-points required for a meaningful Pearson correlation.
# With fewer than this many samples the z-score denominator can be near-zero
# and the correlation estimate is statistically unreliable.
_MIN_CORR_POINTS: int = 2


def _row_zscore(mat: torch.Tensor) -> torch.Tensor:
    """Row-wise z-score normalisation.  Returns (mat - row_mean) / (row_std + eps)."""
    mu  = mat.mean(dim=1, keepdim=True)
    std = mat.std(dim=1, keepdim=True) + 1e-8
    return (mat - mu) / std

class GraphNativeBrainMapper:
    """
    Unified brain data mapper that maintains graph structure.
    
    Key Principles:
    1. Build graph ONCE from anatomical/functional connectivity
    2. Keep node features as temporal signals ON the graph
    3. No unnecessary conversions
    4. Support both fMRI and EEG modalities
    """
    
    def __init__(
        self,
        atlas_name: str = 'schaefer200',
        preserve_temporal: bool = True,
        add_self_loops: bool = True,
        make_undirected: bool = True,
        k_nearest_fmri: int = 20,
        k_nearest_eeg: int = 10,
        threshold_fmri: float = 0.3,
        threshold_eeg: float = 0.2,
        device: Optional[str] = None,
        eeg_connectivity_method: str = 'correlation',
    ):
        """
        Initialize graph-native mapper.
        
        Args:
            atlas_name: Brain atlas for parcellation
            preserve_temporal: Keep temporal dimension in node features
            add_self_loops: Add self-connections to graph
            make_undirected: Symmetrize edge connections
            k_nearest_fmri: Number of nearest neighbors for fMRI graph (default: 20)
            k_nearest_eeg: Number of nearest neighbors for EEG graph (default: 10)
            threshold_fmri: Connectivity threshold for fMRI (default: 0.3)
            threshold_eeg: Connectivity threshold for EEG (default: 0.2)
            device: Device to create tensors on ('cpu', 'cuda', or None for auto-detect)
            eeg_connectivity_method: Method for EEG connectivity estimation.
                'correlation' (default): Pearson correlation — fast, backward-compatible.
                'coherence': Wideband magnitude squared coherence — captures
                    frequency-domain oscillatory coupling (alpha/beta/gamma bands)
                    more faithfully than time-domain correlation.
                    Produces values in [0, 1]. Changes cache key automatically.
        """
        self.atlas_name = atlas_name
        self.preserve_temporal = preserve_temporal
        self.add_self_loops_flag = add_self_loops
        self.make_undirected_flag = make_undirected
        self.k_nearest_fmri = k_nearest_fmri
        self.k_nearest_eeg = k_nearest_eeg
        self.threshold_fmri = threshold_fmri
        self.threshold_eeg = threshold_eeg
        self.eeg_connectivity_method = eeg_connectivity_method
        
        # Device management
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Graph structure (built once, reused)
        self.base_graph_structure = None
        self.node_positions = None  # 3D coordinates
        self.node_labels = None  # Region names
    
    def _compute_correlation_gpu(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix using GPU for 5-10x speedup.
        
        Args:
            timeseries: [N, T] time series data
            
        Returns:
            correlation_matrix: [N, N] absolute correlation matrix
        """
        # Move to GPU
        ts_gpu = torch.from_numpy(timeseries).to(self.device, dtype=torch.float32)
        N, T = ts_gpu.shape
        
        # Normalize: subtract mean, divide by std
        ts_mean = ts_gpu.mean(dim=1, keepdim=True)
        ts_std = ts_gpu.std(dim=1, keepdim=True) + 1e-8
        ts_norm = (ts_gpu - ts_mean) / ts_std
        
        # Correlation via matrix multiplication: O(N²T) but GPU-parallel
        correlation = torch.mm(ts_norm, ts_norm.T) / T
        
        # Absolute value for unsigned connectivity
        correlation = torch.abs(correlation)
        
        # Move back to CPU as numpy
        return correlation.cpu().numpy()
        
    def build_graph_structure(
        self,
        connectivity_matrix: Optional[np.ndarray] = None,
        threshold: float = 0.1,
        k_nearest: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build base graph structure from connectivity.
        
        This is the CORE graph that persists throughout pipeline.
        Built once from anatomical/functional connectivity.
        
        Args:
            connectivity_matrix: [N, N] connectivity weights
            threshold: Minimum weight to create edge
            k_nearest: If set, keep only k nearest neighbors per node
            
        Returns:
            edge_index: [2, E] graph edges
            edge_attr: [E, 1] edge weights
        """
        N = connectivity_matrix.shape[0]
        
        # Build adjacency
        if k_nearest is not None:
            # GPU-accelerated K-nearest neighbors (10-20x faster)
            # Move connectivity to GPU
            conn_gpu = torch.from_numpy(connectivity_matrix).to(self.device, dtype=torch.float32)
            
            # Vectorized top-k: [N, N] -> [N, k_nearest]
            # torch.topk is O(N log k) per row, parallelized across all N rows
            k_actual = min(k_nearest, N)
            top_values, top_indices = torch.topk(conn_gpu, k_actual, dim=1)
            
            # Filter by threshold
            mask = top_values > threshold
            
            # Build edge lists efficiently
            # Create row indices for all entries
            row_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, k_actual)
            
            # Filter out self-loops and below-threshold edges
            self_loop_mask = row_idx != top_indices
            valid_mask = mask & self_loop_mask
            
            # Check for nodes with no edges (edge case warning)
            edges_per_node = valid_mask.sum(dim=1)
            if edges_per_node.min() == 0:
                logger.warning(
                    f"Some nodes have 0 edges after filtering "
                    f"(k_nearest={k_nearest}, threshold={threshold}). "
                    f"Consider lowering threshold or increasing k_nearest."
                )
            
            edge_index = torch.stack([
                row_idx[valid_mask],
                top_indices[valid_mask]
            ], dim=0)
            edge_attr = top_values[valid_mask].unsqueeze(-1)
        
        else:
            # Threshold-based
            adj = connectivity_matrix > threshold
            np.fill_diagonal(adj, False)  # No self-loops initially
            
            edge_index = torch.tensor(np.array(np.where(adj)), dtype=torch.long, device=self.device)
            edge_attr = torch.tensor(
                connectivity_matrix[adj],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(-1)
        
        # Make undirected (brain connections are symmetric)
        if self.make_undirected_flag:
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
        
        # Add self-loops (important for GNN message passing)
        if self.add_self_loops_flag:
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                fill_value=1.0,
                num_nodes=N
            )
        
        logger.info(
            f"Built graph: {N} nodes, {edge_index.shape[1]} edges "
            f"(avg degree: {edge_index.shape[1]/N:.1f})"
        )
        
        return edge_index, edge_attr
    
    def map_fmri_to_graph(
        self,
        timeseries: np.ndarray,
        connectivity_matrix: Optional[np.ndarray] = None,
        node_positions: Optional[np.ndarray] = None,
        node_labels: Optional[List[str]] = None,
        sampling_rate: float = 0.5,
    ) -> HeteroData:
        """
        Map fMRI timeseries to graph structure.
        
        KEEPS temporal dimension as node features!
        No flattening or unnecessary transformations.
        
        Args:
            timeseries: [N_rois, T_time] or [T_time, N_rois]
            connectivity_matrix: [N_rois, N_rois] functional connectivity
            node_positions: [N_rois, 3] brain coordinates
            node_labels: ROI names
            sampling_rate: Temporal sampling rate in Hz (TR⁻¹). Default 0.5 Hz = TR 2 s.
                Pass the actual TR from the NIfTI header when available.
            
        Returns:
            HeteroData with 'fmri' node type
        """
        # Ensure shape [N, T]
        if timeseries.ndim != 2:
            raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")
        
        if timeseries.shape[0] > timeseries.shape[1]:
            timeseries = timeseries.T  # Assume [T, N] -> [N, T]
        
        N_rois, T_time = timeseries.shape
        
        # Build graph structure (if not already built)
        if connectivity_matrix is None:
            # Use temporal correlation as connectivity
            # GPU-accelerated correlation (5-10x faster than numpy.corrcoef)
            connectivity_matrix = self._compute_correlation_gpu(timeseries)
        
        edge_index, edge_attr = self.build_graph_structure(
            connectivity_matrix,
            threshold=self.threshold_fmri,
            k_nearest=self.k_nearest_fmri,
        )
        
        # Node features: KEEP temporal dimension
        # Shape: [N, T, 1] - one feature channel (fMRI signal)
        x = torch.tensor(timeseries, dtype=torch.float32, device=self.device).unsqueeze(-1)  # [N, T, 1]
        
        # Create HeteroData
        data = HeteroData()
        
        # fMRI nodes with temporal features
        data['fmri'].x = x  # [N, T, 1]
        data['fmri'].num_nodes = N_rois
        
        # Graph structure
        data['fmri', 'connects', 'fmri'].edge_index = edge_index
        data['fmri', 'connects', 'fmri'].edge_attr = edge_attr
        
        # Metadata
        if node_positions is not None:
            data['fmri'].pos = torch.tensor(node_positions, dtype=torch.float32, device=self.device)
        
        if node_labels is not None:
            data['fmri'].labels = node_labels
        
        # Store temporal info
        data['fmri'].temporal_length = T_time
        data['fmri'].sampling_rate = sampling_rate
        
        logger.info(
            f"Mapped fMRI to graph: {N_rois} nodes, {T_time} timepoints, "
            f"{edge_index.shape[1]} edges"
        )
        
        return data
    
    def map_eeg_to_graph(
        self,
        timeseries: np.ndarray,
        channel_names: List[str],
        connectivity_matrix: Optional[np.ndarray] = None,
        channel_positions: Optional[np.ndarray] = None,
        atlas_mapping: Optional[Dict[str, int]] = None,
        sampling_rate: float = 250.0,
    ) -> HeteroData:
        """
        Map EEG timeseries to graph structure.
        
        Can map EEG channels to atlas ROIs if atlas_mapping provided,
        otherwise treats channels as graph nodes.
        
        Args:
            timeseries: [N_channels, T_time] EEG signals
            channel_names: Channel labels
            connectivity_matrix: [N_channels, N_channels] connectivity
            channel_positions: [N_channels, 3] electrode positions in mm
                (MNE head coordinate frame, converted from meters to mm).
                Stored as `data['eeg'].pos` for visualization and future use.
                NOTE: For distance-based cross-modal edge creation, EEG positions
                (head space) and fMRI ROI positions (MNI space) are in different
                coordinate systems and require coregistration before comparison.
                Without coregistration, `create_simple_cross_modal_edges` (random
                connections) should be used instead of distance-based mapping.
            atlas_mapping: Map channels to atlas ROIs
            sampling_rate: EEG sampling frequency in Hz. Default 250 Hz.
                Pass the actual sfreq from the MNE raw object.
            
        Returns:
            HeteroData with 'eeg' node type
        """
        # Ensure shape [N, T]
        if timeseries.ndim != 2:
            raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")
        
        if timeseries.shape[0] > timeseries.shape[1]:
            timeseries = timeseries.T
        
        N_channels, T_time = timeseries.shape
        
        # Build connectivity if not provided
        if connectivity_matrix is None:
            # Use coherence-based connectivity
            connectivity_matrix = self._compute_eeg_connectivity(timeseries)
        
        # Build graph structure
        edge_index, edge_attr = self.build_graph_structure(
            connectivity_matrix,
            threshold=self.threshold_eeg,
            k_nearest=self.k_nearest_eeg,
        )

        # Per-channel z-score normalisation.
        # MNE stores EEG in Volts (~1e-5 range); fMRI is z-scored to ~1 by
        # NiftiLabelsMasker(standardize=True) or process_fmri_timeseries().
        # Without normalisation the ~1e10 amplitude gap causes the adaptive
        # loss balancer to drive the EEG reconstruction weight toward its
        # minimum (the EEG MSE/Huber loss is negligibly small in absolute
        # terms), so the decoder never learns to reconstruct EEG and R²_eeg
        # collapses to large negative values.
        # Connectivity is computed from the raw timeseries above (Pearson r
        # and MSC are both scale-invariant, so this order is correct).
        ts_mean = timeseries.mean(axis=1, keepdims=True)
        ts_std  = timeseries.std(axis=1, keepdims=True) + 1e-8
        timeseries = (timeseries - ts_mean) / ts_std

        # Node features: temporal EEG signals
        # Shape: [N, T, 1]
        x = torch.tensor(timeseries, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # Create HeteroData
        data = HeteroData()
        
        # EEG nodes with temporal features
        data['eeg'].x = x  # [N, T, 1]
        data['eeg'].num_nodes = N_channels
        
        # Graph structure
        data['eeg', 'connects', 'eeg'].edge_index = edge_index
        data['eeg', 'connects', 'eeg'].edge_attr = edge_attr
        
        # Metadata
        if channel_positions is not None:
            data['eeg'].pos = torch.tensor(channel_positions, dtype=torch.float32, device=self.device)
        
        data['eeg'].labels = channel_names
        data['eeg'].temporal_length = T_time
        data['eeg'].sampling_rate = sampling_rate
        
        # Atlas mapping if provided (for EEG-fMRI alignment)
        if atlas_mapping is not None:
            data['eeg'].atlas_mapping = atlas_mapping
        
        logger.info(
            f"Mapped EEG to graph: {N_channels} channels, {T_time} timepoints, "
            f"{edge_index.shape[1]} edges"
        )
        
        return data
    
    def _compute_eeg_connectivity_spectral(self, timeseries: np.ndarray) -> np.ndarray:
        """Compute wideband magnitude squared coherence (MSC) between EEG channels.

        Coherence captures frequency-domain linear coupling between channels,
        which is more meaningful for oscillatory neural signals than time-domain
        Pearson correlation.  A single-number summary across all frequencies
        (wideband MSC) is used so that the result has the same [N, N] shape and
        value range [0, 1] as correlation-based connectivity.

        Algorithm
        ---------
        For channels i and j at discrete frequency bin f:
            G_xi(f) = FFT(x_i)[f]
        Wideband MSC:
            wMSC_ij = |mean_f(G_xi × conj(G_xj))|² /
                      (mean_f(|G_xi|²) × mean_f(|G_xj|²))

        Vectorised as matrix operations:
            F: [N, n_freq] complex (rfft output)
            cross_mean = F @ conj(F)^T / n_freq  → [N, N] complex
            psd_mean   = mean(|F|², axis=1)       → [N]
            wMSC = |cross_mean|² / outer(psd_mean, psd_mean)

        The final connectivity value is sqrt(wMSC) (magnitude coherence, range [0,1]),
        chosen for better dynamic range vs. raw MSC which clusters near 1 for
        well-correlated channels.

        References
        ----------
        - Nunez et al. (1997). EEG coherence. Neural Computation.
        - Bullmore & Sporns (2009). Complex brain networks. Nat Rev Neurosci.
        """
        N_ch, T = timeseries.shape
        # rfft for real-valued signals: output shape [N_ch, T//2 + 1]
        # float64 is required for numerical precision in the cross-spectrum
        # computation; only convert if not already float64 to avoid redundant copy.
        ts_f64 = timeseries if timeseries.dtype == np.float64 else timeseries.astype(np.float64)
        F = np.fft.rfft(ts_f64, axis=1)
        n_freq = F.shape[1]

        # Mean cross-power matrix [N_ch, N_ch] complex:
        # cross_mean[i,j] = mean_f( F[i,f] * conj(F[j,f]) )
        cross_mean = np.dot(F, F.conj().T) / n_freq  # [N_ch, N_ch]

        # Mean PSD per channel [N_ch]
        psd_mean = (np.abs(F) ** 2).mean(axis=1)  # [N_ch]

        # Wideband magnitude squared coherence [N_ch, N_ch]
        psd_outer = np.outer(psd_mean, psd_mean) + 1e-12
        msc = (np.abs(cross_mean) ** 2) / psd_outer

        # Convert MSC (magnitude squared coherence, range [0,1²]) to
        # magnitude coherence (sqrt of MSC, range [0,1]).  This is analogous
        # to taking the absolute Pearson correlation: both represent the
        # linear coupling strength between 0 (independent) and 1 (identical),
        # and the sqrt provides better dynamic range since MSC ∈ [0,1] clusters
        # near 1 for strongly coupled channels while sqrt(MSC) spreads the
        # values more uniformly.  The result is directly comparable to the
        # absolute Pearson correlation used in correlation-based connectivity.
        connectivity = np.sqrt(np.clip(msc, 0.0, 1.0)).astype(np.float32)
        np.fill_diagonal(connectivity, 1.0)
        return connectivity

    def _compute_eeg_connectivity(self, timeseries: np.ndarray) -> np.ndarray:
        """Compute EEG connectivity matrix.

        Dispatches to either Pearson correlation (fast, default, backward-compatible)
        or wideband spectral coherence (neuroscientifically superior for oscillatory
        EEG signals) based on ``self.eeg_connectivity_method``.

        Args:
            timeseries: [N_channels, T_time]

        Returns:
            connectivity: [N_channels, N_channels] values in [0, 1]
        """
        if self.eeg_connectivity_method == 'coherence':
            connectivity = self._compute_eeg_connectivity_spectral(timeseries)
            logger.debug(
                f"EEG connectivity: wideband spectral coherence "
                f"(N={timeseries.shape[0]}, T={timeseries.shape[1]})"
            )
        else:
            # Default: GPU-accelerated Pearson correlation (same as fMRI)
            connectivity = self._compute_correlation_gpu(timeseries)

        # Ensure valid values
        connectivity = np.nan_to_num(connectivity, nan=0.0)
        np.fill_diagonal(connectivity, 1.0)
        return connectivity
    
    def _get_graph_device(self, data: HeteroData) -> torch.device:
        """
        Determine the device of a HeteroData graph by checking multiple sources.
        
        Args:
            data: HeteroData graph
            
        Returns:
            Device of the graph (falls back to self.device if not determinable)
        """
        # Try to find device from node features first
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                return data[node_type].x.device
        
        # Try to find device from edge indices
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index') and data[edge_type].edge_index is not None:
                return data[edge_type].edge_index.device
        
        # Fallback to mapper's default device
        return self.device
    
    def create_cross_modal_edges(
        self,
        data: HeteroData,
        eeg_to_fmri_mapping: Optional[Dict[int, int]] = None,
        distance_threshold: Optional[float] = None,
    ) -> HeteroData:
        """
        Add edges between EEG and fMRI nodes.
        
        Creates the cross-modal graph for joint modeling.
        
        Args:
            data: HeteroData with both 'eeg' and 'fmri' nodes
            eeg_to_fmri_mapping: Direct channel->ROI mapping
            distance_threshold: Max spatial distance for connections
            
        Returns:
            HeteroData with cross-modal edges added
        """
        if 'eeg' not in data.node_types or 'fmri' not in data.node_types:
            return data
        
        N_eeg = data['eeg'].num_nodes
        N_fmri = data['fmri'].num_nodes
        
        # Method 1: Direct mapping if provided
        if eeg_to_fmri_mapping is not None:
            edge_list = []
            for eeg_idx, fmri_idx in eeg_to_fmri_mapping.items():
                if 0 <= eeg_idx < N_eeg and 0 <= fmri_idx < N_fmri:
                    edge_list.append([eeg_idx, fmri_idx])
            
            if edge_list:
                # Use helper method to determine device
                target_device = self._get_graph_device(data)
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=target_device).t()
                
                # Bidirectional connections
                data['eeg', 'projects_to', 'fmri'].edge_index = edge_index
                data['fmri', 'receives_from', 'eeg'].edge_index = edge_index.flip(0)
                
                logger.info(f"Added {len(edge_list)} cross-modal edges via mapping")
        
        # Method 2: Distance-based if positions available
        elif distance_threshold is not None:
            if hasattr(data['eeg'], 'pos') and hasattr(data['fmri'], 'pos'):
                eeg_pos = data['eeg'].pos  # [N_eeg, 3]
                fmri_pos = data['fmri'].pos  # [N_fmri, 3]
                
                # Compute pairwise distances
                dist = torch.cdist(eeg_pos, fmri_pos, p=2)  # [N_eeg, N_fmri]
                
                # Keep edges below threshold
                mask = dist < distance_threshold
                edge_index = mask.nonzero().t()  # [2, E]
                
                if edge_index.shape[1] > 0:
                    # Add edges
                    data['eeg', 'projects_to', 'fmri'].edge_index = edge_index
                    data['fmri', 'receives_from', 'eeg'].edge_index = edge_index.flip(0)
                    
                    # Edge attributes: inverse distance (closer = stronger)
                    edge_weights = 1.0 / (dist[mask] + 1e-6)
                    data['eeg', 'projects_to', 'fmri'].edge_attr = edge_weights.unsqueeze(-1)
                    data['fmri', 'receives_from', 'eeg'].edge_attr = edge_weights.unsqueeze(-1)
                    
                    logger.info(
                        f"Added {edge_index.shape[1]} cross-modal edges via distance "
                        f"(threshold={distance_threshold})"
                    )
        
        return data
    
    def create_simple_cross_modal_edges(
        self,
        merged_data: HeteroData,
        connection_ratio: float = 0.1,
        k_cross_modal: int = 5,
        hrf_lag_tr: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create cross-modal edges for merged heterograph.

        **Correlation-based (preferred)**: When both modalities have temporal
        features, compute the EEG-fMRI temporal correlation matrix
        [N_eeg, N_fmri] and keep the top-``k_cross_modal`` most-correlated
        fMRI ROIs for each EEG channel.  Edge weights = |Pearson r|.

        Neuroscientific motivation: Neurovascular coupling (NVC) means that
        local field potentials / high-frequency EEG power are linearly correlated
        with the BOLD signal in the overlying cortical area (Logothetis 2001;
        Laufs 2008).  Using temporal correlation as edge weight lets the
        model allocate more cross-modal message-passing capacity to EEG
        channels that genuinely drive the BOLD response in each ROI.

        **HRF delay compensation (hrf_lag_tr > 0)**:
        The haemodynamic response function (HRF) introduces a ~4–6 s delay
        between neural activity and the peak BOLD response (Buxton et al. 1998;
        Logothetis 2001).  At TR=2 s this corresponds to 2–3 TRs.  By shifting
        the EEG time-series earlier by ``hrf_lag_tr`` steps in the pooled
        time-grid before computing Pearson correlation, we model:

            EEG(t) → neural activity → BOLD(t + hrf_lag)

        rather than the biologically incorrect simultaneous alignment
        EEG(t) ↔ fMRI(t).  The result is that cross-modal edges connect EEG
        channels whose *past* activity best predicts each fMRI ROI's current
        BOLD — a more faithful representation of neurovascular coupling.

        **Fallback (random)**: Used when T_eeg ≠ T_fmri and temporal
        interpolation is not appropriate (e.g. the modalities come from
        different recording sessions), or when node features are unavailable.
        In this case ``connection_ratio`` controls sparsity and weights are
        uniform (1.0), preserving backward-compatible behaviour.

        Returns both edge_index AND edge_attr so that cross-modal messages
        are treated consistently with intra-modal edges (which carry
        correlation-based edge_attr from build_graph_structure).

        Args:
            merged_data: HeteroData with 'eeg' and 'fmri' nodes.
            connection_ratio: Fraction of possible edges for random fallback.
            k_cross_modal: Top-k fMRI ROIs per EEG channel for correlation-based
                edges.  Default 5 gives ≈5×N_eeg edges — comparable in density
                to the intra-modal k-nearest graphs.
            hrf_lag_tr: HRF delay in fMRI TR units applied to cross-modal
                edge weight computation.  EEG time-series is shifted earlier
                by this many steps (in the pooled fMRI-rate time grid) before
                computing Pearson correlation with fMRI.  Default 0 (no lag,
                backward-compatible).  Recommended value: 2 (≈4 s at TR=2 s).

        Returns:
            (edge_index [2, E], edge_attr [E, 1]) or None if modalities absent.
        """
        if 'eeg' not in merged_data.node_types or 'fmri' not in merged_data.node_types:
            return None

        N_eeg = merged_data['eeg'].num_nodes
        N_fmri = merged_data['fmri'].num_nodes

        # Design intent validation: EEG (electrodes) should have FEWER nodes than
        # fMRI (atlas ROIs).  With Schaefer200: N_fmri=200, N_eeg≈32–64.
        # If N_eeg > N_fmri this almost always means atlas parcellation failed and
        # fMRI fell back to a single-node average — graph convolution on 1 node is
        # meaningless.  Warn clearly so the user can fix atlas loading.
        if N_eeg > N_fmri:
            logger.warning(
                f"⚠️  Design intent: N_eeg ({N_eeg}) > N_fmri ({N_fmri}). "
                f"EEG electrodes should have FEWER nodes than fMRI ROIs. "
                f"This usually means atlas parcellation did not load — fMRI has "
                f"collapsed to {N_fmri} node(s). Check that the atlas file exists "
                f"and nilearn is installed."
            )

        # Use helper method to determine device
        target_device = self._get_graph_device(merged_data)

        # ── Attempt correlation-based cross-modal edges ─────────────────────
        # Neurovascular coupling: EEG channels with high temporal correlation to
        # a fMRI ROI's BOLD signal should receive stronger cross-modal messages.
        # Strategy:
        #   1. Extract EEG [N_eeg, T_eeg] and fMRI [N_fmri, T_fmri] timeseries.
        #   2. Down-sample the higher-rate modality to the other's time grid via
        #      adaptive average pooling (causal, no look-ahead).
        #   3. Compute [N_eeg, N_fmri] absolute Pearson correlation matrix on GPU.
        #   4. Top-k per EEG channel → sparse edge_index with |r| as edge_attr.
        # If anything fails, fall through to the random-connection fallback.
        try:
            eeg_x = merged_data['eeg'].x   # [N_eeg,  T_eeg,  1]
            fmri_x = merged_data['fmri'].x  # [N_fmri, T_fmri, 1]

            if eeg_x is not None and fmri_x is not None:
                T_eeg  = eeg_x.shape[1]
                T_fmri = fmri_x.shape[1]

                # Squeeze channel dim and move to GPU for matrix multiplication
                eeg_ts  = eeg_x.squeeze(-1).to(target_device, dtype=torch.float32)   # [N_eeg,  T_eeg]
                fmri_ts = fmri_x.squeeze(-1).to(target_device, dtype=torch.float32)  # [N_fmri, T_fmri]

                # ── Temporal alignment ──────────────────────────────────────
                # Adaptive average pooling down-samples whichever modality has
                # more time-points.  This is equivalent to computing the mean
                # within each non-overlapping window — a faithful low-pass
                # summary at the coarser modality's time-scale.
                T_target = min(T_eeg, T_fmri)
                if T_eeg != T_target:
                    # EEG has more time-points: pool to fMRI rate
                    eeg_ts = torch.nn.functional.adaptive_avg_pool1d(
                        eeg_ts.unsqueeze(0), T_target
                    ).squeeze(0)           # [N_eeg, T_target]
                if T_fmri != T_target:
                    # fMRI has more time-points: pool to EEG rate (unusual)
                    fmri_ts = torch.nn.functional.adaptive_avg_pool1d(
                        fmri_ts.unsqueeze(0), T_target
                    ).squeeze(0)           # [N_fmri, T_target]

                # ── HRF delay compensation ──────────────────────────────────
                # The haemodynamic response function introduces a ~4–6 s delay
                # between neural activity and the BOLD peak (Buxton 1998).
                # When hrf_lag_tr > 0 we compute:
                #   corr( eeg_ts[:, :-lag], fmri_ts[:, lag:] )
                # i.e. EEG at time t predicts fMRI at time t+lag, which is the
                # biologically correct neurovascular coupling direction.
                # We fall back to no-lag if fewer than _MIN_CORR_POINTS would remain.
                lag = int(hrf_lag_tr)
                if lag > 0 and T_target - lag >= _MIN_CORR_POINTS:
                    eeg_ts_c   = eeg_ts[:, :-lag]   # [N_eeg,  T_target-lag]
                    fmri_ts_c  = fmri_ts[:, lag:]   # [N_fmri, T_target-lag]
                    T_corr = T_target - lag
                    logger.debug(
                        f"HRF lag compensation: correlating EEG[0:{T_target-lag}] "
                        f"with fMRI[{lag}:{T_target}] "
                        f"(hrf_lag_tr={lag}, T_corr={T_corr})"
                    )
                else:
                    eeg_ts_c  = eeg_ts
                    fmri_ts_c = fmri_ts
                    T_corr = T_target

                # ── Pearson correlation matrix [N_eeg, N_fmri] ─────────────
                # Row-wise z-score so that corr[i,j] = <z_eeg_i, z_fmri_j> / T
                eeg_z  = _row_zscore(eeg_ts_c)   # [N_eeg, T_corr]
                fmri_z = _row_zscore(fmri_ts_c)  # [N_fmri, T_corr]

                corr = torch.mm(eeg_z, fmri_z.T) / T_corr  # [N_eeg, N_fmri]
                corr = torch.abs(corr)  # unsigned connectivity (same as intra-modal)

                # ── Top-k per EEG channel ───────────────────────────────────
                k = min(k_cross_modal, N_fmri)
                topk_vals, topk_idx = torch.topk(corr, k, dim=1)  # [N_eeg, k]

                # Build edge_index [2, N_eeg*k]
                src = torch.arange(N_eeg, device=target_device).unsqueeze(1).expand(-1, k).reshape(-1)
                dst = topk_idx.reshape(-1)
                edge_index = torch.stack([src, dst], dim=0)      # [2, N_eeg*k]
                edge_attr  = topk_vals.reshape(-1, 1).clamp(0.0, 1.0)  # [N_eeg*k, 1]

                lag_info = f", HRF lag={lag} TR" if lag > 0 else ""
                logger.info(
                    f"Created {edge_index.shape[1]} correlation-based cross-modal edges "
                    f"(top-{k} per EEG channel, mean |r|={edge_attr.mean().item():.3f}{lag_info})"
                )
                return edge_index, edge_attr

        except Exception as _exc:
            logger.warning(
                f"Correlation-based cross-modal edges failed ({_exc}); "
                f"falling back to random connections."
            )

        # ── Fallback: random connections ───────────────────────────────────
        # Used when temporal alignment is infeasible or node features are absent.
        num_edges = max(1, int(N_eeg * N_fmri * connection_ratio))
        eeg_indices  = torch.randint(0, N_eeg,  (num_edges,), device=target_device)
        fmri_indices = torch.randint(0, N_fmri, (num_edges,), device=target_device)
        edge_index = torch.stack([eeg_indices, fmri_indices], dim=0)
        edge_attr  = torch.ones(num_edges, 1, dtype=torch.float32, device=target_device)
        logger.info(f"Created {num_edges} random cross-modal edges (uniform weight=1.0)")
        return edge_index, edge_attr

    def add_dti_structural_edges(
        self,
        data: HeteroData,
        connectivity_matrix: np.ndarray,
        threshold: Optional[float] = None,
        k_nearest: Optional[int] = None,
    ) -> HeteroData:
        """Add DTI structural connectivity as a new edge type on fMRI nodes.

        DTI tractography provides white-matter structural connectivity (number
        of streamlines or FA-weighted connectivity) between the same atlas ROIs
        used for fMRI.  This adds a second edge type
        ``('fmri', 'structural', 'fmri')`` alongside the functional edges
        ``('fmri', 'connects', 'fmri')``, enabling the ST-GCN encoder to exploit
        both structural and functional connectivity simultaneously.

        Design rationale (why structural edges on fMRI nodes, not a separate DTI
        node type):
        - DTI tractography inherently describes connectivity *between* brain
          regions (already defined by the fMRI atlas parcellation).
        - DTI does not carry its own temporal dynamics — it is a static
          connectivity scaffold, not a time series.
        - Adding a structural edge type is the minimal, correct change: same
          nodes, richer edge set.  A separate ``dti`` node type would require
          its own feature representation and temporal model.

        Interface contract (future DTI layer):
        - ``connectivity_matrix``: any [N_rois, N_rois] non-negative matrix.
          Accepted sources: streamline counts, FA-weighted connectivity,
          log-normalised tract density, etc.
        - The method reuses ``build_graph_structure`` so K-NN + threshold
          filtering is consistent with intra-modal edges.
        - If DTI files are absent, simply do not call this method; the encoder
          handles missing edge types gracefully.

        Args:
            data: HeteroData containing 'fmri' nodes (from map_fmri_to_graph).
            connectivity_matrix: [N_rois, N_rois] DTI connectivity weights.
            threshold: Min connectivity strength to create an edge.
                Defaults to self.threshold_fmri.
            k_nearest: K-nearest structural neighbours per ROI.
                Defaults to self.k_nearest_fmri.

        Returns:
            The same HeteroData with ('fmri', 'structural', 'fmri') edges added.
        """
        if 'fmri' not in data.node_types:
            logger.warning("add_dti_structural_edges: no 'fmri' nodes in graph, skipping")
            return data

        N_fmri = data['fmri'].num_nodes
        if connectivity_matrix.shape[0] != N_fmri or connectivity_matrix.shape[1] != N_fmri:
            # This almost always indicates atlas version mismatch (e.g., DTI matrix was
            # computed with a different parcellation or different software version).
            # Raise an error rather than silently padding with zeros: zero-padded rows
            # would produce phantom structural disconnections that corrupt training.
            # Fix: re-parcellate DTI tractography with the same atlas used for fMRI.
            raise ValueError(
                f"DTI matrix shape {connectivity_matrix.shape} does not match the number of "
                f"fMRI ROIs ({N_fmri}, {N_fmri}). "
                f"Ensure the DTI connectivity matrix was computed with the same atlas "
                f"parcellation as the fMRI graph (e.g., Schaefer200). "
                f"Re-run tractography parcellation to produce a [{N_fmri}, {N_fmri}] matrix."
            )

        edge_index, edge_attr = self.build_graph_structure(
            connectivity_matrix,
            threshold=threshold if threshold is not None else self.threshold_fmri,
            k_nearest=k_nearest if k_nearest is not None else self.k_nearest_fmri,
        )

        data['fmri', 'structural', 'fmri'].edge_index = edge_index
        data['fmri', 'structural', 'fmri'].edge_attr = edge_attr

        logger.info(
            f"Added DTI structural edges: {edge_index.shape[1]} connections "
            f"between {N_fmri} fMRI ROIs ('fmri','structural','fmri')"
        )
        return data
