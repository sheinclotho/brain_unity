"""
Real-time Brain Visualization Server
=====================================

WebSocket server for real-time communication with Unity frontend.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Set, Optional
from pathlib import Path
from datetime import datetime

# Try to import websockets, but make it optional
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets package not available. Install with: pip install websockets")

# Import ModelServer
try:
    from .model_server import ModelServer
    MODEL_SERVER_AVAILABLE = True
except ImportError:
    MODEL_SERVER_AVAILABLE = False
    logging.warning("ModelServer not available")


def _import_brain_state_analyzer():
    """Robustly import BrainStateAnalyzer regardless of the import context."""
    # 1. Package-relative import (normal usage)
    try:
        from .brain_state_analyzer import BrainStateAnalyzer as _BSA
        return _BSA
    except (ImportError, SystemError):
        pass
    # 2. Absolute package import (when the package root is on sys.path)
    try:
        from unity_integration.brain_state_analyzer import BrainStateAnalyzer as _BSA
        return _BSA
    except ImportError:
        pass
    # 3. Direct file import via importlib (standalone / test scripts)
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "brain_state_analyzer",
            str(Path(__file__).parent / "brain_state_analyzer.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod.BrainStateAnalyzer
    except Exception:
        return None


class BrainVisualizationServer:
    """
    WebSocket server for real-time brain visualization.
    
    Provides endpoints for:
    - Getting current brain state
    - Predicting future states
    - Simulating stimulation effects
    - Streaming brain activity
    """
    
    def __init__(
        self,
        model = None,
        exporter = None,
        simulator = None,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8765
    ):
        """
        Initialize server.
        
        Args:
            model: Trained neural network model (legacy, use model_path instead)
            exporter: BrainStateExporter instance
            simulator: StimulationSimulator instance
            model_path: Path to trained model file
            output_dir: Output directory for predictions
            host: Server host
            port: Server port
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required. Install with: pip install websockets")
        
        self.model = model
        self.exporter = exporter
        self.simulator = simulator
        self.host = host
        self.port = port
        
        # Initialize ModelServer if available
        self.model_server = None
        if MODEL_SERVER_AVAILABLE and model_path:
            self.model_server = ModelServer(
                model_path=model_path,
                output_dir=output_dir or "unity_project/brain_data/model_output"
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("✓ ModelServer initialized")
        elif MODEL_SERVER_AVAILABLE and model:
            # Create ModelServer without loading (use existing model)
            self.model_server = ModelServer(
                output_dir=output_dir or "unity_project/brain_data/model_output"
            )
            self.model_server.model = model
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
        
        self.clients: Set = set()

        # Cache for trained PerturbationAnalyzer instances.
        # Key: (cache_file_path_str, n_lags) so the surrogate is only retrained
        # when the source data or lag window changes, not on every infer_ec call.
        self._ec_analyzer_cache: dict = {}

        # Last loaded time series — stored so handle_validate_ec and
        # handle_analyze_brain can access it without re-loading the .pt file.
        self._loaded_time_series: Optional["np.ndarray"] = None  # (T, 200)
        self._loaded_cache_path:  Optional[str]          = None
        # Raw (un-per-frame-normalized) time series for deviation analysis.
        # EEG per-frame normalization loses temporal structure, making z-score
        # deviation maps near-zero.  The raw series preserves real task differences.
        self._loaded_raw_time_series: Optional["np.ndarray"] = None  # (T, 200)
    
    async def register_client(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send welcome message
        welcome = {
            "type": "welcome",
            "message": "Connected to TwinBrain server",
            "version": "1.0"
        }
        await websocket.send(json.dumps(welcome))
    
    async def unregister_client(self, websocket):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if self.clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.clients],
                return_exceptions=True
            )
    
    async def handle_client(self, websocket, path=None):
        """Handle client connection and requests."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    response = await self.process_request(request)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    error = {"type": "error", "message": "Invalid JSON"}
                    await websocket.send(json.dumps(error))
                except Exception as e:
                    self.logger.error(f"Error processing request: {e}")
                    error = {"type": "error", "message": str(e)}
                    await websocket.send(json.dumps(error))
        finally:
            await self.unregister_client(websocket)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process client request and return response."""
        request_type = request.get("type", "unknown")

        # Type aliases: accept legacy and alternative naming conventions
        if request_type in ("get_brain_state",):
            request_type = "get_state"
        elif request_type in ("simulate_stimulation",):
            request_type = "simulate"

        # Log the request
        self.logger.info(f"Processing request: {request_type}")
        
        try:
            if request_type == "get_state":
                return await self.handle_get_state(request)
            
            elif request_type == "predict":
                return await self.handle_predict(request)
            
            elif request_type == "simulate":
                return await self.handle_simulate(request)
            
            elif request_type == "stream_start":
                return await self.handle_stream_start(request)
            
            elif request_type == "stream_stop":
                return {"type": "stream_stopped", "success": True}
            
            elif request_type == "convert_cache":
                return await self.handle_convert_cache(request)

            elif request_type == "load_cache":
                return await self.handle_load_cache(request)

            elif request_type == "infer_ec":
                return await self.handle_infer_ec(request)

            elif request_type == "validate_ec":
                return await self.handle_validate_ec(request)

            elif request_type == "analyze_brain":
                return await self.handle_analyze_brain(request)
            
            else:
                self.logger.warning(f"Unknown request type: {request_type}")
                return {
                    "type": "error",
                    "success": False,
                    "message": f"Unknown request type: {request_type}"
                }
        except Exception as e:
            self.logger.error(f"Error processing {request_type}: {e}", exc_info=True)
            return {
                "type": "error",
                "success": False,
                "message": f"Server error: {str(e)}"
            }
    
    async def handle_get_state(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Return current brain state with flat activity array.

        Uses a 7-network (Schaefer) oscillation model for demo mode so that
        different brain networks show distinct, correlated dynamics rather than
        pure white noise.
        """
        import numpy as np
        import time

        n_regions  = 200
        t          = time.time()

        # Schaefer 7-network approximate frequency/phase profile
        # Visual, Somatomotor, DorsAttn, VentAttn, Limbic, FrontPar, Default
        net_freqs  = [0.012, 0.020, 0.035, 0.028, 0.008, 0.025, 0.010]
        net_phases = [0.0,   1.0,   2.1,   0.7,   3.2,   1.8,   0.4  ]
        net_size   = n_regions // 7

        rng      = np.random.default_rng(int(t * 200) % 65536)
        activity = []
        for i in range(n_regions):
            n  = min(i // net_size, 6)
            v  = 0.36 + 0.22 * np.sin(2 * np.pi * net_freqs[n] * t + net_phases[n] + i * 0.08)
            v += 0.04 * np.sin(2 * np.pi * 0.005 * t + i * 0.25)  # slow global wave
            v += float(rng.normal(0, 0.025))                        # per-region noise
            activity.append(float(np.clip(v, 0, 1)))

        return {
            "type":     "brain_state",
            "activity": activity,       # flat [0,1] array, length = n_regions
            "success":  True,
        }


    
    async def handle_predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle prediction request with automatic JSON export.
        
        Automatically saves prediction results to output_dir/predictions/
        for Unity to auto-load.
        """
        n_steps = request.get("n_steps", 10)
        
        # Input validation
        original_n_steps = n_steps
        if not isinstance(n_steps, int) or n_steps <= 0 or n_steps > 1000:
            self.logger.warning(f"Invalid n_steps: {n_steps}, using default 10")
            n_steps = 10
        
        # Inform client if parameter was adjusted
        parameter_adjusted = (n_steps != original_n_steps)
        
        try:
            # Create timestamped output directory for predictions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = Path(self.model_server.output_dir if self.model_server else "unity_project/brain_data/model_output")
            pred_output_dir = output_base / "predictions" / f"pred_{timestamp}"
            pred_output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Prediction output directory: {pred_output_dir}")
            
            # Use ModelServer if available
            if self.model_server:
                self.logger.info(f"Using ModelServer for prediction ({n_steps} steps)")
                predictions = self.model_server.predict_future(
                    n_steps=n_steps,
                    subject_id="prediction"
                )
                
                # Auto-save each prediction frame as JSON
                # Note: Individual files are required for Unity to load frames separately
                # and support streaming playback. Batching would prevent frame-by-frame access.
                self.logger.info(f"Auto-saving {len(predictions)} prediction frames to {pred_output_dir}")
                for idx, prediction in enumerate(predictions):
                    json_path = pred_output_dir / f"frame_{idx:04d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(prediction, f, indent=2)
                
                # Create sequence index
                index_data = {
                    "type": "prediction_sequence",
                    "timestamp": timestamp,
                    "n_frames": len(predictions),
                    "output_dir": str(pred_output_dir),
                    "files": [f"frame_{i:04d}.json" for i in range(len(predictions))]
                }
                index_path = pred_output_dir / "sequence_index.json"
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, indent=2)
                
                self.logger.info(f"✓ Prediction results auto-saved to: {pred_output_dir}")
                
                return {
                    "type": "prediction",
                    "success": True,
                    "n_steps": n_steps,
                    "predictions": predictions,
                    "saved_to": str(pred_output_dir),
                    "index_file": str(index_path),
                    "auto_saved": True,
                    "parameter_adjusted": parameter_adjusted,
                    "warning": "n_steps was adjusted to valid range" if parameter_adjusted else None
                }
            
            # Fallback to simple prediction generation
            import torch
            import numpy as np
            
            n_regions = 200
            
            # Generate prediction sequence
            predictions = []
            for t in range(n_steps):
                # Generate predicted brain state
                fmri_data = torch.randn(n_regions, 1, 1) * (0.5 + t * 0.05)
                eeg_data = torch.randn(n_regions, 1, 1) * (0.5 + t * 0.05)
                
                brain_activity = {
                    'fmri': fmri_data,
                    'eeg': eeg_data
                }
                
                if self.exporter:
                    brain_state = self.exporter.export_brain_state(
                        brain_activity=brain_activity,
                        time_point=t,
                        time_second=float(t),
                        subject_id="prediction"
                    )
                    predictions.append(brain_state)
                    
                    # Auto-save to file
                    json_path = pred_output_dir / f"frame_{t:04d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(brain_state, f, indent=2)
            
            # Create sequence index
            index_data = {
                "type": "prediction_sequence",
                "timestamp": timestamp,
                "n_frames": len(predictions),
                "output_dir": str(pred_output_dir),
                "files": [f"frame_{i:04d}.json" for i in range(len(predictions))]
            }
            index_path = pred_output_dir / "sequence_index.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            
            self.logger.info(f"✓ Prediction results auto-saved to: {pred_output_dir}")
            
            return {
                "type": "prediction",
                "success": True,
                "n_steps": n_steps,
                "predictions": predictions,
                "saved_to": str(pred_output_dir),
                "index_file": str(index_path),
                "auto_saved": True,
                "parameter_adjusted": parameter_adjusted,
                "warning": "n_steps was adjusted to valid range" if parameter_adjusted else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return {
                "type": "error",
                "message": f"Prediction failed: {str(e)}"
            }
    
    async def handle_simulate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle stimulation simulation request with automatic JSON export.
        
        Automatically saves stimulation results to output_dir/stimulation/
        for Unity to auto-load.

        Accepts both formats:
          - Flat:   {type, target_regions, amplitude, pattern, frequency, duration}
          - Nested: {type, stimulation: {target_regions, amplitude, ...}}
        """
        # Accept both nested (legacy) and flat (web frontend) parameter layouts
        if "stimulation" in request and isinstance(request["stimulation"], dict):
            stimulation = request["stimulation"]
        else:
            stimulation = request  # flat format sent by web frontend
        
        try:
            import torch
            import numpy as np
            
            # Parse stimulation parameters
            target_regions = stimulation.get("target_regions", [])
            amplitude = stimulation.get("amplitude", 0.5)
            pattern = stimulation.get("pattern", "sine")
            frequency = stimulation.get("frequency", 10.0)
            duration = stimulation.get("duration", 20)
            # The frontend sends the currently-displayed frame's activity so the
            # stimulation starts from the user's actual selected brain state.
            initial_state = stimulation.get("initial_state", None)
            
            # Validate target regions
            if not isinstance(target_regions, list) or len(target_regions) == 0:
                return {
                    "type": "error",
                    "success": False,
                    "message": "target_regions must be a non-empty list"
                }
            
            # Validate and clamp values
            amplitude = float(max(0.0, min(10.0, amplitude)))
            frequency = float(max(0.1, min(100.0, frequency)))
            duration = int(max(1, min(1000, duration)))
            
            # Filter valid region IDs
            n_regions = 200
            valid_regions = [r for r in target_regions if isinstance(r, int) and 0 <= r < n_regions]
            
            if len(valid_regions) == 0:
                return {
                    "type": "error",
                    "success": False,
                    "message": f"No valid target regions (must be 0-{n_regions-1})"
                }
            
            if len(valid_regions) < len(target_regions):
                self.logger.warning(f"Filtered {len(target_regions) - len(valid_regions)} invalid region IDs")
                target_regions = valid_regions
            
            # Use ModelServer if available
            if self.model_server:
                # Create timestamped output directory only when ModelServer will actually
                # use it — avoids leaving empty dirs on every demo/surrogate simulation.
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_base = Path(self.model_server.output_dir)
                stim_output_dir = output_base / "stimulation" / f"stim_{timestamp}"
                stim_output_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Stimulation output directory: {stim_output_dir}")
                self.logger.info(f"Using ModelServer for stimulation simulation")
                # Convert the user's initial_state list to a tensor so ModelServer
                # starts from the actual brain state rather than random noise.
                init_tensor = None
                if initial_state is not None:
                    arr = np.array(initial_state[:n_regions], dtype=np.float32)
                    init_tensor = torch.tensor(arr).unsqueeze(-1)  # (n_regions, 1)
                responses = self.model_server.simulate_stimulation(
                    target_regions=target_regions,
                    amplitude=amplitude,
                    pattern=pattern,
                    frequency=frequency,
                    duration=duration,
                    initial_state=init_tensor,
                    subject_id="stimulation"
                )
                
                # Auto-save each frame as JSON
                # Note: Individual files are required for Unity's frame-by-frame loading
                # and to support progressive playback during long sequences.
                self.logger.info(f"Auto-saving {len(responses)} stimulation frames to {stim_output_dir}")
                for idx, response in enumerate(responses):
                    json_path = stim_output_dir / f"frame_{idx:04d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(response, f, indent=2)
                
                # Create sequence index for Unity auto-loading
                index_data = {
                    "type": "stimulation_sequence",
                    "timestamp": timestamp,
                    "stimulation_params": {
                        "target_regions": target_regions,
                        "amplitude": amplitude,
                        "pattern": pattern,
                        "frequency": frequency,
                        "duration": duration
                    },
                    "n_frames": len(responses),
                    "output_dir": str(stim_output_dir),
                    "files": [f"frame_{i:04d}.json" for i in range(len(responses))]
                }
                index_path = stim_output_dir / "sequence_index.json"
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, indent=2)
                
                self.logger.info(f"✓ Stimulation results auto-saved to: {stim_output_dir}")
                
                frames = [{"activity": self._extract_activity_array(r)} for r in responses]
                # Counterfactual (null stimulation) via WC dynamics from same initial state
                cf_frames = self._demo_simulate(
                    target_regions=target_regions, amplitude=0.0,
                    pattern=pattern, frequency=frequency,
                    duration=duration, n_regions=n_regions,
                    initial_state=initial_state,
                )
                return {
                    "type": "simulation_result",
                    "success": True,
                    "n_frames": len(frames),
                    "frames": frames,
                    "counterfactual_frames": cf_frames,
                    "modality": "simulation",
                    "start_frame": 0,
                    "saved_to": str(stim_output_dir),
                    "index_file": str(index_path),
                }

            # ── Data-driven surrogate path (requires prior EC inference) ──────
            # If the user has already run "推断有效连接", a trained MLP surrogate is
            # available in _ec_analyzer_cache.  Use predict_trajectory to produce a
            # REAL data-driven forecast instead of generic Wilson-Cowan dynamics.
            surrogate_frames = None
            if initial_state is not None and self._ec_analyzer_cache:
                analyzer = list(self._ec_analyzer_cache.values())[-1]
                if (analyzer._surrogate is not None
                        and analyzer._ts_mean is not None
                        and analyzer.n_regions == n_regions):
                    try:
                        # Compute Gaussian spatial spread weights (same as _demo_simulate)
                        def _bp():
                            daz = 2 * np.pi * (2 - (1 + np.sqrt(5)) / 2)
                            pos = []
                            for h in range(2):
                                sign = -1 if h == 0 else 1
                                for i in range(100):
                                    t_ = (i + 0.5) / 100.0
                                    el = 1.0 - 1.85 * t_
                                    r  = np.sqrt(max(0.0, 1 - el * el))
                                    az = daz * i
                                    lat = abs(r * np.cos(az)) * 0.85 + 0.15
                                    bulge = 9 * np.exp(-((el + 0.22) ** 2) * 5)
                                    pos.append([sign * (lat * 55 + bulge + 9),
                                                el * 63 - 4, r * np.sin(az) * 76 - 8])
                            return np.array(pos, dtype=np.float32)

                        _pos = _bp()
                        sw = np.zeros(n_regions, dtype=np.float32)
                        for tid in target_regions:
                            if 0 <= tid < n_regions:
                                d = np.linalg.norm(_pos - _pos[tid], axis=1)
                                sw += np.exp(-(d ** 2) / (2 * 30.0 ** 2))
                        if sw.max() > 0:
                            sw /= sw.max()

                        PRE_s  = 10
                        DUR_s  = min(int(duration), 60)
                        POST_s = 10

                        def _stim_amp_s(k: int) -> float:
                            # (k+0.5)/DUR_s: centers each frame in its time slot so
                            # the bell envelope is never evaluated at exactly 0 or 1,
                            # ensuring non-zero amplitude even when DUR_s == 1 or 2.
                            progress = (k + 0.5) / max(DUR_s, 1)
                            if pattern == "sine":
                                slow_c = min(frequency / 10.0, 3.0)
                                return amplitude * (np.sin(np.pi * progress)
                                                    + 0.20 * np.sin(2 * np.pi * slow_c * progress))
                            elif pattern == "pulse":
                                return amplitude * np.exp(-k * 0.12)
                            elif pattern == "ramp":
                                return amplitude * progress
                            else:
                                return amplitude * min(1.0, k / max(DUR_s * 0.15, 1.0))

                        # Unified stim_fn covers all three phases:
                        #   k < PRE_s            → pre-stim (surrogate runs, no external input)
                        #   PRE_s ≤ k < PRE_s+DUR_s → stimulation active
                        #   k ≥ PRE_s+DUR_s      → post-stim recovery (no external input)
                        # Using a single call with n_warmup=0 (lag window warms up during
                        # the pre-stim region where stim_fn returns 0), so the pre-stim
                        # frames show genuine baseline dynamics, not static copies.
                        def _stim_fn_full(k: int) -> float:
                            if k < PRE_s:
                                return 0.0
                            stim_k = k - PRE_s
                            if stim_k < DUR_s:
                                return _stim_amp_s(stim_k)
                            return 0.0

                        init_arr = np.array(initial_state[:n_regions], dtype=np.float32)

                        surrogate_frames = analyzer.predict_trajectory(
                            initial_state=init_arr,
                            stim_weights=sw,
                            stim_fn=_stim_fn_full,
                            n_steps=PRE_s + DUR_s + POST_s,
                        )

                        # Counterfactual: same initial state, zero stimulation.
                        # Because predict_trajectory is deterministic (no noise), the
                        # pre-stim frames will be identical between stimulated and
                        # counterfactual; they diverge from frame PRE_s onwards.
                        surrogate_cf = analyzer.predict_trajectory(
                            initial_state=init_arr,
                            stim_weights=sw,
                            stim_fn=lambda k: 0.0,
                            n_steps=PRE_s + DUR_s + POST_s,
                        )
                        self.logger.info(
                            f"使用已训练代理模型预测刺激轨迹 ({len(surrogate_frames)} 帧)"
                        )
                    except Exception as _exc:
                        self.logger.warning(
                            f"代理预测失败，回退到 Wilson-Cowan: {_exc}"
                        )
                        surrogate_frames = None
                        surrogate_cf = None

            if surrogate_frames is not None:
                return {
                    "type": "simulation_result",
                    "success": True,
                    "n_frames": len(surrogate_frames),
                    "frames": surrogate_frames,
                    "counterfactual_frames": surrogate_cf,
                    "modality": "simulation",
                    "method": "surrogate_mlp",
                    "start_frame": PRE_s,
                }

            # Final fallback: Wilson-Cowan recurrent dynamics (no trained model needed).
            # Generate stimulated trajectory, then null (amplitude=0) counterfactual.
            # Both calls re-seed rng=np.random.default_rng(0) so noise is identical,
            # making the comparison between stim and null scientifically clean.
            frames = self._demo_simulate(
                target_regions=target_regions,
                amplitude=amplitude,
                pattern=pattern,
                frequency=frequency,
                duration=duration,
                n_regions=n_regions,
                initial_state=initial_state,
            )
            # Counterfactual: same dynamics but with amplitude=0
            cf_frames = self._demo_simulate(
                target_regions=target_regions,
                amplitude=0.0,
                pattern=pattern,
                frequency=frequency,
                duration=duration,
                n_regions=n_regions,
                initial_state=initial_state,
            )
            return {
                "type": "simulation_result",
                "success": True,
                "n_frames": len(frames),
                "frames": frames,
                "counterfactual_frames": cf_frames,
                "modality": "simulation",
                "method": "wilson_cowan",
                "start_frame": 10,
            }
                
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            return {
                "type": "error",
                "message": f"Simulation failed: {str(e)}"
            }
    
    # ------------------------------------------------------------------ #
    #  Perturbation-based Effective Connectivity inference                 #
    # ------------------------------------------------------------------ #

    async def handle_infer_ec(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Infer effective connectivity using NPI-inspired perturbation analysis.

        Request fields:
          method  (str, optional): "jacobian" (default) | "perturbation" | "demo"
          n_lags  (int, optional): lag window for surrogate model (default 5)
          path    (str, optional): .pt cache file to extract time series from;
                                   if omitted uses the last loaded cache or demo data.

        Response:
          {type: "ec_result",
           method: str,
           ec_flat: [200*200 floats],     -- row-major EC matrix
           top_sources: [int*10],         -- most influential source regions
           top_targets: [int*10],         -- most receptive target regions
           activity_delta: [200 floats],  -- predicted activity change from top 3 sources
           fit_quality: {train_mse, val_mse, overfit_ratio, reliable, n_epochs},
           success: true}
        """
        import numpy as np

        method = request.get("method", "jacobian")
        n_lags = int(request.get("n_lags", 5))

        try:
            # Try relative import first (normal package usage); fall back to
            # absolute import when the module is loaded as a standalone script
            # in test environments where the parent package stub may not support
            # relative imports.
            try:
                from .perturbation_analyzer import PerturbationAnalyzer
            except (ImportError, SystemError):
                from unity_integration.perturbation_analyzer import PerturbationAnalyzer
        except ImportError:
            return {"type": "error", "message": "PerturbationAnalyzer 模块未找到"}

        # ── Demo mode (no data needed) ─────────────────────────────────────
        if method == "demo":
            analyzer = PerturbationAnalyzer(n_regions=200, n_lags=n_lags)
            ec = analyzer.infer_ec_demo()
            result = analyzer.ec_to_dict(ec)
            top_src = result["top_sources"]
            result["activity_delta"] = analyzer.predict_activity_delta(
                top_src[:3], amplitude=0.5, ec_matrix=ec
            ).tolist()
            result["method"]  = "demo"
            result["success"] = True
            result["type"]    = "ec_result"
            self.logger.info("EC 推断完成 (演示模式)")
            return result

        # ── Data-driven modes: need time series ────────────────────────────
        raw_path   = request.get("path")
        cache_path = Path(raw_path) if raw_path else self._find_cache_file()

        time_series = None

        if cache_path and cache_path.exists():
            try:
                import torch
                # weights_only=False is required because cache files contain
                # HeteroData objects (torch_geometric custom classes) that
                # cannot be deserialized with weights_only=True.
                # Mitigated by only loading files from the local project tree.
                data_pt = torch.load(str(cache_path), map_location="cpu",
                                     weights_only=False)
                frames = self._extract_time_series(data_pt)
                if frames:
                    time_series = np.array(
                        [f["activity"] for f in frames], dtype=np.float32
                    )  # (T, 200)
                    self.logger.info(
                        f"EC 推断: 从缓存加载 {time_series.shape[0]} 帧 × 200 区域"
                    )
            except Exception as exc:
                self.logger.warning(f"缓存加载失败，使用演示数据: {exc}")

        if time_series is None or len(time_series) < n_lags + 10:
            self.logger.info("EC 推断: 生成合成振荡时序数据（演示用）")
            time_series = self._generate_demo_time_series(300)
            cache_path = None   # no file backing; don't cache by path

        # ── Reuse cached surrogate when possible ───────────────────────────
        cache_key = (str(cache_path) if cache_path else "__synthetic__", n_lags)
        analyzer  = self._ec_analyzer_cache.get(cache_key)
        need_train = analyzer is None

        if need_train:
            analyzer = PerturbationAnalyzer(n_regions=200, n_lags=n_lags)
            self.logger.info(f"训练 MLP 代理模型: {time_series.shape} …")
            analyzer.fit_surrogate(
                time_series,
                n_lags=n_lags,
                # num_epochs is a ceiling; early stopping (patience=12) will
                # terminate training before this limit when validation MSE
                # stops improving, so a higher cap costs nothing in practice
                # and gives the model more time to converge on larger datasets.
                num_epochs=80,
                batch_size=64,
                lr=1e-3,
            )
            # Cache the trained analyzer so subsequent infer_ec requests (e.g.
            # switching method jacobian→perturbation) skip retraining.
            self._ec_analyzer_cache[cache_key] = analyzer
        else:
            self.logger.info("EC 推断: 复用已缓存代理模型（跳过训练）")

        # ── Infer EC ───────────────────────────────────────────────────────
        if method == "perturbation":
            ec = analyzer.infer_ec_perturbation(pert_strength=0.05)
        else:
            ec = analyzer.infer_ec_jacobian(
                n_samples=min(100, len(time_series) - n_lags)
            )

        result = analyzer.ec_to_dict(ec)
        top_src = result["top_sources"]
        # Use top-3 sources for a more representative activity spread prediction
        result["activity_delta"] = analyzer.predict_activity_delta(
            top_src[:3], amplitude=0.5, ec_matrix=ec
        ).tolist()
        result["method"]  = method
        result["success"] = True
        result["type"]    = "ec_result"
        self.logger.info(
            f"EC 推断完成 ({method}), 影响力最强区域: {top_src[:5]}, "
            f"可靠: {result.get('fit_quality', {}).get('reliable', 'N/A')}"
        )
        return result

    # ------------------------------------------------------------------ #
    #  EC Validation                                                        #
    # ------------------------------------------------------------------ #

    async def handle_validate_ec(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate EC reliability using three complementary checks.

        Checks (all use the EXISTING trained surrogate — no retraining):
          half_split  – EC consistency between first-half vs second-half of
                        training samples (higher Pearson r = more reliable)
          distance    – |EC[i,j]| vs anatomical distance correlation
                        (expected negative: nearby regions interact more strongly)
          fc_compare  – Pearson r between |EC| and FC (correlation matrix);
                        high similarity means EC is redundant with simpler FC.
                        Well-calibrated EC should add information beyond FC.

        Request: { type: "validate_ec" }
        Response: { type: "ec_validation_result", success: True, results: {...} }
        """
        import numpy as np

        BrainStateAnalyzer = _import_brain_state_analyzer()
        if BrainStateAnalyzer is None:
            return {"type": "error", "message": "BrainStateAnalyzer 模块未找到"}

        # Require a trained EC analyzer
        if not self._ec_analyzer_cache:
            return {
                "type": "error",
                "success": False,
                "message": "尚未推断 EC，请先点击「推断有效连接」",
            }

        analyzer = list(self._ec_analyzer_cache.values())[-1]
        if analyzer._surrogate is None or analyzer._input_X is None:
            return {
                "type": "error",
                "success": False,
                "message": "代理模型未训练，请先推断 EC",
            }

        results = {}

        # ── 1. Half-split reliability ──────────────────────────────────────
        try:
            r_hs, detail_hs = BrainStateAnalyzer.ec_half_split_reliability(
                surrogate=analyzer._surrogate,
                input_X=analyzer._input_X,
                n_regions=analyzer.n_regions,
            )
            results["half_split"] = detail_hs
        except Exception as exc:
            results["half_split"] = {"error": str(exc)}

        # ── 2. EC vs anatomical distance ───────────────────────────────────
        if analyzer._last_ec is not None:
            try:
                dist_result = BrainStateAnalyzer.ec_vs_distance_correlation(
                    ec_matrix=analyzer._last_ec,
                )
                results["distance"] = dist_result
            except Exception as exc:
                results["distance"] = {"error": str(exc)}

        # ── 3. EC vs FC (functional connectivity from training data) ───────
        if analyzer._last_ec is not None and self._loaded_time_series is not None:
            try:
                ts = self._loaded_time_series          # (T, 200)
                # Pearson FC (correlation between time courses of all region pairs)
                fc = np.corrcoef(ts.T)                 # (200, 200)
                np.fill_diagonal(fc, 0.0)
                ec_abs = np.abs(analyzer._last_ec)
                np.fill_diagonal(ec_abs, 0.0)
                f_ec = ec_abs.flatten()
                f_fc = np.abs(fc.flatten())
                r_ef = float(np.corrcoef(f_ec, f_fc)[0, 1]) if f_ec.std() > 0 else 0.0
                interp = (
                    "EC ≈ FC（因果关系与相关性高度重叠，EC 的额外信息有限）" if r_ef > 0.7
                    else ("EC 与 FC 中度相关（因果关系略超过单纯相关）" if r_ef > 0.4
                    else "EC 与 FC 差异明显（EC 提供了超出相关分析的因果信息）")
                )
                results["fc_vs_ec"] = {
                    "ec_fc_pearson_r": round(r_ef, 3),
                    "interpretation":  interp,
                }
            except Exception as exc:
                results["fc_vs_ec"] = {"error": str(exc)}

        self.logger.info(f"EC 验证完成: {list(results.keys())}")
        return {
            "type":    "ec_validation_result",
            "success": True,
            "results": results,
        }

    # ------------------------------------------------------------------ #
    #  Brain State Analysis (label-free)                                   #
    # ------------------------------------------------------------------ #

    async def handle_analyze_brain(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Label-free brain state analysis: deviation map or graph metrics.

        Methods:
          deviation     – split loaded time series into two halves, compute
                          per-region z-score deviation map (early vs late window)
          graph_metrics – compute EC-based hub scores and overlay on 3D brain

        Request: { type: "analyze_brain",
                   method: "deviation" | "graph_metrics",
                   window1_start: int, window1_end: int,  (optional)
                   window2_start: int, window2_end: int   (optional)
                 }
        Response: { type: "brain_analysis_result",
                    method: str,
                    activity: [200 floats],   # overlay for 3D view
                    summary: {…},
                    regions_of_interest: [int],
                    success: true
                  }
        """
        import numpy as np

        BrainStateAnalyzer = _import_brain_state_analyzer()
        if BrainStateAnalyzer is None:
            return {"type": "error", "message": "BrainStateAnalyzer 模块未找到"}

        method = request.get("method", "deviation")

        # ── graph_metrics: requires EC matrix ─────────────────────────────
        if method == "graph_metrics":
            if not self._ec_analyzer_cache:
                return {
                    "type": "error",
                    "success": False,
                    "message": "请先推断有效连接（EC），再使用图论分析",
                }
            analyzer = list(self._ec_analyzer_cache.values())[-1]
            if analyzer._last_ec is None:
                return {
                    "type": "error",
                    "success": False,
                    "message": "EC 矩阵不可用，请先推断有效连接",
                }
            try:
                metrics = BrainStateAnalyzer.compute_graph_metrics(analyzer._last_ec)
                hub_scores = np.array(metrics["hub_scores"], dtype=np.float32)
                hs_max = hub_scores.max()
                overlay = (hub_scores / hs_max).tolist() if hs_max > 0 else hub_scores.tolist()
                return {
                    "type":    "brain_analysis_result",
                    "method":  "graph_metrics",
                    "activity": overlay,
                    "summary": {
                        "global_efficiency": metrics["global_eff"],
                        "density":           metrics["density"],
                        "top_hubs":          metrics["top_hubs"],
                        "interpretation":    (
                            f"全脑效率 {metrics['global_eff']:.3f}，"
                            f"连接密度 {metrics['density']:.3f}。"
                            f"最强枢纽区域（Hub）: {[r+1 for r in metrics['top_hubs'][:5]]}"
                        ),
                    },
                    "regions_of_interest": metrics["top_hubs"][:10],
                    "success": True,
                }
            except Exception as exc:
                return {"type": "error", "message": f"图论分析失败: {exc}"}

        # ── deviation: requires loaded time series ─────────────────────────
        if self._loaded_time_series is None:
            return {
                "type": "error",
                "success": False,
                "message": "请先加载 .pt 文件，再使用偏差分析",
            }

        # Prefer raw (un-normalised) time series so that per-frame-normalised EEG
        # data retains genuine between-task amplitude differences.  Fall back to
        # the normalised series if no raw data was stored (e.g. fMRI global-norm).
        ts = (
            self._loaded_raw_time_series
            if self._loaded_raw_time_series is not None
            else self._loaded_time_series
        )   # (T, 200)
        T  = len(ts)
        if T < 6:
            return {"type": "error", "success": False,
                    "message": f"时序数据太短（{T} 帧），至少需要 6 帧"}

        # Default window split: caller can override
        w1s = int(request.get("window1_start", 0))
        w1e = int(request.get("window1_end",   T // 2))
        w2s = int(request.get("window2_start", T // 2))
        w2e = int(request.get("window2_end",   T))

        w1s = max(0, min(w1s, T - 1));  w1e = max(w1s + 1, min(w1e, T))
        w2s = max(0, min(w2s, T - 1));  w2e = max(w2s + 1, min(w2e, T))

        ref_ts  = ts[w1s:w1e]
        test_ts = ts[w2s:w2e]

        try:
            overlay, summary = BrainStateAnalyzer.compute_deviation_map(ref_ts, test_ts)
            summary["window1"] = f"帧 {w1s}–{w1e}"
            summary["window2"] = f"帧 {w2s}–{w2e}"
            summary["total_frames"] = T
            self.logger.info(
                f"偏差分析完成: 均值z={summary['mean_z_score']}, "
                f"最大z={summary['max_z_score']}, "
                f"异常区域数={summary['n_outliers_2std']}"
            )
            return {
                "type":    "brain_analysis_result",
                "method":  "deviation",
                "activity": overlay.tolist(),
                "summary": summary,
                "regions_of_interest": summary["top_regions"],
                "success": True,
            }
        except Exception as exc:
            return {"type": "error", "message": f"偏差分析失败: {exc}"}

    def _generate_demo_time_series(self, T: int = 300) -> "np.ndarray":
        """Generate synthetic 200-region time series using 7-network oscillators."""
        import numpy as np
        n = 200
        net_freqs  = [0.012, 0.020, 0.035, 0.028, 0.008, 0.025, 0.010]
        net_phases = [0.0,   1.0,   2.1,   0.7,   3.2,   1.8,   0.4  ]
        net_size   = n // 7
        rng        = np.random.default_rng(0)
        ts         = np.zeros((T, n), dtype=np.float32)
        for t in range(T):
            for i in range(n):
                k  = min(i // net_size, 6)
                v  = 0.36 + 0.22 * np.sin(net_freqs[k] * t + net_phases[k] + i * 0.08)
                v += 0.04 * np.sin(0.003 * t + i * 0.25)
                v += float(rng.normal(0, 0.025))
                ts[t, i] = float(np.clip(v, 0.0, 1.0))
        return ts

    # ------------------------------------------------------------------ #
    #  Demo stimulation simulation (no trained model required)             #
    # ------------------------------------------------------------------ #

    def _demo_simulate(
        self,
        target_regions: list,
        amplitude: float,
        pattern: str,
        frequency: float,
        duration: int,
        n_regions: int = 200,
        initial_state: list = None,
    ) -> list:
        """Generate a realistic stimulation animation without a trained model.

        Returns a list of {"activity": [200 floats]} frames showing:
          1. Pre-stim baseline  (10 frames)
          2. Stimulation active (``duration`` frames, capped at 60)
          3. Post-stim recovery (10 frames)

        When ``initial_state`` is provided (the user's selected time point
        from loaded data), it is used as the starting brain state so the
        stimulation is clearly anchored to the user's actual data rather than
        a freshly generated synthetic baseline.  When omitted, the 7-network
        sinusoidal model from ``handle_get_state`` is used as a fallback.

        Spatial spread is approximated via a Gaussian kernel over the
        Fibonacci-sphere positions baked into the visualisation (same formula
        as ``makeBrainPositions`` in app.js).
        """
        import numpy as np
        import time

        # Gaussian spatial spread sigma (mm). ~30 mm ≈ typical tDCS/TMS spread
        # radius observed in neuroimaging studies; keeps neighbouring regions
        # within the same cortical network correlated.
        _SPREAD_SIGMA_MM = 30.0
        # Cap stimulation frames to keep the WebSocket payload manageable.
        _MAX_STIM_FRAMES = 60

        # ── Reproduce the brain-region positions from app.js ──────────────
        def _brain_positions():
            daz = 2 * np.pi * (2 - (1 + np.sqrt(5)) / 2)
            pos = []
            for h in range(2):
                sign = -1 if h == 0 else 1
                for i in range(100):
                    t_ = (i + 0.5) / 100.0
                    el = 1.0 - 1.85 * t_
                    r  = np.sqrt(max(0.0, 1 - el * el))
                    az = daz * i
                    ux = r * np.cos(az)
                    uz = r * np.sin(az)
                    lat = abs(ux) * 0.85 + 0.15
                    bulge = 9 * np.exp(-((el + 0.22) ** 2) * 5)
                    pos.append([
                        sign * (lat * 55 + bulge + 9),
                        el * 63 - 4,
                        uz * 76 - 8,
                    ])
            return np.array(pos, dtype=np.float32)

        positions = _brain_positions()  # (200, 3)

        # ── Pre-compute Gaussian spatial-spread weights ────────────────────
        spread_weights = np.zeros(n_regions, dtype=np.float32)
        for tid in target_regions:
            if 0 <= tid < n_regions:
                d = np.linalg.norm(positions - positions[tid], axis=1)
                spread_weights += np.exp(-(d ** 2) / (2 * _SPREAD_SIGMA_MM ** 2))
        # Normalise: peak = amplitude at target, falls off with distance
        if spread_weights.max() > 0:
            spread_weights /= spread_weights.max()

        # ── 7-network baseline oscillation (matches app.js demoUpdate) ────
        net_freqs  = [0.012, 0.020, 0.035, 0.028, 0.008, 0.025, 0.010]
        net_phases = [0.0,   1.0,   2.1,   0.7,   3.2,   1.8,   0.4  ]
        net_size   = n_regions // 7
        t0         = time.time()

        def _baseline(tick: float) -> np.ndarray:
            act = np.empty(n_regions, dtype=np.float32)
            for i in range(n_regions):
                n_ = min(i // net_size, 6)
                v  = 0.36 + 0.22 * np.sin(net_freqs[n_] * tick + net_phases[n_] + i * 0.08)
                v += 0.04 * np.sin(0.003 * tick + i * 0.25)
                act[i] = float(np.clip(v, 0.0, 1.0))
            return act

        PRE  = 10
        DUR  = min(int(duration), _MAX_STIM_FRAMES)

        # ── Stimulation pattern function ───────────────────────────────────
        # NOTE: We model the *neural response envelope*, not the raw electrical
        # waveform.  At 10 fps the Nyquist limit is 5 Hz, so sampling a 10 Hz
        # sine wave at integer frame indices yields sin(2πn)=0 for every frame
        # (zero effect visible to the user).  Instead we show the slow-timescale
        # change in cortical excitability that is the brain's actual response to
        # sustained stimulation — which is the quantity of clinical interest.
        def _stim_amp(relative_t: int) -> float:
            # Center each step in its time slot so the bell envelope never evaluates
            # at exactly 0 or 1 — this prevents zero-amplitude frames for small DUR.
            # Formula: (k + 0.5) / DUR  ∈ (0, 1) exclusive, matching the NPI convention
            # of a small perturbation applied mid-interval.
            progress = (relative_t + 0.5) / max(DUR, 1)
            if pattern == "sine":
                slow_cycles = min(frequency / 10.0, 3.0)   # ≤ 3 visible cycles
                osc = 0.20 * np.sin(2 * np.pi * slow_cycles * progress)
                return amplitude * (np.sin(np.pi * progress) + osc)
            elif pattern == "pulse":
                return amplitude * np.exp(-relative_t * 0.12)
            elif pattern == "ramp":
                return amplitude * progress
            else:  # constant / unknown — smooth ramp-up to avoid sudden color jump
                return amplitude * min(1.0, relative_t / max(DUR * 0.15, 1.0))
        # ── Connectivity matrix for spatial propagation of deviations ────────
        # Used only to spread *deviations from baseline* to neighbouring regions.
        # L1-normalised so propagation stays bounded.
        _CONN_SIGMA_MM = _SPREAD_SIGMA_MM * 1.5
        D = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        W = np.exp(-(D ** 2) / (2 * _CONN_SIGMA_MM ** 2))
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True)
        W = np.where(row_sums > 0, W / row_sums, W).astype(np.float32)

        # Fixed seed so the noise is reproducible (same run → same animation).
        rng = np.random.default_rng(0)

        # ── Determine baseline (init_arr) for both simulation paths ──────────
        if initial_state is not None and len(initial_state) >= n_regions:
            if len(initial_state) > n_regions:
                self.logger.warning(
                    f"initial_state has {len(initial_state)} elements but n_regions={n_regions}; "
                    "truncating — check frontend/backend region count mismatch"
                )
            init_arr = np.array(initial_state[:n_regions], dtype=np.float32)
        else:
            # Fallback: use the sinusoidal 7-network baseline at PRE-1 as reference
            init_arr = _baseline(t0 * 200 + (PRE - 1) * 4)

        def _wc_step(state: np.ndarray, stim_in: np.ndarray) -> np.ndarray:
            """Deviation-based leaky integrator with local spatial propagation.

            Fixes the original model's 0.7-attractor bug: the previous formula
            ``tanh(state + stim + net*0.25)`` had a non-zero fixed point at ~0.7
            so ALL regions converged to yellow regardless of stimulation.

            This model uses init_arr as the stable resting equilibrium:
              - deviation = state − init_arr  (0 when at rest)
              - net_dev   = W @ deviation      (only deviations propagate, not
                                                absolute activity; prevents
                                                global saturation)
              - delta     = tanh(stim_in*2.0 + net_dev*0.35) * 0.04
              - leak      = deviation * 0.10   (returns to init_arr in ~10 steps)

            Without stimulation: deviation stays at 0 → state stable at init_arr.
            With stimulation:    target regions rise clearly above init_arr;
                                 leak prevents runaway saturation.
            """
            deviation = state - init_arr
            net_dev   = W @ deviation
            delta     = np.tanh(stim_in * 2.0 + net_dev * 0.35) * 0.04
            leak      = deviation * 0.10
            noise     = rng.standard_normal(n_regions).astype(np.float32) * 0.008
            return np.clip(state + delta - leak + noise, 0.0, 1.0)

        _NO_STIM = np.zeros(n_regions, dtype=np.float32)
        POST  = 10
        frames = []

        if initial_state is not None and len(initial_state) >= n_regions:
            # 1. Pre-stim: WC evolution from initial state with no stimulation.
            # Using _wc_step (not static copies) so frames 0..PRE-1 evolve
            # naturally — the same continuous-dynamics convention used by the
            # surrogate-MLP path.  The fixed seed ensures the pre-stim noise is
            # reproducible across the stimulated and counterfactual trajectories.
            current = init_arr.copy()
            for _ in range(PRE):
                current = _wc_step(current, _NO_STIM)
                frames.append({"activity": current.tolist()})

            # 2. Stimulation active: deviation-based WC dynamics from end of
            # pre-stim phase (continuous trajectory, not restarted from init_arr).
            for k in range(DUR):
                current = _wc_step(current, _stim_amp(k) * spread_weights)
                frames.append({"activity": current.tolist()})

            # 3. Post-stim: continued dynamics with no external input.
            #    Leak pulls state back toward init_arr naturally.
            for _ in range(POST):
                current = _wc_step(current, _NO_STIM)
                frames.append({"activity": current.tolist()})

        else:
            # Fallback: no data loaded — start from sinusoidal 7-network baseline.
            # 1. Pre-stim baseline (sinusoidal)
            for k in range(PRE):
                frames.append({"activity": _baseline(t0 * 200 + k * 4).tolist()})

            # 2. Stimulation active: deviation-based WC from last pre-stim frame.
            current = np.array(frames[-1]["activity"], dtype=np.float32)
            for k in range(DUR):
                current = _wc_step(current, _stim_amp(k) * spread_weights)
                frames.append({"activity": current.tolist()})

            # 3. Post-stim: return to baseline
            for _ in range(POST):
                current = _wc_step(current, _NO_STIM)
                frames.append({"activity": current.tolist()})

        return frames

    async def handle_convert_cache(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cache to JSON conversion request.
        
        Request format:
        {
            "type": "convert_cache",
            "cache_dir": "/path/to/cache",
            "output_dir": "/path/to/output"
        }
        """
        try:
            cache_dir = request.get("cache_dir")
            output_dir = request.get("output_dir")
            
            if not cache_dir or not output_dir:
                return {
                    "type": "error",
                    "message": "cache_dir and output_dir are required"
                }
            
            cache_path = Path(cache_dir)
            output_path = Path(output_dir)
            
            if not cache_path.exists():
                return {
                    "type": "error",
                    "message": f"Cache directory does not exist: {cache_dir}"
                }
            
            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find cache files (only .pt/.pth files are the actual cache format)
            import glob
            cache_files = []
            for ext in ['*.pt', '*.pth']:
                cache_files.extend(glob.glob(str(cache_path / ext)))
            
            if not cache_files:
                return {
                    "type": "error",
                    "message": f"No cache files found in {cache_dir}"
                }
            
            self.logger.info(f"Found {len(cache_files)} cache files to convert")
            
            # Process each cache file
            converted_count = 0
            errors = []
            
            for cache_file in cache_files:
                try:
                    # Load cache file
                    import torch
                    import numpy as np
                    
                    file_path = Path(cache_file)
                    ext = file_path.suffix
                    
                    # Only .pt files are supported (the actual cache format)
                    if ext not in ['.pt', '.pth']:
                        self.logger.warning(f"Skipping unsupported file format: {file_path.name}")
                        continue
                    
                    self.logger.info(f"Loading cache file: {file_path.name}")
                    data = torch.load(file_path, map_location='cpu', weights_only=False)
                    
                    # Determine cache file type and convert accordingly
                    # Cache files are: eeg_data.pt or hetero_graphs.pt
                    # Both are Dict[str, ...] where keys are task names
                    
                    if 'eeg_data' in file_path.stem:
                        # eeg_data.pt format: Dict[task_name, Dict["on"|"off", HeteroData]]
                        self.logger.info(f"Processing EEG data cache: {file_path.name}")
                        converted = self._convert_eeg_data_cache(data, output_path, file_path.stem)
                        converted_count += converted
                        
                    elif 'hetero_graphs' in file_path.stem:
                        # hetero_graphs.pt format: Dict[task_name, List[HeteroData]]
                        self.logger.info(f"Processing hetero graphs cache: {file_path.name}")
                        converted = self._convert_hetero_graphs_cache(data, output_path, file_path.stem)
                        converted_count += converted
                        
                    else:
                        # Unknown format - try to extract data generically
                        self.logger.warning(f"Unknown cache format for {file_path.name}, attempting generic conversion")
                        if isinstance(data, dict):
                            for task_name, task_data in data.items():
                                try:
                                    brain_activity = self._extract_brain_activity_from_hetero(task_data)
                                    if brain_activity:
                                        output_file = output_path / f"brain_state_{file_path.stem}_{task_name}.json"
                                        self._export_brain_state_json(brain_activity, output_file, task_name)
                                        converted_count += 1
                                except Exception as e:
                                    error_msg = f"Error converting task {task_name}: {str(e)}"
                                    self.logger.error(error_msg)
                                    errors.append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error processing {cache_file}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            if converted_count > 0:
                return {
                    "type": "convert_cache_response",
                    "success": True,
                    "message": f"Successfully converted {converted_count} cache files",
                    "converted_count": converted_count,
                    "errors": errors if errors else None,
                    "output_dir": str(output_path)
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "No files were converted",
                    "errors": errors
                }
                
        except Exception as e:
            self.logger.error(f"Error in handle_convert_cache: {e}")
            return {
                "type": "error",
                "success": False,
                "message": str(e)
            }

    # ------------------------------------------------------------------ #
    #  New: load a .pt cache file and stream all time frames               #
    # ------------------------------------------------------------------ #

    async def handle_load_cache(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Load a .pt cache file and return its full time series as frames.

        Accepted request fields:
          path (str, optional): path to the .pt file; if omitted, auto-detects
                                the first suitable file under the project tree.

        Response:
          { "type": "cache_loaded", "path": "…", "n_frames": N,
            "frames": [{"activity": [200 floats]}, …],
            "frames_fmri": [...] or null,
            "frames_eeg":  [...] or null,
            "modalities":  ["fmri"] | ["eeg"] | ["fmri","eeg"] }
        """
        import torch

        raw_path = request.get("path")
        cache_path = Path(raw_path) if raw_path else self._find_cache_file()

        if cache_path is None or not cache_path.exists():
            return {
                "type": "error",
                "message": f"找不到缓存文件: {raw_path or '(自动检测失败)'}",
            }

        try:
            data = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            return {"type": "error", "message": f"文件加载失败: {exc}"}

        both = self._extract_time_series_both(data)
        frames_fmri = both.get("fmri", [])
        frames_eeg  = both.get("eeg", [])
        modalities  = both.get("modalities", [])

        # Primary frames: prefer fMRI, fall back to EEG
        primary = frames_fmri if frames_fmri else frames_eeg
        if not primary:
            return {"type": "error", "message": "无法从文件中提取脑区时间序列"}

        self.logger.info(
            f"✓ 缓存加载完成: {cache_path} → {len(primary)} 帧 "
            f"(模态: {modalities})"
        )

        # Store the primary time series for later use by validate_ec / analyze_brain
        import numpy as np
        try:
            self._loaded_time_series = np.array(
                [f["activity"] for f in primary], dtype=np.float32
            )   # (T, 200)
            self._loaded_cache_path = str(cache_path)
            # Also store raw (un-per-frame-normalised) values so that the deviation
            # analysis uses the original sensor amplitudes, not normalised [0,1] data.
            # EEG frames are per-frame min-max normalised (loses temporal structure);
            # raw values preserve the genuine between-task amplitude differences.
            if primary and "raw" in primary[0]:
                self._loaded_raw_time_series = np.array(
                    [f.get("raw", f["activity"]) for f in primary], dtype=np.float32
                )   # (T, 200)
            else:
                self._loaded_raw_time_series = None
        except Exception:
            self._loaded_raw_time_series = None

        return {
            "type":           "cache_loaded",
            "path":           str(cache_path),
            "n_frames":       len(primary),
            "frames":         primary,           # backward-compat: primary modality
            "frames_fmri":    frames_fmri or None,
            "frames_eeg":     frames_eeg  or None,
            "modalities":     modalities,
            # Actual counts before interpolation (fallback = 200 standard Schaefer parcellation)
            "n_fmri_regions": both.get("n_fmri_regions", 200),
            "n_eeg_channels": both.get("n_eeg_channels", 0),
            "success":        True,
        }

    def _find_cache_file(self) -> Optional[Path]:
        """Auto-detect the first suitable .pt cache file in common locations."""
        search_dirs = [Path("."), Path("test_file3"), Path("Unity_TwinBrain"), Path("..")]
        preferred   = ["hetero_graphs.pt", "eeg_data.pt"]
        for d in search_dirs:
            if not d.exists():
                continue
            for name in preferred:
                for f in d.glob(f"**/{name}"):
                    if f.stat().st_size > 1024:
                        return f
            for f in d.glob("**/*.pt"):
                if "__pycache__" not in str(f) and f.stat().st_size > 1024:
                    return f
        return None

    def _find_hetero_data(self, data):
        """Recursively find the first HeteroData in a nested structure."""
        if hasattr(data, "node_types"):
            return data
        if isinstance(data, (list, tuple)) and len(data) > 0:
            return self._find_hetero_data(data[0])
        if isinstance(data, dict):
            for v in data.values():
                result = self._find_hetero_data(v)
                if result is not None:
                    return result
        return None

    def _find_all_hetero_data(self, data) -> list:
        """Recursively collect ALL HeteroData objects from a nested structure.

        When a cache file stores ``Dict[task, List[HeteroData]]`` the full
        time series is split across multiple HeteroData chunks (each holding
        ``spatial_T`` frames, e.g. 384) to keep training memory bounded.
        This method collects every chunk so the caller can concatenate them
        and recover the complete time series.
        """
        results = []
        if hasattr(data, "node_types"):
            results.append(data)
        elif isinstance(data, (list, tuple)):
            for item in data:
                results.extend(self._find_all_hetero_data(item))
        elif isinstance(data, dict):
            for v in data.values():
                results.extend(self._find_all_hetero_data(v))
        return results

    def _extract_time_series(self, data) -> list:
        """Extract per-time-point activity frames from a cache .pt object.

        Strategy (in order of preference):
          1. ``fmri.x_seq`` (shape N×T×F, N == 200 ROIs) → direct 1-to-1 mapping.
          2. ``eeg.x_seq``  (shape N_eeg×T×F, N_eeg ≠ 200) → linear interpolation
             to 200 visualisation slots using np.interp.

        All activity values are normalised to [0, 1] via global 5th/95th
        percentile clipping (robust to outliers in both modalities).
        """
        both = self._extract_time_series_both(data)
        primary = both.get("fmri") or both.get("eeg") or []
        return primary

    def _extract_time_series_both(self, data) -> dict:
        """Extract fMRI and EEG frame sequences from a cache .pt object.

        Returns a dict with keys:
          fmri       – list of {"activity": [200 floats]} or [] if unavailable
          eeg        – list of {"activity": [200 floats]} or [] if unavailable
          modalities – list of available modality names, e.g. ["fmri","eeg"]

        Normalisation strategy:
          fMRI: global 5th/95th-percentile across all time + channels → preserves
                absolute activity level relationships between frames.
          EEG:  per-frame min-max → shows relative spatial variation at each
                instant (EEG temporal variance dominates spatial variance, so
                global normalisation makes all channels appear the same colour).

        Full-series reconstruction:
          Training stores data as Dict[task, List[HeteroData]] where each
          HeteroData holds exactly spatial_T (e.g. 384) consecutive frames to
          keep GPU memory bounded.  This method collects ALL HeteroData chunks
          and concatenates their x_seq tensors along the time axis (dim=1) so
          that the visualiser sees the complete, un-truncated recording.
        """
        import torch
        import numpy as np

        graphs = self._find_all_hetero_data(data)
        if not graphs:
            return {"fmri": [], "eeg": [], "modalities": []}

        n_regions = 200

        def _frames_from_xseq(x_seq, n_out: int) -> list:
            """Global percentile normalisation for fMRI.

            Each frame contains:
              activity – [0,1] normalised values (drives sphere colour)
              raw      – original sensor values (shown in hover tooltip)
            """
            x = x_seq.cpu().float()
            if x.ndim == 2:
                x = x.unsqueeze(-1)      # (N, T) → (N, T, 1)
            N, T, _ = x.shape
            flat  = x[:, :, 0].numpy().ravel()
            p5, p95 = np.percentile(flat, 5), np.percentile(flat, 95)
            scale   = max(p95 - p5, 1e-6)
            result  = []
            src_idx = np.linspace(0.0, 1.0, N) if N != n_out else None
            dst_idx = np.linspace(0.0, 1.0, n_out) if N != n_out else None
            for t in range(T):
                sig  = x[:, t, 0].numpy()
                norm = np.clip((sig - p5) / scale, 0.0, 1.0)
                if N == n_out:
                    arr = norm.astype(np.float32)
                    raw = sig.astype(np.float32)
                else:
                    arr = np.interp(dst_idx, src_idx, norm).astype(np.float32)
                    raw = np.interp(dst_idx, src_idx, sig).astype(np.float32)
                result.append({"activity": arr.tolist(), "raw": raw.tolist()})
            return result

        def _frames_from_xseq_eeg(x_seq, n_out: int) -> list:
            """Per-frame min-max normalisation for EEG.

            EEG temporal variance >> spatial variance, so global normalisation
            collapses all channels to the same colour at any given instant.
            Per-frame normalisation exposes the relative spatial distribution.

            Each frame contains:
              activity – [0,1] normalised values (drives sphere colour)
              raw      – original sensor values (shown in hover tooltip)
            """
            x = x_seq.cpu().float()
            if x.ndim == 2:
                x = x.unsqueeze(-1)      # (N, T) → (N, T, 1)
            N, T, _ = x.shape
            result = []
            src_idx = np.linspace(0.0, 1.0, N) if N != n_out else None
            dst_idx = np.linspace(0.0, 1.0, n_out) if N != n_out else None
            for t in range(T):
                sig = x[:, t, 0].numpy()
                sig_min, sig_max = float(sig.min()), float(sig.max())
                scale = max(sig_max - sig_min, 1e-6)
                norm = np.clip((sig - sig_min) / scale, 0.0, 1.0)
                if N == n_out:
                    arr = norm.astype(np.float32)
                    raw = sig.astype(np.float32)
                else:
                    arr = np.interp(dst_idx, src_idx, norm).astype(np.float32)
                    raw = np.interp(dst_idx, src_idx, sig).astype(np.float32)
                result.append({"activity": arr.tolist(), "raw": raw.tolist()})
            return result

        frames_fmri: list = []
        frames_eeg:  list = []
        n_fmri_raw: int   = 0   # actual number of fMRI ROIs before any interpolation
        n_eeg_raw:  int   = 0   # actual number of EEG channels before interpolation

        try:
            # Collect and concatenate x_seq tensors across all HeteroData chunks.
            # Each chunk covers spatial_T (e.g. 384) consecutive time points; the
            # full recording is recovered by concatenating along dim=1 (time axis).
            fmri_chunks: list = []
            eeg_chunks:  list = []

            for g in graphs:
                if "fmri" in g.node_types:
                    xseq = getattr(g["fmri"], "x_seq", None)
                    if xseq is not None:
                        if xseq.ndim == 2:
                            xseq = xseq.unsqueeze(-1)
                        fmri_chunks.append(xseq)
                        if n_fmri_raw == 0:
                            n_fmri_raw = int(xseq.shape[0])

                if "eeg" in g.node_types:
                    xseq = getattr(g["eeg"], "x_seq", None)
                    if xseq is None:
                        xseq = getattr(g["eeg"], "x", None)
                    if xseq is not None:
                        if xseq.ndim == 2:
                            xseq = xseq.unsqueeze(-1)
                        eeg_chunks.append(xseq)
                        if n_eeg_raw == 0:
                            n_eeg_raw = int(xseq.shape[0])

            if fmri_chunks:
                fmri_cat = torch.cat(fmri_chunks, dim=1)  # (N, T_total, F)
                frames_fmri = _frames_from_xseq(fmri_cat, n_regions)

            if eeg_chunks:
                eeg_cat = torch.cat(eeg_chunks, dim=1)    # (N_eeg, T_total, F)
                frames_eeg = _frames_from_xseq_eeg(eeg_cat, n_regions)

            if len(graphs) > 1:
                self.logger.info(
                    f"_extract_time_series_both: 合并 {len(graphs)} 个数据块 → "
                    f"fMRI {len(frames_fmri)} 帧, EEG {len(frames_eeg)} 帧"
                )

        except Exception as exc:
            self.logger.error(f"_extract_time_series_both error: {exc}")

        modalities = (["fmri"] if frames_fmri else []) + (["eeg"] if frames_eeg else [])
        return {
            "fmri":           frames_fmri,
            "eeg":            frames_eeg,
            "modalities":     modalities,
            "n_fmri_regions": n_fmri_raw,   # e.g. 200 (direct mapping)
            "n_eeg_channels": n_eeg_raw,    # e.g. 64 or 128 (interpolated to 200)
        }

    def _extract_activity_array(self, result: Dict) -> list:
        """Extract a flat [0,1] activity list from various result-dict formats.

        Handles both:
          A) model_server._state_to_json():
             {"regions": [{"activity": float, …}, …], …}
          B) BrainStateExporter.export_brain_state():
             {"brain_state": {"regions": [{"activity": {"fmri": {"amplitude": float}}}}]}}
        """
        # Format A: model_server output
        if "regions" in result:
            return [float(r.get("activity", 0.5)) for r in result["regions"]]

        # Format B: exporter output
        if "brain_state" in result:
            out = []
            for r in result["brain_state"].get("regions", []):
                act = r.get("activity", {})
                if isinstance(act, dict):
                    fmri = act.get("fmri", {})
                    out.append(float(fmri.get("amplitude", 0.5))
                               if isinstance(fmri, dict) else 0.5)
                else:
                    out.append(float(act))
            return out

        return []

    def _extract_brain_activity_from_hetero(self, hetero_data):
        """
        Extract brain activity tensors from HeteroData objects.
        
        Args:
            hetero_data: Can be HeteroData, Dict, or List[HeteroData]
            
        Returns:
            Dict with 'fmri' and/or 'eeg' tensors, or None if extraction fails
        """
        import torch
        brain_activity = {}
        
        try:
            # Handle different input types
            if hasattr(hetero_data, 'node_types'):  # HeteroData object
                # Extract from HeteroData
                if 'fmri' in hetero_data.node_types:
                    fmri_node = hetero_data['fmri']
                    # x_seq is the typical attribute for sequence data
                    if hasattr(fmri_node, 'x_seq'):
                        brain_activity['fmri'] = fmri_node.x_seq
                    elif hasattr(fmri_node, 'x'):
                        brain_activity['fmri'] = fmri_node.x
                
                if 'eeg' in hetero_data.node_types:
                    eeg_node = hetero_data['eeg']
                    if hasattr(eeg_node, 'x_seq'):
                        brain_activity['eeg'] = eeg_node.x_seq
                    elif hasattr(eeg_node, 'x'):
                        brain_activity['eeg'] = eeg_node.x
                        
            elif isinstance(hetero_data, dict):
                # Dict with 'on'/'off' keys (EEG data format)
                if 'on' in hetero_data:
                    on_activity = self._extract_brain_activity_from_hetero(hetero_data['on'])
                    if on_activity and 'eeg' in on_activity:
                        brain_activity['eeg'] = on_activity['eeg']
                        
            elif isinstance(hetero_data, list) and len(hetero_data) > 0:
                # List of HeteroData objects - use first one
                first_activity = self._extract_brain_activity_from_hetero(hetero_data[0])
                if first_activity:
                    brain_activity = first_activity
                    
        except Exception as e:
            self.logger.error(f"Error extracting brain activity: {e}")
            return None
        
        return brain_activity if brain_activity else None
    
    def _convert_eeg_data_cache(self, data, output_path, base_name):
        """
        Convert eeg_data.pt cache file.
        Format: Dict[task_name, Dict["on"|"off", HeteroData]]
        """
        converted = 0
        
        if not isinstance(data, dict):
            self.logger.error(f"EEG data cache is not a dictionary: {type(data)}")
            return 0
        
        for task_name, task_data in data.items():
            try:
                if not isinstance(task_data, dict):
                    continue
                
                # Process "on" and "off" separately
                for condition in ['on', 'off']:
                    if condition not in task_data:
                        continue
                    
                    hetero_data = task_data[condition]
                    brain_activity = self._extract_brain_activity_from_hetero(hetero_data)
                    
                    if brain_activity:
                        output_file = output_path / f"brain_state_{base_name}_{task_name}_{condition}.json"
                        self._export_brain_state_json(
                            brain_activity, 
                            output_file, 
                            f"{task_name}_{condition}"
                        )
                        converted += 1
                        self.logger.info(f"  Converted: {task_name}/{condition} -> {output_file.name}")
                        
            except Exception as e:
                self.logger.error(f"Error converting EEG task {task_name}: {e}")
        
        return converted
    
    def _convert_hetero_graphs_cache(self, data, output_path, base_name):
        """
        Convert hetero_graphs.pt cache file.
        Format: Dict[task_name, List[HeteroData]]
        """
        converted = 0
        
        if not isinstance(data, dict):
            self.logger.error(f"Hetero graphs cache is not a dictionary: {type(data)}")
            return 0
        
        for task_name, graph_list in data.items():
            try:
                if not isinstance(graph_list, list) or len(graph_list) == 0:
                    continue
                
                # Convert first few graphs (to avoid generating too many files)
                max_graphs = min(10, len(graph_list))
                for idx in range(max_graphs):
                    hetero_data = graph_list[idx]
                    brain_activity = self._extract_brain_activity_from_hetero(hetero_data)
                    
                    if brain_activity:
                        output_file = output_path / f"brain_state_{base_name}_{task_name}_{idx:03d}.json"
                        self._export_brain_state_json(
                            brain_activity,
                            output_file,
                            f"{task_name}_frame{idx}"
                        )
                        converted += 1
                        
                if max_graphs < len(graph_list):
                    self.logger.info(f"  Converted {max_graphs}/{len(graph_list)} graphs for task {task_name}")
                else:
                    self.logger.info(f"  Converted all {len(graph_list)} graphs for task {task_name}")
                    
            except Exception as e:
                self.logger.error(f"Error converting hetero graph task {task_name}: {e}")
        
        return converted
    
    def _export_brain_state_json(self, brain_activity, output_file, subject_id):
        """
        Export brain activity to JSON file.
        
        Args:
            brain_activity: Dict with 'fmri' and/or 'eeg' tensors
            output_file: Path to save JSON
            subject_id: Subject identifier for metadata
        """
        import torch
        
        try:
            # Use exporter if available
            if self.exporter:
                self.exporter.export_brain_state(
                    brain_activity=brain_activity,
                    time_point=0,
                    time_second=0.0,
                    subject_id=subject_id,
                    output_path=output_file
                )
            else:
                # Fallback: create minimal JSON
                json_data = {
                    "version": "2.0",
                    "timestamp": datetime.now().isoformat(),
                    "subject_id": subject_id,
                    "brain_state": {
                        "time_point": 0,
                        "regions": []
                    }
                }
                
                # Extract basic statistics for each modality
                for modality, tensor in brain_activity.items():
                    if isinstance(tensor, torch.Tensor):
                        json_data[f"{modality}_shape"] = list(tensor.shape)
                        json_data[f"{modality}_mean"] = float(tensor.mean())
                        json_data[f"{modality}_std"] = float(tensor.std())
                
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            raise
    
    async def handle_stream_start(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stream start request."""
        fps = request.get("fps", 10)
        duration = request.get("duration", 60)
        
        # Input validation
        fps = int(max(1, min(60, fps)))
        duration = int(max(1, min(3600, duration)))
        
        # Start streaming in background
        asyncio.create_task(self.stream_brain_activity(fps, duration))
        
        return {
            "type": "stream_started",
            "success": True,
            "fps": fps,
            "duration": duration
        }
    
    async def stream_brain_activity(self, fps: int = 10, duration: int = 60):
        """Stream brain activity to all connected clients."""
        n_frames = int(fps * duration)
        
        try:
            import torch
            import numpy as np
            
            n_regions = 200
            
            for frame_idx in range(n_frames):
                # Generate dynamic brain state
                # Simulate wave-like activity patterns
                phase = 2 * np.pi * frame_idx / (fps * 5)  # 5 second cycle
                
                fmri_data = torch.randn(n_regions, 1, 1)
                # Add wave pattern
                for i in range(n_regions):
                    fmri_data[i] += 0.5 * np.sin(phase + i * 0.1)
                
                eeg_data = torch.randn(n_regions, 1, 1)
                
                brain_activity = {
                    'fmri': fmri_data,
                    'eeg': eeg_data
                }
                
                # Export brain state
                if self.exporter:
                    brain_state = self.exporter.export_brain_state(
                        brain_activity=brain_activity,
                        time_point=frame_idx,
                        time_second=frame_idx / fps,
                        subject_id="stream"
                    )
                    
                    frame_data = {
                        "type": "stream_frame",
                        "frame": frame_idx,
                        "time": frame_idx / fps,
                        "data": brain_state
                    }
                else:
                    frame_data = {
                        "type": "stream_frame",
                        "frame": frame_idx,
                        "time": frame_idx / fps,
                        "data": {}
                    }
                
                # Broadcast to all clients
                await self.broadcast(frame_data)
                
                # Control frame rate
                await asyncio.sleep(1.0 / fps)
            
            # Send stream end message
            await self.broadcast({"type": "stream_ended", "n_frames": n_frames})
            
        except Exception as e:
            self.logger.error(f"Error in streaming: {e}")
            await self.broadcast({
                "type": "error",
                "message": f"Streaming failed: {str(e)}"
            })
    
    def start(self):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required")
        
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        start_server = websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        asyncio.get_event_loop().run_until_complete(start_server)
        self.logger.info("Server started successfully")
        
        # Run forever
        asyncio.get_event_loop().run_forever()
    
    async def start_async(self):
        """Start server asynchronously (for use in existing event loop)."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required")
        
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            self.logger.info("Server started successfully")
            await asyncio.Future()  # Run forever


# Standalone server function
def start_server(
    model=None,
    exporter=None,
    simulator=None,
    host: str = "0.0.0.0",
    port: int = 8765
):
    """
    Start the brain visualization server.
    
    Usage:
        from unity_integration.realtime_server import start_server
        start_server(model, exporter, simulator)
    """
    server = BrainVisualizationServer(
        model=model,
        exporter=exporter,
        simulator=simulator,
        host=host,
        port=port
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
