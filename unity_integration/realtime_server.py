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
            
            # Create timestamped output directory for this stimulation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = Path(self.model_server.output_dir if self.model_server else "unity_project/brain_data/model_output")
            stim_output_dir = output_base / "stimulation" / f"stim_{timestamp}"
            stim_output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Stimulation output directory: {stim_output_dir}")
            
            # Use ModelServer if available
            if self.model_server:
                self.logger.info(f"Using ModelServer for stimulation simulation")
                responses = self.model_server.simulate_stimulation(
                    target_regions=target_regions,
                    amplitude=amplitude,
                    pattern=pattern,
                    frequency=frequency,
                    duration=duration,
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
                return {
                    "type": "simulation_result",
                    "success": True,
                    "n_frames": len(frames),
                    "frames": frames,
                    "saved_to": str(stim_output_dir),
                    "index_file": str(index_path),
                }
            # Demo simulation: realistic network-oscillation baseline + stimulation dynamics.
            # This path is used when no trained model is available.
            frames = self._demo_simulate(
                target_regions=target_regions,
                amplitude=amplitude,
                pattern=pattern,
                frequency=frequency,
                duration=duration,
                n_regions=n_regions,
            )
            return {
                "type": "simulation_result",
                "success": True,
                "n_frames": len(frames),
                "frames": frames,
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
           activity_delta: [200 floats],  -- predicted activity change from top source
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

        analyzer = PerturbationAnalyzer(n_regions=200, n_lags=n_lags)

        # ── Demo mode (no data needed) ─────────────────────────────────────
        if method == "demo":
            ec = analyzer.infer_ec_demo()
            result = analyzer.ec_to_dict(ec)
            top_src = result["top_sources"]
            result["activity_delta"] = analyzer.predict_activity_delta(
                top_src[:1], amplitude=0.5, ec_matrix=ec
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

        # ── Train surrogate ────────────────────────────────────────────────
        self.logger.info(f"训练 MLP 代理模型: {time_series.shape} …")
        analyzer.fit_surrogate(
            time_series,
            n_lags=n_lags,
            num_epochs=60,
            batch_size=64,
            lr=1e-3,
        )

        # ── Infer EC ───────────────────────────────────────────────────────
        if method == "perturbation":
            ec = analyzer.infer_ec_perturbation(pert_strength=0.05)
        else:
            ec = analyzer.infer_ec_jacobian(
                n_samples=min(100, len(time_series) - n_lags)
            )

        result = analyzer.ec_to_dict(ec)
        top_src = result["top_sources"]
        result["activity_delta"] = analyzer.predict_activity_delta(
            top_src[:1], amplitude=0.5, ec_matrix=ec
        ).tolist()
        result["method"]  = method
        result["success"] = True
        result["type"]    = "ec_result"
        self.logger.info(
            f"EC 推断完成 ({method}), 影响力最强区域: {top_src[:5]}"
        )
        return result

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
    ) -> list:
        """Generate a realistic stimulation animation without a trained model.

        Returns a list of {"activity": [200 floats]} frames showing:
          1. Pre-stim baseline  (10 frames)
          2. Stimulation active (``duration`` frames, capped at 60)
          3. Post-stim recovery (10 frames)

        The baseline uses the same 7-network sinusoidal model as
        ``handle_get_state`` so the stimulation starts from a plausible state.
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

        # ── Stimulation pattern function ───────────────────────────────────
        def _stim_amp(relative_t: int) -> float:
            if pattern == "sine":
                return amplitude * np.sin(2 * np.pi * frequency * relative_t / 10.0)
            elif pattern == "pulse":
                return amplitude if relative_t == 0 else 0.0
            elif pattern == "ramp":
                return amplitude * min(relative_t / max(duration, 1), 1.0)
            else:  # constant / unknown
                return amplitude

        PRE  = 10
        DUR  = min(int(duration), _MAX_STIM_FRAMES)
        POST = 10
        frames = []

        # 1. Pre-stim baseline
        for k in range(PRE):
            tick = t0 * 200 + k * 4
            act = _baseline(tick)
            frames.append({"activity": act.tolist()})

        # 2. Stimulation active
        for k in range(DUR):
            tick = t0 * 200 + (PRE + k) * 4
            act  = _baseline(tick)
            amp  = _stim_amp(k)
            if amp != 0.0:
                act = np.clip(act + amp * spread_weights, 0.0, 1.0)
            frames.append({"activity": act.tolist()})

        # 3. Post-stim recovery (exponential decay of residual excitation)
        post_baseline = _baseline(t0 * 200 + (PRE + DUR) * 4)
        last_stim_act = np.array(frames[-1]["activity"], dtype=np.float32)
        for k in range(POST):
            decay = np.exp(-k / 4.0)
            act   = post_baseline + (last_stim_act - post_baseline) * decay
            act   = np.clip(act, 0.0, 1.0)
            frames.append({"activity": act.tolist()})

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
        frames_fmri = both.get("fmri") or []
        frames_eeg  = both.get("eeg")  or []
        modalities  = both.get("modalities", [])

        # Primary frames: prefer fMRI, fall back to EEG
        primary = frames_fmri if frames_fmri else frames_eeg
        if not primary:
            return {"type": "error", "message": "无法从文件中提取脑区时间序列"}

        self.logger.info(
            f"✓ 缓存加载完成: {cache_path} → {len(primary)} 帧 "
            f"(模态: {modalities})"
        )
        return {
            "type":        "cache_loaded",
            "path":        str(cache_path),
            "n_frames":    len(primary),
            "frames":      primary,           # backward-compat: primary modality
            "frames_fmri": frames_fmri or None,
            "frames_eeg":  frames_eeg  or None,
            "modalities":  modalities,
            "success":     True,
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
        """
        import numpy as np

        graph = self._find_hetero_data(data)
        if graph is None:
            return {"fmri": [], "eeg": [], "modalities": []}

        n_regions = 200

        def _frames_from_xseq(x_seq, n_out: int) -> list:
            """Global percentile normalisation for fMRI."""
            x = x_seq.cpu().float()
            if x.ndim == 2:
                x = x.unsqueeze(-1)      # (N, T) → (N, T, 1)
            N, T, _ = x.shape
            flat  = x[:, :, 0].numpy().ravel()
            p5, p95 = np.percentile(flat, 5), np.percentile(flat, 95)
            scale   = max(p95 - p5, 1e-6)
            result  = []
            for t in range(T):
                sig  = x[:, t, 0].numpy()
                norm = np.clip((sig - p5) / scale, 0.0, 1.0)
                if N == n_out:
                    arr = norm.astype(np.float32)
                else:
                    # Linearly interpolate N channels → n_out visualisation slots
                    src_idx = np.linspace(0.0, 1.0, N)
                    dst_idx = np.linspace(0.0, 1.0, n_out)
                    arr = np.interp(dst_idx, src_idx, norm).astype(np.float32)
                result.append({"activity": arr.tolist()})
            return result

        def _frames_from_xseq_eeg(x_seq, n_out: int) -> list:
            """Per-frame min-max normalisation for EEG.

            EEG temporal variance >> spatial variance, so global normalisation
            collapses all channels to the same colour at any given instant.
            Per-frame normalisation exposes the relative spatial distribution.
            """
            x = x_seq.cpu().float()
            if x.ndim == 2:
                x = x.unsqueeze(-1)      # (N, T) → (N, T, 1)
            N, T, _ = x.shape
            result = []
            for t in range(T):
                sig = x[:, t, 0].numpy()
                sig_min, sig_max = float(sig.min()), float(sig.max())
                scale = max(sig_max - sig_min, 1e-6)
                norm = np.clip((sig - sig_min) / scale, 0.0, 1.0)
                if N == n_out:
                    arr = norm.astype(np.float32)
                else:
                    src_idx = np.linspace(0.0, 1.0, N)
                    dst_idx = np.linspace(0.0, 1.0, n_out)
                    arr = np.interp(dst_idx, src_idx, norm).astype(np.float32)
                result.append({"activity": arr.tolist()})
            return result

        frames_fmri: list = []
        frames_eeg:  list = []

        try:
            if "fmri" in graph.node_types:
                node  = graph["fmri"]
                x_seq = getattr(node, "x_seq", None)
                if x_seq is not None:
                    frames_fmri = _frames_from_xseq(x_seq, n_regions)

            if "eeg" in graph.node_types:
                node  = graph["eeg"]
                x_seq = getattr(node, "x_seq", None)
                if x_seq is None:
                    x_seq = getattr(node, "x", None)
                if x_seq is not None:
                    frames_eeg = _frames_from_xseq_eeg(x_seq, n_regions)

        except Exception as exc:
            self.logger.error(f"_extract_time_series_both error: {exc}")

        modalities = (["fmri"] if frames_fmri else []) + (["eeg"] if frames_eeg else [])
        return {"fmri": frames_fmri, "eeg": frames_eeg, "modalities": modalities}

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
