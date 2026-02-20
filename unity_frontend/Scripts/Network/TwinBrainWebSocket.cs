// TwinBrain – TwinBrainWebSocket.cs
// Real WebSocket client using the NativeWebSocket library.
// 
// SETUP: Add NativeWebSocket to your Unity project via the Package Manager:
//   1. Open  Edit → Project Settings → Package Manager
//   2. Add a scoped registry  OR  edit Packages/manifest.json and add:
//      "com.endel.nativewebsocket": "https://github.com/endel/NativeWebSocket.git#upm"
//   3. (Alternative) Install via git URL in the Package Manager window.
//
// This replaces the old HTTP-polling WebSocketClient which connected to a
// WebSocket server over plain HTTP – an approach that fundamentally cannot
// receive server-push messages.

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_WEBGL && !UNITY_EDITOR
using NativeWebSocket;
#else
using NativeWebSocket;
#endif

namespace TwinBrain
{
    /// <summary>
    /// Real-time WebSocket client for TwinBrain backend.
    /// Attach to any persistent GameObject (e.g. BrainManager).
    /// </summary>
    public class TwinBrainWebSocket : MonoBehaviour
    {
        [Header("Connection")]
        [Tooltip("Backend WebSocket URL")]
        public string serverUrl = "ws://127.0.0.1:8765";

        [Tooltip("Auto-connect on Start")]
        public bool autoConnect = true;

        [Tooltip("Seconds between reconnect attempts")]
        public float reconnectDelay = 3f;

        // ── Events (thread-safe – dispatched on main thread via queue) ──────
        public event Action                  OnConnected;
        public event Action                  OnDisconnected;
        public event Action<string>          OnError;
        public event Action<float[]>         OnBrainState;       // single-frame activity
        public event Action<ActivityFrame[]> OnFrameSequence;    // multi-frame sequence
        public event Action<string>          OnServerVersion;

        // ── State ────────────────────────────────────────────────────────────
        public bool IsConnected => _ws != null && _ws.State == WebSocketState.Open;

        private WebSocket _ws;
        private bool      _reconnecting;

        // ── Unity lifecycle ──────────────────────────────────────────────────
        private void Start()
        {
            if (autoConnect) Connect();
        }

        private void Update()
        {
            // NativeWebSocket requires DispatchMessageQueue() on the main thread
            _ws?.DispatchMessageQueue();
        }

        private void OnDestroy()
        {
            _ws?.Close();
        }

        private void OnApplicationQuit()
        {
            _ws?.Close();
        }

        // ── Public API ───────────────────────────────────────────────────────

        public async void Connect()
        {
            if (IsConnected) return;
            _reconnecting = false;

            try
            {
                _ws = new WebSocket(serverUrl);
                _ws.OnOpen    += HandleOpen;
                _ws.OnClose   += HandleClose;
                _ws.OnError   += HandleError;
                _ws.OnMessage += HandleMessage;

                Debug.Log($"[TwinBrain] Connecting to {serverUrl}…");
                await _ws.Connect();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[TwinBrain] Connect failed: {ex.Message}");
                ScheduleReconnect();
            }
        }

        public async void SendGetState()
        {
            if (!IsConnected) return;
            await _ws.SendText(JsonUtility.ToJson(new GetStateRequest()));
        }

        public async void SendSimulate(int[] targetRegions, float amplitude,
                                       string pattern, float frequency, int duration = 60)
        {
            if (!IsConnected) return;
            var req = new SimulateRequest
            {
                target_regions = targetRegions,
                amplitude      = amplitude,
                pattern        = pattern,
                frequency      = frequency,
                duration       = duration
            };
            await _ws.SendText(JsonUtility.ToJson(req));
        }

        public async void SendLoadCache(string path = null)
        {
            if (!IsConnected) return;
            // Build JSON manually so we can handle null path correctly.
            // Escape path to guard against special characters.
            if (path == null)
            {
                await _ws.SendText("{\"type\":\"load_cache\"}");
            }
            else
            {
                string safePath = path.Replace("\\", "\\\\").Replace("\"", "\\\"");
                await _ws.SendText($"{{\"type\":\"load_cache\",\"path\":\"{safePath}\"}}");
            }
        }

        // ── Internal handlers ────────────────────────────────────────────────

        private void HandleOpen()
        {
            Debug.Log("[TwinBrain] Connected.");
            OnConnected?.Invoke();
            // Request initial state immediately
            SendGetState();
        }

        private void HandleClose(WebSocketCloseCode code)
        {
            Debug.Log($"[TwinBrain] Disconnected ({code}).");
            OnDisconnected?.Invoke();
            if (!_reconnecting) ScheduleReconnect();
        }

        private void HandleError(string err)
        {
            Debug.LogWarning($"[TwinBrain] WS error: {err}");
            OnError?.Invoke(err);
        }

        private void HandleMessage(byte[] data)
        {
            string json = System.Text.Encoding.UTF8.GetString(data);
            ParseAndDispatch(json);
        }

        private void ParseAndDispatch(string json)
        {
            // Read the 'type' field cheaply before full deserialization
            BaseMessage hdr;
            try { hdr = JsonUtility.FromJson<BaseMessage>(json); }
            catch { Debug.LogWarning("[TwinBrain] Malformed JSON"); return; }

            switch (hdr.type)
            {
                case "welcome":
                    var welcome = JsonUtility.FromJson<WelcomeMessage>(json);
                    OnServerVersion?.Invoke(welcome.version ?? "?");
                    break;

                case "brain_state":
                    var bs = JsonUtility.FromJson<BrainStateMessage>(json);
                    if (bs.activity != null) OnBrainState?.Invoke(bs.activity);
                    break;

                case "simulation_result":
                case "cache_loaded":
                    var seq = JsonUtility.FromJson<FrameSequenceMessage>(json);
                    if (seq.frames != null && seq.frames.Length > 0)
                        OnFrameSequence?.Invoke(seq.frames);
                    break;

                case "error":
                    Debug.LogWarning($"[TwinBrain] Server error: {hdr.message}");
                    break;

                default:
                    Debug.Log($"[TwinBrain] Unhandled message type: {hdr.type}");
                    break;
            }
        }

        private void ScheduleReconnect()
        {
            if (_reconnecting) return;
            _reconnecting = true;
            StartCoroutine(ReconnectAfterDelay());
        }

        private IEnumerator ReconnectAfterDelay()
        {
            yield return new WaitForSeconds(reconnectDelay);
            _reconnecting = false;
            Connect();
        }
    }
}
