// TwinBrain – TwinBrainManager.cs
// Central orchestrator.  Attach to the root "BrainManager" GameObject.
// 
// Responsibilities:
//   • Spawns 200 sphere GameObjects at anatomical positions on Start()
//   • Applies heatmap colors + scale based on incoming activity arrays
//   • Owns the WebSocket component
//   • Delegates timeline playback to TimelinePlayer
//   • Exposes API to StimulationPanel and RegionSelector

using System.Collections.Generic;
using UnityEngine;

namespace TwinBrain
{
    [RequireComponent(typeof(TwinBrainWebSocket))]
    [RequireComponent(typeof(TimelinePlayer))]
    public class TwinBrainManager : MonoBehaviour
    {
        // ── Inspector ────────────────────────────────────────────────────────
        [Header("Brain Region Prefab")]
        [Tooltip("Assign a sphere prefab; if null, a plain Unity sphere is created.")]
        public GameObject regionPrefab;

        [Tooltip("Uniform scale applied to each region sphere (Unity units = mm)")]
        public float regionBaseScale = 0.01f;   // 1 unit = 100 mm, sphere radius ≈ 4 mm

        [Header("Brain Container")]
        [Tooltip("Parent transform for all region GameObjects.")]
        public Transform brainRoot;

        [Header("Glass Brain Outline")]
        [Tooltip("Transparent mesh representing the outer brain silhouette (optional).")]
        public GameObject glassOutlinePrefab;

        // ── Internal ─────────────────────────────────────────────────────────
        private GameObject[]    _regionObjects;
        private Renderer[]      _regionRenderers;
        private HashSet<int>    _selected = new HashSet<int>();

        private TwinBrainWebSocket _ws;
        private TimelinePlayer     _timeline;

        // current live activity (for demo / idle animation)
        private float[] _activity = new float[BrainRegionPositions.N_REGIONS];
        private float   _demoTick;

        // ── Modality cache ────────────────────────────────────────────────────
        /// <summary>Cached fMRI frame sequence from the last cache_loaded message.</summary>
        public ActivityFrame[] CachedFmriFrames { get; private set; }
        /// <summary>Cached EEG frame sequence from the last cache_loaded message.</summary>
        public ActivityFrame[] CachedEegFrames  { get; private set; }
        /// <summary>Available modalities in the last loaded cache (e.g. "fmri", "eeg").</summary>
        public string[]        AvailableModalities { get; private set; } = new string[0];

        // ── Events ────────────────────────────────────────────────────────────
        public delegate void RegionClickedDelegate(int regionId);
        public static event RegionClickedDelegate OnRegionClicked;

        // ── Unity lifecycle ───────────────────────────────────────────────────
        private void Awake()
        {
            _ws       = GetComponent<TwinBrainWebSocket>();
            _timeline = GetComponent<TimelinePlayer>();
        }

        private void Start()
        {
            SpawnBrainRegions();
            SpawnGlassBrain();

            // Wire WebSocket events
            _ws.OnBrainState    += ApplyActivity;
            _ws.OnFrameSequence += OnFrameSequenceReceived;
            _ws.OnConnected     += () => Debug.Log("[BrainManager] WS connected");
            _ws.OnDisconnected  += () => Debug.Log("[BrainManager] WS disconnected");

            // Demo: initialise with network-oscillation activity
            for (int i = 0; i < _activity.Length; i++) _activity[i] = 0.35f;
            ApplyActivity(_activity);
        }

        private void Update()
        {
            // Demo oscillation when not receiving live data
            if (!_ws.IsConnected && !_timeline.IsPlaying)
            {
                _demoTick += Time.deltaTime;
                UpdateDemoActivity();
            }

            // Mouse / touch selection forwarded from RegionSelector
            if (Input.GetMouseButtonDown(0))
                TryPickRegion(Input.mousePosition);
        }

        // ── Public API ────────────────────────────────────────────────────────

        /// <summary>Update all region colors from a flat [0,1] activity array.</summary>
        public void ApplyActivity(float[] activity)
        {
            if (activity == null) return;
            int n = Mathf.Min(activity.Length, _regionObjects.Length);
            for (int i = 0; i < n; i++)
            {
                float v = Mathf.Clamp01(activity[i]);
                _activity[i] = v;
                if (_regionRenderers[i] == null) continue;

                bool sel   = _selected.Contains(i);
                float scale = regionBaseScale * (0.65f + v * 0.60f) * (sel ? 1.6f : 1f);
                _regionObjects[i].transform.localScale = Vector3.one * scale;
                ActivityHeatmap.Apply(_regionRenderers[i], v, sel ? 0.55f : 0.22f);
            }
        }

        public void SetSelected(int regionId, bool selected)
        {
            if (selected) _selected.Add(regionId);
            else          _selected.Remove(regionId);
            // Refresh that region's visual
            ApplyActivity(_activity);
        }

        public HashSet<int> GetSelectedRegions() => _selected;

        public void ClearSelection()
        {
            _selected.Clear();
            ApplyActivity(_activity);
        }

        /// <summary>
        /// Switch the timeline to display a specific modality ("fmri" or "eeg").
        /// Uses the frames cached from the last cache_loaded server response.
        /// </summary>
        public void SwitchModality(string modality)
        {
            ActivityFrame[] frames = null;
            if (modality == "fmri" && CachedFmriFrames != null && CachedFmriFrames.Length > 0)
                frames = CachedFmriFrames;
            else if (modality == "eeg" && CachedEegFrames != null && CachedEegFrames.Length > 0)
                frames = CachedEegFrames;

            if (frames != null)
            {
                _timeline.Load(frames, this);
                Debug.Log($"[BrainManager] Switched to modality: {modality} ({frames.Length} frames)");
            }
            else
            {
                Debug.LogWarning($"[BrainManager] Modality '{modality}' not available in current cache.");
            }
        }

        // ── Private helpers ───────────────────────────────────────────────────

        private void OnFrameSequenceReceived(FrameSequenceMessage msg)
        {
            // Store per-modality caches when available (cache_loaded messages)
            if (msg.frames_fmri != null && msg.frames_fmri.Length > 0)
                CachedFmriFrames = msg.frames_fmri;
            if (msg.frames_eeg  != null && msg.frames_eeg.Length  > 0)
                CachedEegFrames  = msg.frames_eeg;
            if (msg.modalities  != null)
                AvailableModalities = msg.modalities;

            // Load the primary frames into the timeline as before
            if (msg.frames != null && msg.frames.Length > 0)
                _timeline.Load(msg.frames, this);
        }

        private void SpawnBrainRegions()
        {
            Vector3[] positions = BrainRegionPositions.Generate();
            int n = positions.Length;

            _regionObjects   = new GameObject[n];
            _regionRenderers = new Renderer[n];

            Transform root = brainRoot != null ? brainRoot : transform;

            for (int i = 0; i < n; i++)
            {
                GameObject go;
                if (regionPrefab != null)
                    go = Instantiate(regionPrefab, root);
                else
                {
                    go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    go.transform.SetParent(root, false);
                    // Remove collider from the primitive; add a simpler trigger
                    Destroy(go.GetComponent<SphereCollider>());
                    var sc = go.AddComponent<SphereCollider>();
                    sc.isTrigger = false;
                    sc.radius    = 1.0f;
                }

                go.name = $"Region_{i:000}";
                go.transform.localPosition = positions[i] * regionBaseScale;
                float initScale = regionBaseScale * 0.85f;
                go.transform.localScale = Vector3.one * initScale;

                // Tag for raycast lookup
                var tag = go.AddComponent<BrainRegionTag>();
                tag.regionId = i;

                _regionObjects[i]   = go;
                _regionRenderers[i] = go.GetComponent<Renderer>();
            }
        }

        private void SpawnGlassBrain()
        {
            if (glassOutlinePrefab == null) return;
            Transform root = brainRoot != null ? brainRoot : transform;
            Instantiate(glassOutlinePrefab, root);
        }

        private void TryPickRegion(Vector3 screenPos)
        {
            if (Camera.main == null) return;
            Ray ray = Camera.main.ScreenPointToRay(screenPos);
            if (!Physics.Raycast(ray, out RaycastHit hit)) return;

            var tag = hit.collider.GetComponent<BrainRegionTag>();
            if (tag == null) tag = hit.collider.GetComponentInParent<BrainRegionTag>();
            if (tag == null) return;

            int id = tag.regionId;
            bool nowSelected = !_selected.Contains(id);
            SetSelected(id, nowSelected);
            OnRegionClicked?.Invoke(id);
        }

        private static readonly float[] NetFreqs  = { 0.012f, 0.020f, 0.035f, 0.028f, 0.008f, 0.025f, 0.010f };
        private static readonly float[] NetPhases = { 0f,     1f,     2.1f,   0.7f,   3.2f,   1.8f,   0.4f   };

        private void UpdateDemoActivity()
        {
            int netSize = BrainRegionPositions.N_REGIONS / 7;
            for (int i = 0; i < _activity.Length; i++)
            {
                int  n = Mathf.Min(i / netSize, 6);
                float v = 0.36f + 0.22f * Mathf.Sin(NetFreqs[n] * _demoTick + NetPhases[n] + i * 0.08f)
                               + 0.04f * Mathf.Sin(0.003f * _demoTick + i * 0.25f);
                _activity[i] = Mathf.Clamp01(v);
            }
            ApplyActivity(_activity);
        }
    }
}
