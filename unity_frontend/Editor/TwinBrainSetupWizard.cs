// TwinBrain – TwinBrainSetupWizard.cs (Editor-only)
// One-click scene setup: creates BrainManager, UI Canvas, camera, lighting.
// Access via  Unity menu → TwinBrain → Setup Scene
//
// PREREQUISITES:
//   1. NativeWebSocket package installed (see TwinBrainWebSocket.cs)
//   2. TextMeshPro imported (included in Unity 2020+; click "Import TMP Essentials" if prompted)

#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;

namespace TwinBrain.Editor
{
    public class TwinBrainSetupWizard : EditorWindow
    {
        private string wsUrl = "ws://127.0.0.1:8765";
        private bool   createCamera    = true;
        private bool   createLighting  = true;
        private bool   createUI        = true;
        private bool   createTimeline  = true;
        private bool   createStimPanel = true;
        private Vector2 scroll;

        [MenuItem("TwinBrain/Setup Scene", false, 1)]
        public static void ShowWindow()
        {
            var win = GetWindow<TwinBrainSetupWizard>("TwinBrain Setup");
            win.minSize = new Vector2(420, 500);
        }

        [MenuItem("TwinBrain/Check NativeWebSocket", false, 10)]
        public static void CheckNativeWebSocket()
        {
            // Try to reference the NativeWebSocket assembly
            var type = System.Type.GetType("NativeWebSocket.WebSocket, NativeWebSocket");
            if (type != null)
                EditorUtility.DisplayDialog("NativeWebSocket", "✅ NativeWebSocket is installed.", "OK");
            else
                EditorUtility.DisplayDialog(
                    "NativeWebSocket Missing",
                    "NativeWebSocket is NOT installed.\n\n" +
                    "Add the following line to  Packages/manifest.json  under \"dependencies\":\n\n" +
                    "\"com.endel.nativewebsocket\": \"https://github.com/endel/NativeWebSocket.git#upm\"\n\n" +
                    "Then return to Unity so the package imports automatically.",
                    "OK");
        }

        private void OnGUI()
        {
            scroll = EditorGUILayout.BeginScrollView(scroll);

            GUILayout.Label("TwinBrain — Scene Setup Wizard", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            EditorGUILayout.HelpBox(
                "Creates a complete TwinBrain scene:\n" +
                "• BrainManager (TwinBrainManager + WebSocket + Timeline)\n" +
                "• 200 brain-region spheres at anatomical positions\n" +
                "• Directional light + ambient\n" +
                "• Main Camera with orbit script\n" +
                "• UI Canvas: timeline slider, stimulation panel, status HUD\n\n" +
                "Run this on an EMPTY scene. You can re-run it to reset.",
                MessageType.Info
            );

            EditorGUILayout.Space();
            GUILayout.Label("Configuration", EditorStyles.boldLabel);
            wsUrl         = EditorGUILayout.TextField("Backend WebSocket URL", wsUrl);
            createCamera  = EditorGUILayout.Toggle("Create Camera",         createCamera);
            createLighting= EditorGUILayout.Toggle("Create Lighting",       createLighting);
            createUI      = EditorGUILayout.Toggle("Create UI Canvas",      createUI);
            createTimeline= EditorGUILayout.Toggle("  ↳ Timeline bar",      createTimeline);
            createStimPanel=EditorGUILayout.Toggle("  ↳ Stimulation panel", createStimPanel);

            EditorGUILayout.Space();

            using (new EditorGUI.DisabledScope(!NativeWebSocketAvailable()))
            {
                if (GUILayout.Button("⚡ Build Scene", GUILayout.Height(44)))
                {
                    if (EditorUtility.DisplayDialog("TwinBrain Setup",
                        "This will modify the current scene.\nContinue?", "Yes", "Cancel"))
                        BuildScene();
                }
            }

            if (!NativeWebSocketAvailable())
            {
                EditorGUILayout.HelpBox(
                    "NativeWebSocket not found. Use  TwinBrain → Check NativeWebSocket  for install instructions.",
                    MessageType.Warning
                );
            }

            EditorGUILayout.Space();
            if (GUILayout.Button("Open NativeWebSocket Install Guide"))
                Application.OpenURL("https://github.com/endel/NativeWebSocket#installation");

            EditorGUILayout.EndScrollView();
        }

        private bool NativeWebSocketAvailable()
        {
            return System.Type.GetType("NativeWebSocket.WebSocket, NativeWebSocket") != null;
        }

        // ── Scene builder ───────────────────────────────────────────────────

        private void BuildScene()
        {
            Undo.SetCurrentGroupName("TwinBrain Setup");
            int undoGroup = Undo.GetCurrentGroup();

            try
            {
                EditorUtility.DisplayProgressBar("TwinBrain", "Creating BrainManager…", 0.1f);
                var brainRoot = CreateBrainManager();

                if (createLighting)
                {
                    EditorUtility.DisplayProgressBar("TwinBrain", "Setting up lighting…", 0.3f);
                    SetupLighting();
                }

                if (createCamera)
                {
                    EditorUtility.DisplayProgressBar("TwinBrain", "Creating camera…", 0.5f);
                    SetupCamera();
                }

                if (createUI)
                {
                    EditorUtility.DisplayProgressBar("TwinBrain", "Building UI…", 0.7f);
                    CreateUICanvas(brainRoot);
                }

                Undo.CollapseUndoOperations(undoGroup);

                EditorUtility.DisplayDialog("TwinBrain",
                    "✅ Scene setup complete!\n\n" +
                    "Next steps:\n" +
                    "1. Make sure  python start.py  is running\n" +
                    "2. Press Play in Unity\n" +
                    "3. Click brain regions → stimulate → observe propagation",
                    "OK");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }

        // ── BrainManager ────────────────────────────────────────────────────

        private GameObject CreateBrainManager()
        {
            var go = new GameObject("BrainManager");
            Undo.RegisterCreatedObjectUndo(go, "Create BrainManager");

            var ws  = go.AddComponent<TwinBrainWebSocket>();
            ws.serverUrl = wsUrl;

            var tl  = go.AddComponent<TimelinePlayer>();
            var mgr = go.AddComponent<TwinBrainManager>();

            // Create a child transform as the brain root
            var brainContainer = new GameObject("BrainRegions");
            brainContainer.transform.SetParent(go.transform, false);
            mgr.brainRoot = brainContainer.transform;

            // Add a BrainRegionTag helper (regions are spawned at runtime; none needed here)
            Debug.Log("[TwinBrain Setup] BrainManager created.");
            return go;
        }

        // ── Lighting ────────────────────────────────────────────────────────

        private void SetupLighting()
        {
            // Remove default directional light if present
            var existing = Object.FindObjectOfType<Light>();
            if (existing != null && existing.type == LightType.Directional)
                Undo.DestroyObjectImmediate(existing.gameObject);

            var lightGo = new GameObject("DirectionalLight");
            Undo.RegisterCreatedObjectUndo(lightGo, "Create Light");
            var light = lightGo.AddComponent<Light>();
            light.type      = LightType.Directional;
            light.intensity = 0.85f;
            light.color     = Color.white;
            lightGo.transform.rotation = Quaternion.Euler(45f, -30f, 0f);

            RenderSettings.ambientMode  = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.15f, 0.15f, 0.28f);
        }

        // ── Camera ──────────────────────────────────────────────────────────

        private void SetupCamera()
        {
            var camGo = Camera.main != null ? Camera.main.gameObject : new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            Undo.RegisterFullObjectHierarchyUndo(camGo, "Setup Camera");

            var cam = camGo.GetComponent<Camera>();
            if (cam == null) cam = camGo.AddComponent<Camera>();
            cam.backgroundColor = new Color(0.027f, 0.031f, 0.102f);
            cam.clearFlags      = CameraClearFlags.SolidColor;
            cam.fieldOfView     = 48f;

            camGo.transform.position = new Vector3(0f, 1f, -3.2f);
            camGo.transform.LookAt(Vector3.zero);

            // Attach a simple orbit controller so the user can rotate with mouse
            if (camGo.GetComponent<OrbitCamera>() == null)
                camGo.AddComponent<OrbitCamera>();
        }

        // ── UI Canvas ───────────────────────────────────────────────────────

        private void CreateUICanvas(GameObject brainManagerGo)
        {
            var canvasGo = new GameObject("TwinBrainUI");
            Undo.RegisterCreatedObjectUndo(canvasGo, "Create Canvas");

            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 10;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            // --- Timeline bar (bottom strip) ---
            if (createTimeline)
            {
                var tlBar = CreateTimelineBar(canvasGo.transform);
                // Wire TimelineUI to the TimelinePlayer
                var tlUI = tlBar.AddComponent<TimelineUI>();
                tlUI.timeline = brainManagerGo?.GetComponent<TimelinePlayer>();
            }

            // --- Stimulation panel (right side) ---
            if (createStimPanel)
            {
                var stimPanel = CreateStimulationPanel(canvasGo.transform);
                var sp = stimPanel.AddComponent<StimulationPanel>();
                sp.brainManager = brainManagerGo?.GetComponent<TwinBrainManager>();
                sp.wsClient     = brainManagerGo?.GetComponent<TwinBrainWebSocket>();
            }

            // --- Status HUD (top-left) ---
            var hud = CreateStatusHUD(canvasGo.transform);
            var sh  = hud.AddComponent<StatusHUD>();
            sh.wsClient     = brainManagerGo?.GetComponent<TwinBrainWebSocket>();
            sh.brainManager = brainManagerGo?.GetComponent<TwinBrainManager>();
            sh.timeline     = brainManagerGo?.GetComponent<TimelinePlayer>();
        }

        // ── UI element builders (minimal boilerplate) ────────────────────────

        private GameObject CreateTimelineBar(Transform parent)
        {
            var bar = new GameObject("TimelineBar");
            bar.transform.SetParent(parent, false);
            var rt = bar.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 0f);
            rt.anchorMax = new Vector2(1f, 0f);
            rt.pivot     = new Vector2(0.5f, 0f);
            rt.sizeDelta = new Vector2(0f, 52f);
            rt.anchoredPosition = Vector2.zero;

            var img = bar.AddComponent<Image>();
            img.color = new Color(0.04f, 0.04f, 0.09f, 0.90f);

            // Play/pause button
            var btnGo = CreateButton(bar.transform, "▶", new Vector2(26f, 26f));
            btnGo.GetComponent<RectTransform>().anchoredPosition = new Vector2(-20f, 0f);
            // Slider
            var slider = CreateSlider(bar.transform);
            return bar;
        }

        private GameObject CreateStimulationPanel(Transform parent)
        {
            var panel = new GameObject("StimPanel");
            panel.transform.SetParent(parent, false);
            var rt = panel.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(1f, 0f);
            rt.anchorMax = new Vector2(1f, 1f);
            rt.pivot     = new Vector2(1f, 0.5f);
            rt.sizeDelta = new Vector2(240f, 0f);
            rt.anchoredPosition = Vector2.zero;

            var img = panel.AddComponent<Image>();
            img.color = new Color(0.05f, 0.05f, 0.12f, 0.90f);

            AddLabel(panel.transform, "TwinBrain 控制", new Vector2(0f, -20f), 16, FontStyles.Bold);
            return panel;
        }

        private GameObject CreateStatusHUD(Transform parent)
        {
            var hud = new GameObject("StatusHUD");
            hud.transform.SetParent(parent, false);
            var rt = hud.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot     = new Vector2(0f, 1f);
            rt.anchoredPosition = new Vector2(10f, -10f);
            rt.sizeDelta = new Vector2(220f, 80f);

            var img = hud.AddComponent<Image>();
            img.color = new Color(0.04f, 0.04f, 0.09f, 0.80f);
            return hud;
        }

        // ── Generic UI helpers ───────────────────────────────────────────────

        private GameObject CreateButton(Transform parent, string label, Vector2 size)
        {
            var go  = new GameObject("Button");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = new Color(0.15f, 0.15f, 0.25f);
            var btn = go.AddComponent<Button>();
            var rt  = go.GetComponent<RectTransform>();
            rt.sizeDelta = size;

            var txtGo = new GameObject("Label");
            txtGo.transform.SetParent(go.transform, false);
            var tmp = txtGo.AddComponent<TextMeshProUGUI>();
            tmp.text      = label;
            tmp.fontSize  = 14;
            tmp.alignment = TextAlignmentOptions.Center;
            var trt = txtGo.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero;
            trt.anchorMax = Vector2.one;
            trt.sizeDelta = Vector2.zero;
            return go;
        }

        private GameObject CreateSlider(Transform parent)
        {
            var go = new GameObject("TimelineSlider");
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(0.08f, 0.5f);
            rt.anchorMax = new Vector2(0.92f, 0.5f);
            rt.sizeDelta = new Vector2(0f, 20f);
            rt.anchoredPosition = Vector2.zero;
            go.AddComponent<Slider>();
            return go;
        }

        private void AddLabel(Transform parent, string text, Vector2 pos, int size = 12,
                               FontStyles style = FontStyles.Normal)
        {
            if (string.IsNullOrEmpty(text)) return;
            int maxLen = Mathf.Min(text.Length, 8);
            var go = new GameObject("Label_" + text.Substring(0, maxLen));
            go.transform.SetParent(parent, false);
            var rt  = go.AddComponent<RectTransform>();
            rt.anchoredPosition = pos;
            rt.sizeDelta = new Vector2(220f, 30f);
            var tmp = go.AddComponent<TextMeshProUGUI>();
            tmp.text       = text;
            tmp.fontSize   = size;
            tmp.fontStyle  = style;
            tmp.color      = new Color(0.88f, 0.88f, 1f);
        }
    }
}
#endif
