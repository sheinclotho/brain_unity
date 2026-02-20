// TwinBrain – StimulationPanel.cs
// Provides stimulation controls matching the web frontend:
//   • Amplitude slider
//   • Frequency slider
//   • Pattern dropdown (sine / pulse / ramp / constant)
//   • Apply button → sends "simulate" request via TwinBrainWebSocket
//   • Load Cache button → sends "load_cache" request
// 
// Reads selected regions from TwinBrainManager.
// Uses TextMeshPro for labels (TMP is pre-installed in Unity 2020+).

using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace TwinBrain
{
    public class StimulationPanel : MonoBehaviour
    {
        [Header("References")]
        public TwinBrainManager   brainManager;
        public TwinBrainWebSocket wsClient;

        [Header("Amplitude")]
        public Slider   amplitudeSlider;
        public TMP_Text amplitudeLabel;

        [Header("Frequency")]
        public Slider   frequencySlider;
        public TMP_Text frequencyLabel;

        [Header("Pattern")]
        public TMP_Dropdown patternDropdown;

        [Header("Buttons")]
        public Button applyButton;
        public Button resetButton;
        public Button loadCacheButton;
        public TMP_InputField cachePathInput;

        [Header("Status")]
        public TMP_Text statusLabel;

        [Header("Defaults")]
        public float defaultAmplitude = 0.5f;
        public float defaultFrequency = 10f;

        // Pattern mapping: dropdown index → backend pattern name
        // Index 0="sine" (正弦波/tACS), 1="pulse" (脉冲), 2="ramp" (斜坡), 3="constant" (持续)
        private static readonly string[] Patterns = { "sine", "pulse", "ramp", "constant" };

        private void Start()
        {
            // Auto-find if not assigned
            if (brainManager == null) brainManager = FindObjectOfType<TwinBrainManager>();
            if (wsClient    == null) wsClient     = FindObjectOfType<TwinBrainWebSocket>();

            if (amplitudeSlider != null)
            {
                amplitudeSlider.minValue = 0.01f;
                amplitudeSlider.maxValue = 1.0f;
                amplitudeSlider.value    = defaultAmplitude;
                amplitudeSlider.onValueChanged.AddListener(v =>
                {
                    if (amplitudeLabel != null) amplitudeLabel.text = v.ToString("F2");
                });
            }

            if (frequencySlider != null)
            {
                frequencySlider.minValue = 1f;
                frequencySlider.maxValue = 50f;
                frequencySlider.value    = defaultFrequency;
                frequencySlider.onValueChanged.AddListener(v =>
                {
                    if (frequencyLabel != null) frequencyLabel.text = $"{v:F0} Hz";
                });
            }

            if (patternDropdown != null)
            {
                patternDropdown.ClearOptions();
                patternDropdown.AddOptions(new List<string> { "正弦波 (tACS)", "脉冲", "斜坡", "持续" });
            }

            if (applyButton    != null) applyButton.onClick.AddListener(OnApplyClicked);
            if (resetButton    != null) resetButton.onClick.AddListener(OnResetClicked);
            if (loadCacheButton != null) loadCacheButton.onClick.AddListener(OnLoadCacheClicked);

            // Wire WebSocket status
            if (wsClient != null)
            {
                wsClient.OnConnected    += () => SetStatus("已连接后端");
                wsClient.OnDisconnected += () => SetStatus("演示模式（后端未连接）");
                wsClient.OnError        += err => SetStatus($"错误: {err}");
            }
        }

        private void OnApplyClicked()
        {
            if (wsClient == null || !wsClient.IsConnected)
            {
                SetStatus("未连接，请启动 python start.py");
                return;
            }

            int[] targets = brainManager != null
                ? brainManager.GetSelectedRegions().ToArray()
                : new int[0];

            // If nothing selected, pick 5 random regions
            if (targets.Length == 0)
            {
                targets = new int[5];
                for (int i = 0; i < 5; i++)
                    targets[i] = Random.Range(0, BrainRegionPositions.N_REGIONS);
            }

            float amp  = amplitudeSlider != null ? amplitudeSlider.value : defaultAmplitude;
            float freq = frequencySlider  != null ? frequencySlider.value  : defaultFrequency;
            int   pat  = patternDropdown  != null ? patternDropdown.value  : 0;

            wsClient.SendSimulate(targets, amp, Patterns[pat], freq, 60);
            SetStatus($"⚡ 刺激发送 → {targets.Length} 个脑区");
        }

        private void OnResetClicked()
        {
            if (brainManager != null) brainManager.ClearSelection();
            if (wsClient != null && wsClient.IsConnected) wsClient.SendGetState();
            SetStatus("已重置");
        }

        private void OnLoadCacheClicked()
        {
            if (wsClient == null || !wsClient.IsConnected)
            {
                SetStatus("未连接后端，请先运行 python start.py");
                return;
            }

            string path = (cachePathInput != null && !string.IsNullOrEmpty(cachePathInput.text))
                ? cachePathInput.text.Trim()
                : null;

            wsClient.SendLoadCache(path);
            SetStatus("加载缓存中…");
        }

        private void SetStatus(string msg)
        {
            if (statusLabel != null) statusLabel.text = msg;
            Debug.Log($"[StimPanel] {msg}");
        }
    }
}
