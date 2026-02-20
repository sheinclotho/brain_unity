// TwinBrain – StatusHUD.cs
// Displays connection status, frame info, and selected-region count.
// Attach anywhere in the scene; auto-finds required components.

using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace TwinBrain
{
    public class StatusHUD : MonoBehaviour
    {
        [Header("References")]
        public TwinBrainWebSocket wsClient;
        public TwinBrainManager   brainManager;
        public TimelinePlayer     timeline;

        [Header("UI Elements")]
        public Image    statusDot;          // green / red indicator
        public TMP_Text statusText;
        public TMP_Text backendLabel;
        public TMP_Text selectionLabel;
        public TMP_Text dataSourceLabel;

        [Header("Colors")]
        public Color connectedColor    = new Color(0.27f, 1f, 0.53f);
        public Color disconnectedColor = new Color(1f,   0.27f, 0.40f);

        private int _lastSelectedCount = -1;

        private void Start()
        {
            if (wsClient    == null) wsClient    = FindObjectOfType<TwinBrainWebSocket>();
            if (brainManager == null) brainManager = FindObjectOfType<TwinBrainManager>();
            if (timeline    == null) timeline    = FindObjectOfType<TimelinePlayer>();

            if (wsClient != null)
            {
                wsClient.OnConnected    += () => SetConnected(true);
                wsClient.OnDisconnected += () => SetConnected(false);
                wsClient.OnServerVersion += ver =>
                {
                    if (statusText != null) statusText.text = $"已连接 (v{ver})";
                };
            }

            if (timeline != null)
                timeline.OnFrameChanged += (cur, total) =>
                {
                    if (dataSourceLabel != null)
                        dataSourceLabel.text = $"{cur + 1} / {total} 帧";
                };

            SetConnected(false);
        }

        private void Update()
        {
            // Only refresh when selection count changes
            int selCount = brainManager?.GetSelectedRegions().Count ?? 0;
            if (selCount != _lastSelectedCount)
            {
                _lastSelectedCount = selCount;
                if (selectionLabel != null) selectionLabel.text = $"已选: {selCount}";
            }
        }

        private void SetConnected(bool connected)
        {
            if (statusDot  != null) statusDot.color = connected ? connectedColor : disconnectedColor;
            if (statusText != null) statusText.text  = connected ? "已连接后端" : "演示模式（后端未连接）";
            if (backendLabel != null) backendLabel.text = connected ? "已连接" : "未连接";
        }
    }
}
