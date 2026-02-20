// TwinBrain – TimelineUI.cs
// Connects a UnityUI Slider + Button to the TimelinePlayer.
// Attach to a Canvas GameObject that also has access to TimelinePlayer.

using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace TwinBrain
{
    public class TimelineUI : MonoBehaviour
    {
        [Header("References")]
        public TimelinePlayer timeline;

        [Header("UI Elements")]
        public Slider   timelineSlider;
        public Button   playPauseButton;
        public TMP_Text frameLabel;          // e.g. "12 / 60"
        public TMP_Text playPauseLabel;      // "▶" or "⏸"

        private bool _suppressSliderCallback;

        private void Start()
        {
            if (timeline == null)
                timeline = FindObjectOfType<TimelinePlayer>();

            if (timeline == null)
            {
                Debug.LogError("[TimelineUI] No TimelinePlayer found in scene.");
                enabled = false;
                return;
            }

            timeline.OnFrameChanged += HandleFrameChanged;

            if (timelineSlider != null)
                timelineSlider.onValueChanged.AddListener(OnSliderMoved);

            if (playPauseButton != null)
                playPauseButton.onClick.AddListener(OnPlayPauseClicked);

            RefreshUI();
        }

        private void OnDestroy()
        {
            if (timeline != null)
                timeline.OnFrameChanged -= HandleFrameChanged;
        }

        // ── UI callbacks ─────────────────────────────────────────────────────

        private void OnSliderMoved(float value)
        {
            if (_suppressSliderCallback) return;
            int frame = Mathf.RoundToInt(value);
            timeline.Scrub(frame);
        }

        private void OnPlayPauseClicked()
        {
            timeline.TogglePlayPause();
            RefreshUI();
        }

        // ── Timeline event handler ────────────────────────────────────────────

        private void HandleFrameChanged(int current, int total)
        {
            _suppressSliderCallback = true;

            if (timelineSlider != null)
            {
                timelineSlider.maxValue = Mathf.Max(0, total - 1);
                timelineSlider.value    = current;
            }

            if (frameLabel != null)
                frameLabel.text = total > 0 ? $"{current + 1} / {total}" : "—";

            _suppressSliderCallback = false;
            RefreshPlayPauseButton();
        }

        private void RefreshPlayPauseButton()
        {
            if (playPauseLabel != null)
                playPauseLabel.text = (timeline != null && timeline.IsPlaying) ? "⏸" : "▶";
        }

        private void RefreshUI()
        {
            int total = timeline?.FrameCount ?? 0;

            if (timelineSlider != null)
            {
                timelineSlider.maxValue = Mathf.Max(0, total - 1);
                timelineSlider.value    = timeline?.CurrentFrame ?? 0;
                timelineSlider.interactable = total > 1;
            }

            if (frameLabel != null)
                frameLabel.text = total > 0
                    ? $"{(timeline?.CurrentFrame ?? 0) + 1} / {total}"
                    : "—";

            RefreshPlayPauseButton();
        }
    }
}
