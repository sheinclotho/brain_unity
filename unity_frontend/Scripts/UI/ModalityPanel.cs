// TwinBrain – ModalityPanel.cs
// Modality toggle UI: switch between fMRI and EEG visualisation when a
// cache file containing both modalities is loaded.
//
// Wire up:
//   brainManager → the scene's TwinBrainManager
//   fmriButton   → a UnityEngine.UI.Button for "fMRI"
//   eegButton    → a UnityEngine.UI.Button for "EEG"
//   statusLabel  → (optional) TMP_Text showing current modality info
//   panel        → (optional) parent GameObject; hidden when only one modality

using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace TwinBrain
{
    public class ModalityPanel : MonoBehaviour
    {
        [Header("References")]
        public TwinBrainManager brainManager;

        [Header("UI Elements")]
        public Button    fmriButton;
        public Button    eegButton;
        public TMP_Text  statusLabel;
        [Tooltip("Optional container to hide when only one modality is available")]
        public GameObject panel;

        // Button highlight colours
        [Header("Colours")]
        public Color activeColor   = new Color(0.20f, 0.33f, 1.00f, 1f);
        public Color inactiveColor = new Color(0.20f, 0.20f, 0.20f, 1f);

        private string _activeModality = "fmri";

        private void Start()
        {
            if (fmriButton) fmriButton.onClick.AddListener(OnFmriClicked);
            if (eegButton)  eegButton.onClick.AddListener(OnEegClicked);
            Refresh();
        }

        // Call this from TwinBrainManager or a listener when new cache data arrives
        public void Refresh()
        {
            if (brainManager == null) return;

            bool hasFmri = brainManager.CachedFmriFrames != null
                           && brainManager.CachedFmriFrames.Length > 0;
            bool hasEeg  = brainManager.CachedEegFrames  != null
                           && brainManager.CachedEegFrames.Length  > 0;

            // Show panel only when at least one modality is cached
            bool showPanel = hasFmri || hasEeg;
            if (panel) panel.SetActive(showPanel);

            if (fmriButton) fmriButton.interactable = hasFmri;
            if (eegButton)  eegButton.interactable  = hasEeg;

            // Keep selection valid
            if (_activeModality == "eeg" && !hasEeg) _activeModality = "fmri";
            if (_activeModality == "fmri" && !hasFmri) _activeModality = "eeg";

            UpdateButtonStyles();

            if (statusLabel)
            {
                var parts = new System.Collections.Generic.List<string>();
                if (hasFmri) parts.Add($"fMRI {brainManager.CachedFmriFrames.Length}帧");
                if (hasEeg)  parts.Add($"EEG {brainManager.CachedEegFrames.Length}帧");
                statusLabel.text = string.Join(" · ", parts);
            }
        }

        private void OnFmriClicked()
        {
            _activeModality = "fmri";
            UpdateButtonStyles();
            brainManager.SwitchModality("fmri");
        }

        private void OnEegClicked()
        {
            _activeModality = "eeg";
            UpdateButtonStyles();
            brainManager.SwitchModality("eeg");
        }

        private void UpdateButtonStyles()
        {
            SetButtonColor(fmriButton, _activeModality == "fmri" ? activeColor : inactiveColor);
            SetButtonColor(eegButton,  _activeModality == "eeg"  ? activeColor : inactiveColor);
        }

        private static void SetButtonColor(Button btn, Color c)
        {
            if (btn == null) return;
            var colors = btn.colors;
            colors.normalColor = c;
            btn.colors = colors;
        }
    }
}
