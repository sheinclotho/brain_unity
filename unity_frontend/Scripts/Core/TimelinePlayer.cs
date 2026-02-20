// TwinBrain – TimelinePlayer.cs
// Plays back a sequence of ActivityFrame objects at a configurable FPS.
// Receives sequences from TwinBrainWebSocket (simulation_result / cache_loaded).
// UI is driven by TimelineUI.cs which listens to events and calls Scrub/Play/Pause.

using System;
using UnityEngine;

namespace TwinBrain
{
    public class TimelinePlayer : MonoBehaviour
    {
        [Header("Playback")]
        [Tooltip("Frames per second during auto-play")]
        public float fps = 10f;

        // ── State ─────────────────────────────────────────────────────────
        private ActivityFrame[] _frames;
        private int  _current;
        private bool _playing;
        private float _elapsed;

        private TwinBrainManager _manager;

        // ── Events ─────────────────────────────────────────────────────────
        public event Action<int, int> OnFrameChanged;   // (currentFrame, totalFrames)
        public event Action           OnSequenceEnd;

        public bool  IsPlaying   => _playing;
        public int   FrameCount  => _frames?.Length ?? 0;
        public int   CurrentFrame => _current;

        // ── Public API ──────────────────────────────────────────────────────

        /// <summary>Load a new frame sequence and start playing immediately.</summary>
        public void Load(ActivityFrame[] frames, TwinBrainManager manager)
        {
            _frames  = frames;
            _manager = manager;
            _current = 0;
            _elapsed = 0f;
            _playing = true;
            ApplyFrame(_current);
            OnFrameChanged?.Invoke(_current, FrameCount);
        }

        public void Play()
        {
            if (FrameCount == 0) return;
            _playing = true;
        }

        public void Pause()
        {
            _playing = false;
        }

        public void TogglePlayPause()
        {
            if (_playing) Pause(); else Play();
        }

        /// <summary>Jump to a specific frame (0-based).</summary>
        public void Scrub(int frame)
        {
            if (FrameCount == 0) return;
            _current = Mathf.Clamp(frame, 0, FrameCount - 1);
            _elapsed = 0f;
            ApplyFrame(_current);
            OnFrameChanged?.Invoke(_current, FrameCount);
        }

        // ── Unity lifecycle ─────────────────────────────────────────────────

        private void Update()
        {
            if (!_playing || FrameCount == 0) return;

            _elapsed += Time.deltaTime;
            float interval = (fps > 0f) ? 1f / fps : 0.1f;

            while (_elapsed >= interval)
            {
                _elapsed -= interval;
                _current = (_current + 1) % FrameCount;
                ApplyFrame(_current);
                OnFrameChanged?.Invoke(_current, FrameCount);

                if (_current == FrameCount - 1)
                    OnSequenceEnd?.Invoke();
            }
        }

        // ── Internal ────────────────────────────────────────────────────────

        private void ApplyFrame(int idx)
        {
            if (_manager != null && _frames != null && idx < _frames.Length)
                _manager.ApplyActivity(_frames[idx].activity);
        }
    }
}
