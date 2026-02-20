// TwinBrain – MessageProtocol.cs
// Strongly-typed JSON message types for the TwinBrain WebSocket protocol.
// All messages are serialized with JsonUtility (built-in, no dependencies).

using System;
using System.Collections.Generic;
using UnityEngine;

namespace TwinBrain
{
    // ── Outgoing requests ───────────────────────────────────────────────────

    [Serializable]
    public class GetStateRequest
    {
        public string type = "get_state";
    }

    [Serializable]
    public class SimulateRequest
    {
        public string type          = "simulate";
        public int[]  target_regions;
        public float  amplitude     = 0.5f;
        public string pattern       = "sine";
        public float  frequency     = 10f;
        public int    duration      = 60;
    }

    [Serializable]
    public class LoadCacheRequest
    {
        public string type = "load_cache";
        public string path = null;          // null → server auto-detects
    }

    // ── Incoming responses ──────────────────────────────────────────────────

    /// <summary>
    /// Minimal header; read "type" field first to decide which class to use.
    /// </summary>
    [Serializable]
    public class BaseMessage
    {
        public string type;
        public bool   success;
        public string message;
    }

    /// <summary>
    /// get_state → brain_state response: flat float array, length 200.
    /// </summary>
    [Serializable]
    public class BrainStateMessage
    {
        public string  type;
        public bool    success;
        public float[] activity;           // [0,1] per region
    }

    /// <summary>
    /// simulate → simulation_result / load_cache → cache_loaded:
    /// contains a sequence of frames, each with a flat activity array.
    /// </summary>
    [Serializable]
    public class FrameSequenceMessage
    {
        public string        type;
        public bool          success;
        public int           n_frames;
        public string        path;
        public ActivityFrame[] frames;
    }

    [Serializable]
    public class ActivityFrame
    {
        public float[] activity;           // [0,1] per region, length 200
    }

    /// <summary>
    /// welcome message sent on connect.
    /// </summary>
    [Serializable]
    public class WelcomeMessage
    {
        public string type;
        public string version;
    }
}
