// TwinBrain – ActivityHeatmap.cs
// Blue→Cyan→Green→Yellow→Red gradient matching the web frontend.

using UnityEngine;

namespace TwinBrain
{
    public static class ActivityHeatmap
    {
        private static readonly Color[] StopColors = {
            new Color(0f,    0.12f, 0.78f), // blue
            new Color(0f,    0.50f, 1.00f), // cyan-blue
            new Color(0f,    0.91f, 0.80f), // cyan-green
            new Color(1f,    0.85f, 0f   ), // yellow
            new Color(1f,    0.16f, 0f   ), // red
        };
        private static readonly float[] StopT = { 0f, 0.25f, 0.5f, 0.75f, 1f };

        /// <summary>
        /// Maps activity v ∈ [0,1] to a heatmap color.
        /// </summary>
        public static Color Evaluate(float v)
        {
            v = Mathf.Clamp01(v);
            for (int i = 1; i < StopT.Length; i++)
            {
                if (v <= StopT[i])
                {
                    float f = (v - StopT[i - 1]) / (StopT[i] - StopT[i - 1]);
                    return Color.Lerp(StopColors[i - 1], StopColors[i], f);
                }
            }
            return StopColors[StopColors.Length - 1];
        }

        /// <summary>
        /// Sets both color and emission on a renderer's first material.
        /// </summary>
        public static void Apply(Renderer rend, float activity, float emissionStrength = 0.25f)
        {
            if (rend == null) return;
            Color c = Evaluate(activity);
            var mat = rend.material;
            mat.color = c;
            if (mat.HasProperty("_EmissionColor"))
            {
                mat.SetColor("_EmissionColor", c * emissionStrength);
                mat.EnableKeyword("_EMISSION");
            }
        }
    }
}
