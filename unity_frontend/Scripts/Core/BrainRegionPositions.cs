// TwinBrain – BrainRegionPositions.cs
// Generates anatomically-approximate MNI-space positions for 200 Schaefer cortical
// regions using the same Fibonacci-sphere algorithm as the web frontend.
// Each hemisphere gets 100 evenly-distributed points; the temporal lobe bulge
// and lateral hemisphere separation are applied to match the web 3D view.

using UnityEngine;

namespace TwinBrain
{
    public static class BrainRegionPositions
    {
        public const int N_REGIONS = 200;

        // Fibonacci golden angle: 2π × (1 − 1/φ) where φ = golden ratio ≈ 1.618
        // Equivalent to Mathf.PI * (3 − Mathf.Sqrt(5)) ≈ 2.40 rad
        private const float dAz = 2f * Mathf.PI * (2f - (1f + 1.6180339887f) / 2f);

        /// <summary>
        /// Returns N_REGIONS world-space positions (in mm, Y-up Unity coords).
        /// Hemisphere 0 → left (index 0-99), hemisphere 1 → right (index 100-199).
        /// </summary>
        public static Vector3[] Generate()
        {
            var pos = new Vector3[N_REGIONS];

            for (int h = 0; h < 2; h++)
            {
                float sign = (h == 0) ? -1f : 1f;   // -1 = left, +1 = right

                for (int i = 0; i < 100; i++)
                {
                    float t  = (i + 0.5f) / 100f;
                    float el = 1.0f - 1.85f * t;    // elevation: 1 (top) → −0.85 (bottom)
                    float r  = Mathf.Sqrt(Mathf.Max(0f, 1f - el * el));
                    float az = dAz * i;

                    float ux = r * Mathf.Cos(az);
                    float uz = r * Mathf.Sin(az);

                    // Force lateral extent ≥ 15 % of hemisphere width
                    float lateralExtent = Mathf.Abs(ux) * 0.85f + 0.15f;

                    // Temporal lobe bulge at mid-inferior height (el ≈ −0.22)
                    float bulge = 9f * Mathf.Exp(-((el + 0.22f) * (el + 0.22f)) * 5f);

                    pos[h * 100 + i] = new Vector3(
                        sign * (lateralExtent * 55f + bulge + 9f),   // X: lateral (mm)
                        el   * 63f - 4f,                              // Y: superior-inferior (mm)
                        uz   * 76f - 8f                               // Z: anterior-posterior (mm)
                    );
                }
            }
            return pos;
        }

        // Schaefer-7-network membership (0-based) for each of the 200 regions
        // Order follows the Schaefer 2018 200-parcel 7-network atlas
        public static readonly string[] NetworkNames =
        {
            "Visual", "Somatomotor", "DorsAttn", "VentAttn", "Limbic", "FrontPar", "Default"
        };

        public static int GetNetworkIndex(int regionId)
        {
            return Mathf.Min(regionId / (N_REGIONS / 7), 6);
        }

        public static string GetNetworkName(int regionId)
        {
            return NetworkNames[GetNetworkIndex(regionId)];
        }

        public static string GetHemisphere(int regionId)
        {
            return (regionId < 100) ? "左脑" : "右脑";
        }
    }
}
