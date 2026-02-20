// TwinBrain – RegionTooltip.cs
// Shows a World-Space tooltip when hovering over a brain region.
// Attach to a Canvas set to "World Space" mode.
// Raycast on every frame in Update(); lightweight because sphere colliders
// are primitive and Unity's Physics.Raycast is fast.

using UnityEngine;
using TMPro;

namespace TwinBrain
{
    public class RegionTooltip : MonoBehaviour
    {
        [Header("References")]
        public TwinBrainManager brainManager;
        public TMP_Text         tooltipText;   // assign a TMP text inside a small panel
        public Canvas           tooltipCanvas;

        [Header("Offset from camera near-plane")]
        public float screenOffsetX = 20f;
        public float screenOffsetY = -20f;

        private Camera _cam;

        private void Start()
        {
            _cam = Camera.main;
            if (_cam == null)
            {
                Debug.LogWarning("[RegionTooltip] No MainCamera found. Tooltip disabled.");
                enabled = false;
                return;
            }
            if (brainManager == null) brainManager = FindObjectOfType<TwinBrainManager>();
            if (tooltipCanvas != null) tooltipCanvas.gameObject.SetActive(false);
        }

        private void Update()
        {
            Ray ray = _cam.ScreenPointToRay(Input.mousePosition);

            if (Physics.Raycast(ray, out RaycastHit hit, 2000f))
            {
                var tag = hit.collider.GetComponent<BrainRegionTag>();
                if (tag == null) tag = hit.collider.GetComponentInParent<BrainRegionTag>();

                if (tag != null)
                {
                    ShowTooltip(tag.regionId);
                    return;
                }
            }
            HideTooltip();
        }

        private void ShowTooltip(int id)
        {
            if (tooltipCanvas != null) tooltipCanvas.gameObject.SetActive(true);
            if (tooltipText != null)
            {
                string net  = BrainRegionPositions.GetNetworkName(id);
                string hemi = BrainRegionPositions.GetHemisphere(id);
                tooltipText.text = $"区域 {id + 1}  {net}\n{hemi}";
            }

            // Position the canvas near the cursor in screen space
            if (tooltipCanvas != null && tooltipCanvas.renderMode == RenderMode.ScreenSpaceOverlay)
            {
                var rt = tooltipCanvas.GetComponent<RectTransform>();
                if (rt != null)
                {
                    rt.anchoredPosition = new Vector2(
                        Input.mousePosition.x + screenOffsetX,
                        Input.mousePosition.y + screenOffsetY - Screen.height
                    );
                }
            }
        }

        private void HideTooltip()
        {
            if (tooltipCanvas != null) tooltipCanvas.gameObject.SetActive(false);
        }
    }
}
