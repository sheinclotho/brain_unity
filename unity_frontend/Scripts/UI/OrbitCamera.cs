// TwinBrain – OrbitCamera.cs
// Simple orbit camera: left-drag rotates, scroll zooms, middle-drag pans.
// Mirrors the web frontend's orbit controls.

using UnityEngine;

namespace TwinBrain
{
    public class OrbitCamera : MonoBehaviour
    {
        [Header("Orbit")]
        public Vector3 target     = Vector3.zero;
        public float   distance   = 3.2f;
        public float   minDist    = 0.8f;
        public float   maxDist    = 12f;
        public float   rotateSpeed = 0.4f;
        public float   zoomSpeed   = 0.6f;

        [Header("Constraints")]
        public float minPitch = 5f;
        public float maxPitch = 175f;

        private float _theta = 180f;   // yaw  (degrees)
        private float _phi   = 70f;    // pitch (degrees, 90 = equator)

        private void Start()
        {
            // Initialise from current camera pose
            Vector3 offset = transform.position - target;
            distance = offset.magnitude;
            if (distance < 0.01f) distance = 3.2f;
            _theta = Mathf.Atan2(offset.x, offset.z) * Mathf.Rad2Deg;
            _phi   = Mathf.Acos(Mathf.Clamp(offset.y / distance, -1f, 1f)) * Mathf.Rad2Deg;
        }

        private void LateUpdate()
        {
            // Rotate
            if (Input.GetMouseButton(0))
            {
                _theta -= Input.GetAxis("Mouse X") * rotateSpeed * 100f * Time.deltaTime;
                _phi   -= Input.GetAxis("Mouse Y") * rotateSpeed * 100f * Time.deltaTime;
                _phi    = Mathf.Clamp(_phi, minPitch, maxPitch);
            }

            // Zoom
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            distance = Mathf.Clamp(distance - scroll * zoomSpeed * 5f, minDist, maxDist);

            // Apply spherical coordinates
            float rad    = distance;
            float phiR   = _phi   * Mathf.Deg2Rad;
            float thetaR = _theta * Mathf.Deg2Rad;

            Vector3 pos = target + new Vector3(
                rad * Mathf.Sin(phiR) * Mathf.Sin(thetaR),
                rad * Mathf.Cos(phiR),
                rad * Mathf.Sin(phiR) * Mathf.Cos(thetaR)
            );

            transform.position = pos;
            transform.LookAt(target);
        }
    }
}
