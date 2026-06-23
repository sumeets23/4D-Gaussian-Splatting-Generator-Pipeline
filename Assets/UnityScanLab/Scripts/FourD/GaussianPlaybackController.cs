using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace UnityScanLab.FourD
{
    /// <summary>
    /// Drives 4D Gaussian Splat playback.
    ///
    /// Streams PLY frame data to GPU ComputeBuffers and renders them each frame
    /// via Graphics.DrawProcedural using the FourDGaussianSplat geometry shader.
    ///
    /// Based on the 4C4D / UnityGaussianSplatting rendering pattern:
    ///   https://github.com/yangzf-1023/4C4D
    ///   https://github.com/aras-p/UnityGaussianSplatting
    /// </summary>
    [ExecuteAlways]
    public class GaussianPlaybackController : MonoBehaviour
    {
        private const int MinSmoothFrameCache = 4;
        private const float RemovedSplatPosition = 1000000f;

        [Header("Animation Settings")]
        public float playbackSpeed = 1.0f;
        public bool  loop          = true;
        [Tooltip("Set to true to animate. False freezes on current frame.")]
        public bool  isPlaying     = true;

        [Tooltip("Automatically start playback when entering Play Mode. Edit Mode stays paused until Play is pressed in the tool.")]
        public bool autoPlayOnPlayMode = true;

        [Header("Data Source")]
        [Tooltip("GaussianFrame ScriptableObjects â€” one per time step.")]
        public List<GaussianFrame> sequenceFrames = new List<GaussianFrame>();

        [Header("VRAM Management")]
        [Tooltip("How many frames to keep decoded in CPU RAM at once. Keep low for 8GB VRAM setups.")]
        public int activeWindowSize = 5;

        [Header("Rendering")]
        [Tooltip("Aras-p GaussianSplatRenderer instance. If assigned, this controller streams data into it.")]
        [HideInInspector]
        public GaussianSplatting.Runtime.GaussianSplatRenderer targetRenderer;

        [Tooltip("Use UnityScanLab's per-frame dynamic renderer for 4D playback. The Aras asset is kept only as a static first-frame reference.")]
        public bool useUnityScanLabDynamicRenderer = true;

        [Tooltip("Compute shader used to interpolate and format splat data for the target renderer.")]
        [HideInInspector]
        public ComputeShader temporalInterpolator;

        [Tooltip("Material using the UnityScanLab/FourDGaussianSplat shader. Auto-created if null (used only for fallback).")]
        [HideInInspector]
        public Material customRenderMaterial;

        [Header("Editor Playback Performance")]
        [Tooltip("Maximum splats drawn by the UnityScanLab fallback renderer. Lower this if Play Mode FPS is low. Set 0 for full quality.")]
        public int maxRenderedSplats = 0;

        [Tooltip("Render 4D splats in Scene View while Play Mode is running. Disable for faster editor playback.")]
        public bool renderInSceneViewDuringPlay = false;

        [Tooltip("Use the capped UnityScanLab renderer during Play Mode instead of the full-quality renderer.")]
        public bool preferFastEditorPlayback = false;

        [Tooltip("Automatically use the capped renderer in Editor Play Mode. This does not affect player builds or the stored Gaussian data.")]
        public bool optimizeEditorPlayback = true;

        [Tooltip("Maximum splats rendered in Editor Play Mode when automatic optimization is enabled.")]
        public int editorSplatBudget = 60000;

        [Tooltip("Center static 3DGS data around the prefab origin at render time. Leave disabled for 4D sequences with authored coordinates.")]
        public bool autoCenterBounds = false;

        [Tooltip("Maximum 4D frame uploads/interpolation updates per second in Play Mode. Lower values keep the editor responsive.")]
        public float playModeUpdateRate = 24f;

        [Tooltip("SH degree actually present in the generated sequence. Runtime rendering never evaluates a higher degree.")]
        public int sourceSHDegree = 1;

        [Tooltip("Minimum rendered-frame interval between depth sorts. Four is a practical balance for animated splats.")]
        public int minimumSortInterval = 4;

        [Tooltip("Spherical harmonics order used by GaussianSplatRenderer in Play Mode. 0 is fastest.")]
        public int playModeSHOrder = 3;

        [Tooltip("Sort Gaussian splats every N rendered frames in Play Mode. Higher values improve editor FPS.")]
        public int playModeSortNthFrame = 1;

        [Tooltip("Use GaussianSplatRenderer debug point mode in Play Mode. This is much faster for editor playback.")]
        public bool playModeUsePointRenderer = false;

        [Tooltip("Point size used when Play Mode point renderer is enabled.")]
        public float playModePointSize = 2f;

        [Header("Runtime Status")]
        public int currentFrameIndex = -1;
        public int currentSplatCount = 0;
        public int renderedSplatCount = 0;
        public bool runtimeBuffersReady = false;
        public string playbackStatus = "Not initialized";

        // â”€â”€ Exposed GPU buffers (can be read by external renderers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        public ComputeBuffer PositionBuffer => _bufferA?.positions;
        public ComputeBuffer RotationBuffer => _bufferA?.rotations;
        public ComputeBuffer ScaleBuffer    => _bufferA?.scales;
        public ComputeBuffer ColorBuffer    => _bufferA?.colors;
        public ComputeBuffer OpacityBuffer  => _bufferA?.opacities;

        private GPUFrameBuffer _bufferA = new GPUFrameBuffer();
        private GPUFrameBuffer _bufferB = new GPUFrameBuffer();
        private int _loadedIndexA = -1;
        private int _loadedIndexB = -1;

        private float   _playbackTime    = 0f;
        private int     _currentFrame    = -1;
        private float   _maxTime         = 0f;
        private int     _lastSplatCount  = 0;
        private float   _lastEditorTime  = -1f;
        private Material _runtimeMat;
        private bool _frameBound = false;
        private bool _requestedPausedSceneRepaint = false;
        private bool _hasRenderPositionOffset = false;
        private Vector3 _renderPositionOffset = Vector3.zero;
        private int _residentIndexA = -1;
        private int _residentIndexB = -1;
        private int _residentWindowStart = -1;
        private int _residentWindowEnd = -1;
        private float _lastPlayModeFrameUpdateTime = -1f;

        private System.Reflection.FieldInfo _fiPosData;
        private System.Reflection.FieldInfo _fiOtherData;
        private System.Reflection.FieldInfo _fiColorData;
        private System.Reflection.FieldInfo _fiChunksValid;
        private int _csKernelInterpolate = -1;
        // True only when the temporal interpolator compiled AND the kernel index is valid
        private bool _kernelValid = false;
        private RenderTexture _dynamicColorTexture;
        private Texture _originalRendererColorTexture;
        private int _dynamicColorSplatCount = -1;
        private bool _fallbackCallbacksSubscribed;
        private bool _nativeFailureLogged;
        private int _nativeReadinessMisses;
        private bool _streamingBlocked;

        private class GPUFrameBuffer
        {
            public ComputeBuffer positions;
            public ComputeBuffer rotations;
            public ComputeBuffer scales;
            public ComputeBuffer colors;
            public ComputeBuffer opacities;
            public int splatCount;

            public void Release()
            {
                positions?.Release(); positions = null;
                rotations?.Release(); rotations = null;
                scales?.Release();    scales = null;
                colors?.Release();    colors = null;
                opacities?.Release(); opacities = null;
                splatCount = 0;
            }
        }

        private MeshFilter _meshFilter;
        private MeshRenderer _meshRenderer;

        // â”€â”€ Unity Callbacks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        private void OnEnable()
        {
            UpgradeLegacyLowQualityDefaults();
            activeWindowSize = Mathf.Max(activeWindowSize, MinSmoothFrameCache);
            if (maxRenderedSplats < 0)
                maxRenderedSplats = 0;
            ClampPerformanceSettings();

            _meshFilter = GetComponent<MeshFilter>();
            _meshRenderer = GetComponent<MeshRenderer>();
            if (_meshRenderer != null)
            {
                _meshRenderer.enabled = false;
            }

            if (Application.isPlaying && autoPlayOnPlayMode)
                isPlaying = true;

            SetFallbackCallbacks(ShouldUseExplicitFallback());

            if (sequenceFrames != null && sequenceFrames.Count > 0)
                InitIfReady();

#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                EditorApplication.update += EditorUpdate;
            }
#endif
        }

        private void OnDisable()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                EditorApplication.update -= EditorUpdate;
            }
#endif
            SetFallbackCallbacks(false);
            ReleaseBuffers();
            UnloadAllFrames();
            if (_meshFilter != null)
            {
                _meshFilter.sharedMesh = null;
            }
        }

        private void OnDestroy()
        {
            ReleaseBuffers();
            if (_runtimeMat != null)
            {
#if UNITY_EDITOR
                DestroyImmediate(_runtimeMat);
#else
                Destroy(_runtimeMat);
#endif
            }
            ReleaseDynamicColorTexture();
        }

        private void OnValidate()
        {
            UpgradeLegacyLowQualityDefaults();
            activeWindowSize = Mathf.Max(activeWindowSize, MinSmoothFrameCache);
            if (maxRenderedSplats < 0)
                maxRenderedSplats = 0;
            ClampPerformanceSettings();
        }

        private void Update()
        {
            // Lazily initialize if frames were assigned after OnEnable
            if (_lastSplatCount == 0 && sequenceFrames.Count > 0)
                InitIfReady();

            if (sequenceFrames.Count == 0)
            {
                playbackStatus = "No sequence frames assigned";
                return;
            }

            if (!_frameBound)
            {
                if (ShouldUpdatePlayModeFrame())
                    UpdateFrame();
            }

            if (!Application.isPlaying && !isPlaying)
            {
                playbackStatus = $"Paused frame {currentFrameIndex + 1}/{sequenceFrames.Count} ({currentSplatCount:n0} splats, rendering {renderedSplatCount:n0})";
#if UNITY_EDITOR
                if (!_requestedPausedSceneRepaint && runtimeBuffersReady)
                {
                    _requestedPausedSceneRepaint = true;
                    SceneView.RepaintAll();
                }
#endif
                return;
            }

            // â”€â”€ Timing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Only advance time automatically in Play mode. In Edit mode, EditorUpdate handles it.
            if (isPlaying && Application.isPlaying)
            {
                // Pause the clock while streaming, but keep polling UpdateFrame so
                // the completed asynchronous load can release the hold.
                if (!_streamingBlocked)
                {
                    _playbackTime += Time.deltaTime * playbackSpeed;

                    if (_playbackTime > _maxTime)
                    {
                        if (loop)
                        {
                            _playbackTime = 0f;
                            _currentFrame = -1;
                        }
                        else
                        {
                            _playbackTime = _maxTime;
                            isPlaying = false;
                        }
                    }
                }

                if (ShouldUpdatePlayModeFrame())
                    UpdateFrame();
            }

            // â”€â”€ Render matrix update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            Material mat = GetActiveMaterial();
            if (mat != null)
            {
                mat.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);
            }
        }
#if UNITY_EDITOR
        private void EditorUpdate()
        {
            if (!Application.isPlaying && isPlaying)
            {
                if (_lastSplatCount == 0 && sequenceFrames.Count > 0)
                    InitIfReady();

                if (sequenceFrames.Count == 0)
                    return;

                // Advance time manually in Edit mode without flooding the Player Loop queue
                float dt = GetEditorDeltaTime();
                _playbackTime += dt * playbackSpeed;

                if (_playbackTime > _maxTime)
                {
                    if (loop)
                    {
                        _playbackTime = 0f;
                        _currentFrame = -1;
                    }
                    else
                    {
                        _playbackTime = _maxTime;
                        isPlaying = false;
                    }
                }

                UpdateFrame();
                
                Material mat = GetActiveMaterial();
                if (mat != null)
                {
                    mat.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);
                }

                SceneView.RepaintAll();
            }
        }
#endif


        // â”€â”€ Initialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        private void InitIfReady()
        {
            if (sequenceFrames == null) return;
            sequenceFrames.RemoveAll(f => f == null);
            if (sequenceFrames.Count == 0)
            {
                playbackStatus = "No valid sequence frames";
                return;
            }

            sequenceFrames.Sort((a, b) => {
                if (a == null && b == null) return 0;
                if (a == null) return 1;
                if (b == null) return -1;
                return a.Timestamp.CompareTo(b.Timestamp);
            });
            _maxTime = sequenceFrames[sequenceFrames.Count - 1].Timestamp;

            _loadedIndexA = -1;
            _loadedIndexB = -1;
            _currentFrame = -1;
            _frameBound = false;
            _requestedPausedSceneRepaint = false;
            _hasRenderPositionOffset = false;
            _renderPositionOffset = Vector3.zero;
            _residentWindowStart = -1;
            _residentWindowEnd = -1;

            if (targetRenderer == null)
            {
                targetRenderer = GetComponent<GaussianSplatting.Runtime.GaussianSplatRenderer>();
            }

            if (targetRenderer != null)
            {
                var type = targetRenderer.GetType();
                _fiPosData = type.GetField("m_GpuPosData", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                _fiOtherData = type.GetField("m_GpuOtherData", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                _fiColorData = type.GetField("m_GpuColorData", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                _fiChunksValid = type.GetField("m_GpuChunksValid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

                if (Application.isPlaying)
                {
                    int runtimeSHOrder = Application.isEditor ? 0 : Mathf.Min(playModeSHOrder, sourceSHDegree);
                    int runtimeSortInterval = Application.isEditor ? 30 : Mathf.Max(playModeSortNthFrame, minimumSortInterval);
                    targetRenderer.m_SHOrder = runtimeSHOrder;
                    targetRenderer.m_SortNthFrame = runtimeSortInterval;
                    targetRenderer.enabled = true;
                    targetRenderer.m_PointDisplaySize = playModePointSize;
                    targetRenderer.m_RenderMode = GaussianSplatting.Runtime.GaussianSplatRenderer.RenderMode.Splats;
                }
            }


            SetFallbackCallbacks(ShouldUseExplicitFallback());

            if (temporalInterpolator != null)
            {
                try
                {
                    _csKernelInterpolate = temporalInterpolator.FindKernel("CSInterpolateSplats");
                    // Probe thread group sizes â€” this throws if the kernel didn't compile on this platform
                    temporalInterpolator.GetKernelThreadGroupSizes(_csKernelInterpolate, out _, out _, out _);
                    _kernelValid = true;
                }
                catch
                {
                    _csKernelInterpolate = -1;
                    _kernelValid = false;
                    Debug.LogWarning("[4DGS] SplatTemporalInterpolator kernel could not be compiled on this platform. " +
                                     "The native renderer will remain enabled with its static template frame.");
                }
            }

            // Ensure material is on the MeshRenderer (may be null if prefab was saved
            // before Update ran, or after a domain reload).
            if (_meshRenderer != null && _meshRenderer.sharedMaterial == null)
            {
                Material mat = GetActiveMaterial();
                if (mat != null)
                    _meshRenderer.sharedMaterial = mat;
            }

            UpdateFrame();
        }

        // â”€â”€ Frame Management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        private void UpdateFrame()
        {
            if (sequenceFrames.Count == 0) return;

            // 1. Determine indexA and indexB based on current playback time
            int indexA = 0;
            for (int i = 0; i < sequenceFrames.Count; i++)
            {
                if (sequenceFrames[i].Timestamp <= _playbackTime)
                {
                    indexA = i;
                }
                else
                {
                    break;
                }
            }

            int indexB = indexA;
            float interpolationFactor = 0f;

            if (indexA < sequenceFrames.Count - 1)
            {
                indexB = indexA + 1;
                float tA = sequenceFrames[indexA].Timestamp;
                float tB = sequenceFrames[indexB].Timestamp;
                float diff = tB - tA;
                interpolationFactor = (diff > 0.0001f) ? (_playbackTime - tA) / diff : 0f;
                interpolationFactor = Mathf.Clamp01(interpolationFactor);
            }

            // 2. Keep a small rolling CPU cache and never block the main thread on disk I/O.
            UpdateResidentFrames(indexA, indexB);
            GaussianFrame frameA = sequenceFrames[indexA];
            GaussianFrame frameB = sequenceFrames[indexB];
            bool frameAReady = frameA.TryApplyAsyncLoad();
            bool frameBReady = indexA == indexB || frameB.TryApplyAsyncLoad();
            if (!frameAReady || !frameBReady)
            {
                if (!frameAReady) frameA.BeginLoadDataAsync();
                if (!frameBReady) frameB.BeginLoadDataAsync();
                _streamingBlocked = true;
                playbackStatus = $"Streaming frame {indexA + 1}/{sequenceFrames.Count}";
                return;
            }

            _streamingBlocked = false;
            _currentFrame = indexA;
            currentFrameIndex = indexA;
            EnsureRenderPositionOffset(frameA);

            // 3. Perform Buffer Swapping if playhead crossed boundaries to avoid re-uploading
            if (_loadedIndexB == indexA)
            {
                SwapBuffers();
            }
            else if (_loadedIndexA == indexB)
            {
                SwapBuffers();
            }

            // 4. Upload missing frame data to GPU
            if (_loadedIndexA != indexA)
            {
                if (!UploadFrameToGPU(indexA, _bufferA)) return;
                _loadedIndexA = indexA;
            }
            if (_loadedIndexB != indexB)
            {
                if (!UploadFrameToGPU(indexB, _bufferB)) return;
                _loadedIndexB = indexB;
            }

            // 5. Validate frame A buffers before binding render data.
            int countA = _bufferA.splatCount;
            int countB = _bufferB.splatCount;

            if (countA <= 0 ||
                _bufferA.positions == null ||
                _bufferA.rotations == null ||
                _bufferA.scales == null ||
                _bufferA.colors == null ||
                _bufferA.opacities == null)
            {
                _frameBound = false;
                runtimeBuffersReady = false;
                playbackStatus = $"Frame {indexA} buffers are not ready";
                if (_meshRenderer != null)
                {
                    _meshRenderer.enabled = false;
                }
                return;
            }

            _lastSplatCount = countA;
            currentSplatCount = countA;
            runtimeBuffersReady = true;

            // 6. Compute Shader Integration for Aras-p Renderer
            // Only dispatch if: kernel compiled OK, renderer has allocated its GPU buffers
            GraphicsBuffer dstPos = null;
            GraphicsBuffer dstOther = null;
            bool canUseTargetRenderer = false;

            bool forceFastRenderer = ShouldUseExplicitFallback();
            SetFallbackCallbacks(forceFastRenderer);

            if (!forceFastRenderer && targetRenderer != null && !targetRenderer.enabled)
                targetRenderer.enabled = true;

            if (!forceFastRenderer && targetRenderer != null && temporalInterpolator != null && _kernelValid && targetRenderer.HasValidRenderSetup)
            {
                dstPos = _fiPosData?.GetValue(targetRenderer) as GraphicsBuffer;
                dstOther = _fiOtherData?.GetValue(targetRenderer) as GraphicsBuffer;
                canUseTargetRenderer =
                    dstPos != null &&
                    dstOther != null &&
                    EnsureDynamicColorTexture(countA) &&
                    targetRenderer.splatCount == countA &&
                    countA > 0;
            }

            if (canUseTargetRenderer)
            {
                _nativeReadinessMisses = 0;
                _nativeFailureLogged = false;
                targetRenderer.enabled = true;
                targetRenderer.m_RenderMode = GaussianSplatting.Runtime.GaussianSplatRenderer.RenderMode.Splats;

                if (_meshRenderer != null && _meshRenderer.enabled)
                {
                    _meshRenderer.enabled = false;
                }

                ComputeBuffer posB   = (countA == countB) ? _bufferB.positions : _bufferA.positions;
                ComputeBuffer rotB   = (countA == countB) ? _bufferB.rotations : _bufferA.rotations;
                ComputeBuffer scaleB = (countA == countB) ? _bufferB.scales    : _bufferA.scales;

                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_PosA",    _bufferA.positions);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_PosB",    posB);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_RotA",    _bufferA.rotations);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_RotB",    rotB);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_ScaleA",  _bufferA.scales);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_ScaleB",  scaleB);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_ColorA",  _bufferA.colors);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_ColorB",  (countA == countB) ? _bufferB.colors : _bufferA.colors);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_OpacityA",_bufferA.opacities);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_OpacityB",(countA == countB) ? _bufferB.opacities : _bufferA.opacities);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_DstPos",  dstPos);
                temporalInterpolator.SetBuffer(_csKernelInterpolate, "_DstOther",dstOther);
                temporalInterpolator.SetTexture(_csKernelInterpolate, "_DstColor", _dynamicColorTexture);

                float actualFactor = (countA == countB) ? interpolationFactor : 0f;
                temporalInterpolator.SetFloat("_InterpolationFactor", actualFactor);
                temporalInterpolator.SetInt("_SplatCount", countA);

                int threadGroups = (countA + 1023) / 1024;
                temporalInterpolator.Dispatch(_csKernelInterpolate, threadGroups, 1, 1);
            }
            else
            {
                if (!forceFastRenderer)
                {
                    HandleNativeRendererNotReady(countA);
                    return;
                }

                if (_meshRenderer != null && _meshRenderer.enabled)
                {
                    _meshRenderer.enabled = false;
                }

                // Do not submit the full splat set while the capped Editor renderer is active.
                if (targetRenderer != null && targetRenderer.enabled)
                    targetRenderer.enabled = false;

                // Fallback: Bind buffers and set interpolation properties on the material
                Material mat = GetActiveMaterial();
                if (mat != null)
                {
                    mat.SetInt("_SplatCount", countA);

                    // If splat count differs, disable interpolation to avoid out-of-bounds reads on B
                    if (countA != countB)
                    {
                        interpolationFactor = 0f;
                        mat.SetBuffer("_PositionBufferA", _bufferA.positions);
                        mat.SetBuffer("_PositionBufferB", _bufferA.positions);
                        mat.SetBuffer("_RotationBufferA", _bufferA.rotations);
                        mat.SetBuffer("_RotationBufferB", _bufferA.rotations);
                        mat.SetBuffer("_ScaleBufferA",    _bufferA.scales);
                        mat.SetBuffer("_ScaleBufferB",    _bufferA.scales);
                        mat.SetBuffer("_ColorBufferA",    _bufferA.colors);
                        mat.SetBuffer("_ColorBufferB",    _bufferA.colors);
                        mat.SetBuffer("_OpacityBufferA",  _bufferA.opacities);
                        mat.SetBuffer("_OpacityBufferB",  _bufferA.opacities);
                    }
                    else
                    {
                        mat.SetBuffer("_PositionBufferA", _bufferA.positions);
                        mat.SetBuffer("_PositionBufferB", _bufferB.positions);
                        mat.SetBuffer("_RotationBufferA", _bufferA.rotations);
                        mat.SetBuffer("_RotationBufferB", _bufferB.rotations);
                        mat.SetBuffer("_ScaleBufferA",    _bufferA.scales);
                        mat.SetBuffer("_ScaleBufferB",    _bufferB.scales);
                        mat.SetBuffer("_ColorBufferA",    _bufferA.colors);
                        mat.SetBuffer("_ColorBufferB",    _bufferB.colors);
                        mat.SetBuffer("_OpacityBufferA",  _bufferA.opacities);
                        mat.SetBuffer("_OpacityBufferB",  _bufferB.opacities);
                    }

                    mat.SetFloat("_InterpolationFactor", interpolationFactor);

                }
            }

            _frameBound = true;
            playbackStatus = $"Playing frame {indexA + 1}/{sequenceFrames.Count} ({countA:n0} splats, rendering {GetRenderSplatCount():n0})";
        }

        private void ClampPerformanceSettings()
        {
            editorSplatBudget = Mathf.Max(1000, editorSplatBudget);
            playModeUpdateRate = Mathf.Max(1f, playModeUpdateRate);
            playModeSHOrder = Mathf.Clamp(playModeSHOrder, 0, 3);
            sourceSHDegree = Mathf.Clamp(sourceSHDegree, 0, 3);
            playModeSortNthFrame = Mathf.Clamp(playModeSortNthFrame, 1, 30);
            minimumSortInterval = Mathf.Clamp(minimumSortInterval, 1, 30);
            playModePointSize = Mathf.Clamp(playModePointSize, 1f, 30f);
        }

        private void UpgradeLegacyLowQualityDefaults()
        {
            bool hasLegacyImporterSettings =
                maxRenderedSplats == 30000 &&
                !preferFastEditorPlayback &&
                Mathf.Approximately(playModeUpdateRate, 4f) &&
                playModeSHOrder == 0 &&
                playModeSortNthFrame == 30 &&
                playModeUsePointRenderer;

            if (!hasLegacyImporterSettings)
                return;

            maxRenderedSplats = 0;
            playModeUpdateRate = 24f;
            playModeSHOrder = 3;
            playModeSortNthFrame = 1;
            playModeUsePointRenderer = false;

            GaussianSplatting.Runtime.GaussianSplatRenderer renderer =
                targetRenderer != null
                    ? targetRenderer
                    : GetComponent<GaussianSplatting.Runtime.GaussianSplatRenderer>();

            if (renderer != null)
            {
                renderer.m_SHOrder = 3;
                renderer.m_SortNthFrame = 1;
                renderer.m_RenderMode = GaussianSplatting.Runtime.GaussianSplatRenderer.RenderMode.Splats;
            }
        }

        private bool ShouldUpdatePlayModeFrame()
        {
            if (!Application.isPlaying || !_frameBound)
                return true;

            float minInterval = 1f / Mathf.Max(1f, playModeUpdateRate);
            float now = Time.unscaledTime;

            if (_lastPlayModeFrameUpdateTime >= 0f && now - _lastPlayModeFrameUpdateTime < minInterval)
                return false;

            _lastPlayModeFrameUpdateTime = now;
            return true;
        }

        private void SwapBuffers()
        {
            GPUFrameBuffer temp = _bufferA;
            _bufferA = _bufferB;
            _bufferB = temp;

            int tempIdx = _loadedIndexA;
            _loadedIndexA = _loadedIndexB;
            _loadedIndexB = tempIdx;
        }

        private void UpdateResidentFrames(int indexA, int indexB)
        {
            int frameCount = sequenceFrames.Count;
            int minWindowSize = Mathf.Min(2, frameCount);
            int requestedWindowSize = Mathf.Max(activeWindowSize, MinSmoothFrameCache);
            int windowSize = Mathf.Clamp(requestedWindowSize, minWindowSize, frameCount);
            int prefetchAhead = Mathf.Max(1, windowSize - 2);
            int desiredStart = Mathf.Clamp(indexA - 1, 0, frameCount - 1);
            int desiredEnd = Mathf.Min(frameCount - 1, indexB + prefetchAhead);

            while (desiredEnd - desiredStart + 1 > windowSize)
                desiredStart++;

            if (_residentWindowStart == desiredStart &&
                _residentWindowEnd == desiredEnd &&
                _residentIndexA == indexA &&
                _residentIndexB == indexB)
            {
                return;
            }

            UnloadFramesOutsideWindow(_residentWindowStart, _residentWindowEnd, desiredStart, desiredEnd);

            for (int i = desiredStart; i <= desiredEnd; i++)
            {
                GaussianFrame frame = sequenceFrames[i];
                if (frame == null)
                    continue;

                frame.TryApplyAsyncLoad();
                frame.BeginLoadDataAsync();
            }

            _residentIndexA = indexA;
            _residentIndexB = indexB;
            _residentWindowStart = desiredStart;
            _residentWindowEnd = desiredEnd;
        }

        private void UnloadFramesOutsideWindow(int previousStart, int previousEnd, int newStart, int newEnd)
        {
            if (sequenceFrames == null)
                return;

            if (previousStart < 0 || previousEnd < 0)
                return;

            previousStart = Mathf.Clamp(previousStart, 0, sequenceFrames.Count - 1);
            previousEnd = Mathf.Clamp(previousEnd, previousStart, sequenceFrames.Count - 1);

            for (int i = previousStart; i <= previousEnd; i++)
            {
                if (i >= newStart && i <= newEnd)
                    continue;

                sequenceFrames[i].UnloadData();
            }
        }

        private bool UploadFrameToGPU(int fi, GPUFrameBuffer targetBuffer)
        {
            if (fi < 0 || fi >= sequenceFrames.Count) return false;

            GaussianFrame frame = sequenceFrames[fi];
            if (!frame.TryApplyAsyncLoad())
            {
                frame.BeginLoadDataAsync();
                playbackStatus = $"Streaming frame {fi + 1}/{sequenceFrames.Count}";
                return false;
            }

            if (!frame.Data.IsValid)
            {
                Debug.LogWarning($"[4DGS] Frame {fi} has no valid data.");
                playbackStatus = $"Frame {fi} has no valid data";
                return false;
            }

            int n = frame.Data.Positions.Length;
            if (n != targetBuffer.splatCount || targetBuffer.positions == null)
            {
                targetBuffer.Release();
                targetBuffer.positions = new ComputeBuffer(n, 12);
                targetBuffer.rotations = new ComputeBuffer(n, 16);
                targetBuffer.scales = new ComputeBuffer(n, 12);
                targetBuffer.colors = new ComputeBuffer(n, 16);
                targetBuffer.opacities = new ComputeBuffer(n, 4);
                targetBuffer.splatCount = n;
            }

            targetBuffer.positions.SetData(frame.Data.Positions);
            targetBuffer.rotations.SetData(frame.Data.Rotations);
            targetBuffer.scales.SetData(frame.Data.Covariance);
            targetBuffer.colors.SetData(frame.Data.SHColors);
            targetBuffer.opacities.SetData(frame.Data.Opacity);

            playbackStatus = $"Uploaded frame {fi + 1}/{sequenceFrames.Count} ({n:n0} splats)";
            return true;
        }

#if UNITY_EDITOR
        public void EnableEditorBrushPreview()
        {
            if (targetRenderer == null)
                targetRenderer = GetComponent<GaussianSplatting.Runtime.GaussianSplatRenderer>();

            // Brush editing needs dynamic frame data, but it does not require the
            // slow fallback renderer when the native renderer is available.
            useUnityScanLabDynamicRenderer = targetRenderer == null;
            if (targetRenderer != null)
            {
                targetRenderer.enabled = true;
                targetRenderer.m_RenderMode = GaussianSplatting.Runtime.GaussianSplatRenderer.RenderMode.Splats;
            }
            SetFallbackCallbacks(useUnityScanLabDynamicRenderer);

            if (sequenceFrames == null || sequenceFrames.Count == 0)
                return;

            if (currentFrameIndex < 0)
                currentFrameIndex = 0;

            _loadedIndexA = -1;
            _loadedIndexB = -1;
            _frameBound = false;
            _requestedPausedSceneRepaint = false;
            UpdateFrame();
            SceneView.RepaintAll();
        }

        public void ApplyEditorOpacityMask(bool[] eraseMask)
        {
            if (eraseMask == null || sequenceFrames == null || sequenceFrames.Count == 0)
                return;

            int frameIndex = Mathf.Clamp(currentFrameIndex < 0 ? 0 : currentFrameIndex, 0, sequenceFrames.Count - 1);
            ApplyMaskToFrameData(frameIndex, eraseMask);

            if (_loadedIndexA == frameIndex)
                UploadFrameToGPU(frameIndex, _bufferA);

            if (_loadedIndexB == frameIndex)
                UploadFrameToGPU(frameIndex, _bufferB);

            _frameBound = false;
            _requestedPausedSceneRepaint = false;
            UpdateFrame();
            SceneView.RepaintAll();
        }

        public Vector3 EditorSplatLocalToWorld(Vector3 localPosition)
        {
            return transform.TransformPoint(localPosition + _renderPositionOffset);
        }

        private void ApplyMaskToFrameData(int frameIndex, bool[] eraseMask)
        {
            if (frameIndex < 0 || frameIndex >= sequenceFrames.Count)
                return;

            GaussianFrame frame = sequenceFrames[frameIndex];
            if (frame == null)
                return;

            frame.UnloadData();
            frame.LoadData();
            if (!frame.Data.IsValid || frame.Data.Opacity == null)
                return;

            int count = Mathf.Min(frame.Data.Opacity.Length, eraseMask.Length);
            for (int i = 0; i < count; i++)
            {
                if (eraseMask[i])
                {
                    frame.Data.Opacity[i] = 0f;
                    if (frame.Data.Positions != null && i < frame.Data.Positions.Length)
                        frame.Data.Positions[i] = new Vector3(RemovedSplatPosition, RemovedSplatPosition, RemovedSplatPosition);
                }
            }
        }
#endif

        // â”€â”€ Rendering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        private void DrawSplats()
        {
            if (!CanDrawSplats(out Material mat))
                return;

            if (!mat.SetPass(0))
                return;

            int drawCount = GetRenderSplatCount();
            if (drawCount <= 0)
                return;

            Graphics.DrawProceduralNow(MeshTopology.Points, drawCount, 1);
        }

        private void OnRenderObject()
        {
            // Camera callbacks below handle procedural splat rendering more reliably
            // for prefab roots that do not have a visible MeshRenderer.
        }

        private void OnCameraPostRender(Camera camera)
        {
            DrawSplatsForCamera(camera);
        }

        private void OnEndCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            DrawSplatsForCamera(camera);
        }

        private void DrawSplatsForCamera(Camera currentCamera)
        {
            if (currentCamera == null)
                return;

#if UNITY_EDITOR
            if (Application.isPlaying &&
                currentCamera.cameraType == CameraType.SceneView &&
                !renderInSceneViewDuringPlay)
            {
                return;
            }
#endif

            if (currentCamera.cameraType == CameraType.Preview ||
                currentCamera.cameraType == CameraType.Reflection)
            {
                return;
            }

            DrawSplats();
        }

        private bool CanDrawSplats(out Material mat)
        {
            mat = null;

            bool forceFastRenderer = ShouldUseExplicitFallback();

            if (!_frameBound || (!useUnityScanLabDynamicRenderer && !forceFastRenderer) || _lastSplatCount <= 0)
            {
                return false;
            }

            if (_bufferA.positions == null ||
                _bufferA.rotations == null ||
                _bufferA.scales == null ||
                _bufferA.colors == null ||
                _bufferA.opacities == null ||
                _bufferB.positions == null ||
                _bufferB.rotations == null ||
                _bufferB.scales == null ||
                _bufferB.colors == null ||
                _bufferB.opacities == null)
            {
                return false;
            }

            mat = GetActiveMaterial();
            if (mat == null)
            {
                return false;
            }

            mat.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);
            mat.SetVector("_PositionOffset", _renderPositionOffset);
            mat.SetInt("_SplatStride", GetRenderSplatStride());
            return true;
        }

        private void EnsureRenderPositionOffset(GaussianFrame frame)
        {
            if (_hasRenderPositionOffset)
                return;

            _renderPositionOffset = Vector3.zero;
            _hasRenderPositionOffset = true;

            if (!autoCenterBounds || frame == null || !frame.Data.IsValid)
                return;

            Vector3[] positions = frame.Data.Positions;
            if (positions == null || positions.Length == 0)
                return;

            Vector3 min = positions[0];
            Vector3 max = positions[0];

            for (int i = 1; i < positions.Length; i++)
            {
                Vector3 p = positions[i];
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }

            _renderPositionOffset = -((min + max) * 0.5f);
        }

        private int GetRenderSplatCount()
        {
            if (_lastSplatCount <= 0)
                return 0;

            int splatBudget = GetEffectiveSplatBudget();
            if (splatBudget <= 0 || splatBudget >= _lastSplatCount)
            {
                renderedSplatCount = _lastSplatCount;
                return _lastSplatCount;
            }

            int stride = GetRenderSplatStride();
            renderedSplatCount = Mathf.CeilToInt((float)_lastSplatCount / stride);
            return renderedSplatCount;
        }

        private int GetRenderSplatStride()
        {
            int splatBudget = GetEffectiveSplatBudget();
            if (_lastSplatCount <= 0 || splatBudget <= 0 || splatBudget >= _lastSplatCount)
                return 1;

            return Mathf.Max(1, Mathf.CeilToInt((float)_lastSplatCount / splatBudget));
        }

        private bool ShouldUseExplicitFallback()
        {
            return useUnityScanLabDynamicRenderer || (Application.isPlaying && preferFastEditorPlayback);
        }

        private void SetFallbackCallbacks(bool enabled)
        {
            if (_fallbackCallbacksSubscribed == enabled)
                return;

            Camera.onPostRender -= OnCameraPostRender;
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
            if (enabled)
            {
                Camera.onPostRender += OnCameraPostRender;
                RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
            }

            _fallbackCallbacksSubscribed = enabled;
        }

        private void HandleNativeRendererNotReady(int splatCount)
        {
            _nativeReadinessMisses++;
            string reason = targetRenderer == null ? "renderer reference is missing" :
                temporalInterpolator == null ? "temporal interpolator is missing" :
                !_kernelValid ? "temporal interpolation kernel is unavailable" :
                !targetRenderer.HasValidRenderSetup ? "renderer GPU resources are initializing" :
                targetRenderer.splatCount != splatCount ? $"template has {targetRenderer.splatCount:n0} splats but frame has {splatCount:n0}" :
                "renderer GPU buffers are unavailable";

            playbackStatus = $"Native renderer waiting: {reason}";
            bool permanentFailure = targetRenderer == null || temporalInterpolator == null || !_kernelValid ||
                                    (targetRenderer != null && targetRenderer.HasValidRenderSetup && targetRenderer.splatCount != splatCount);
            if (!_nativeFailureLogged && (permanentFailure || _nativeReadinessMisses >= 120))
            {
                Debug.LogError($"[4DGS] Native Gaussian renderer could not start: {reason}. The renderer remains enabled; automatic fallback is disabled.");
                _nativeFailureLogged = true;
            }
        }

        private int GetEffectiveSplatBudget()
        {
            if (!ShouldUseExplicitFallback())
                return 0;

            int budget = maxRenderedSplats;
            if (Application.isEditor && Application.isPlaying && optimizeEditorPlayback)
            {
                int editorBudget = Mathf.Max(1000, editorSplatBudget);
                budget = budget <= 0 ? editorBudget : Mathf.Min(budget, editorBudget);
            }

            return budget;
        }

        private Material GetActiveMaterial()
        {
            if (_runtimeMat == null)
            {
                if (customRenderMaterial != null)
                {
                    _runtimeMat = new Material(customRenderMaterial);
                    _runtimeMat.name = customRenderMaterial.name + " (Runtime)";
                }
                else
                {
                    Shader shader = Shader.Find("UnityScanLab/FourDGaussianSplat");
                    if (shader == null)
                    {
                        Debug.LogError("[4DGS] Shader 'UnityScanLab/FourDGaussianSplat' not found! Make sure Assets/Shaders/FourDGaussianSplat.shader is in the project.");
                        shader = Shader.Find("Sprites/Default");
                    }

                    if (shader != null)
                    {
                        _runtimeMat = new Material(shader);
                        if (shader.name.Contains("FourDGaussianSplat"))
                        {
                            _runtimeMat.SetFloat("_PointSize",    1.0f);
                            _runtimeMat.SetFloat("_OpacityScale", 1.0f);
                        }
                    }
                }

                if (_runtimeMat != null)
                {
                    _runtimeMat.hideFlags = HideFlags.HideAndDontSave;
                }
            }

            return _runtimeMat;
        }

        // â”€â”€ Utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        private void ReleaseBuffers()
        {
            _bufferA?.Release();
            _bufferB?.Release();
            _lastSplatCount = 0;
            currentSplatCount = 0;
            renderedSplatCount = 0;
            runtimeBuffersReady = false;
            _loadedIndexA = -1;
            _loadedIndexB = -1;
            _residentIndexA = -1;
            _residentIndexB = -1;
            _residentWindowStart = -1;
            _residentWindowEnd = -1;
            _frameBound = false;
            _lastPlayModeFrameUpdateTime = -1f;
            _nativeReadinessMisses = 0;
            _nativeFailureLogged = false;
            _streamingBlocked = false;
            _hasRenderPositionOffset = false;
            _renderPositionOffset = Vector3.zero;
            ReleaseDynamicColorTexture();
        }

        private void UnloadAllFrames()
        {
            if (sequenceFrames == null) return;
            foreach (var f in sequenceFrames)
                if (f != null) f.UnloadData();
        }

        private float GetEditorDeltaTime()
        {
            float now = Time.realtimeSinceStartup;
            float dt  = (_lastEditorTime >= 0f) ? (now - _lastEditorTime) : 0f;
            _lastEditorTime = now;
            return Mathf.Min(dt, 0.1f); // Cap at 100ms to prevent huge jumps
        }

        private bool EnsureDynamicColorTexture(int splatCount)
        {
            if (splatCount <= 0 || targetRenderer == null || _fiColorData == null)
                return false;

            if (_dynamicColorTexture == null || _dynamicColorSplatCount != splatCount)
            {
                ReleaseDynamicColorTexture();

                int width = 2048;
                int height = Mathf.Max(1, (splatCount + width - 1) / width);
                height = ((height + 15) / 16) * 16;

                _dynamicColorTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat)
                {
                    name = "4DGS Dynamic Color",
                    enableRandomWrite = true,
                    useMipMap = false,
                    autoGenerateMips = false,
                    hideFlags = HideFlags.HideAndDontSave
                };

                if (!_dynamicColorTexture.Create())
                {
                    ReleaseDynamicColorTexture();
                    return false;
                }

                _dynamicColorSplatCount = splatCount;
            }

            if (_originalRendererColorTexture == null)
            {
                Texture currentTexture = _fiColorData.GetValue(targetRenderer) as Texture;
                if (currentTexture != _dynamicColorTexture)
                    _originalRendererColorTexture = currentTexture;
            }

            _fiColorData.SetValue(targetRenderer, _dynamicColorTexture);

            // The imported first-frame chunk metadata no longer describes the
            // dynamic positions/colors after interpolation. Disable chunk-based
            // culling/decoding so Aras renders from raw buffers directly.
            _fiChunksValid?.SetValue(targetRenderer, false);

            return true;
        }

        private void ReleaseDynamicColorTexture()
        {
            if (targetRenderer != null && _fiColorData != null && _dynamicColorTexture != null)
            {
                Texture currentTexture = _fiColorData.GetValue(targetRenderer) as Texture;
                if (currentTexture == _dynamicColorTexture)
                    _fiColorData.SetValue(targetRenderer, _originalRendererColorTexture);
            }

            if (_dynamicColorTexture != null)
            {
                _dynamicColorTexture.Release();
#if UNITY_EDITOR
                DestroyImmediate(_dynamicColorTexture);
#else
                Destroy(_dynamicColorTexture);
#endif
                _dynamicColorTexture = null;
            }

            _dynamicColorSplatCount = -1;
            _originalRendererColorTexture = null;
        }
    }
}
