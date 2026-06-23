using UnityEngine;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace UnityScanLab.FourD
{
    /// <summary>
    /// Runtime container for one frame's Gaussian splat data.
    /// Arrays are NOT Unity-serialized (too large); instead the PLY file path
    /// is stored in the .asset and data is loaded at runtime via LoadData().
    /// </summary>
    [System.Serializable]
    public struct GaussianFrameData
    {
        [System.NonSerialized] public Vector3[] Positions;
        [System.NonSerialized] public Quaternion[] Rotations;
        [System.NonSerialized] public Vector3[] Covariance;
        [System.NonSerialized] public Color[] SHColors;
        [System.NonSerialized] public float[] SHRest;
        [System.NonSerialized] public float[] Opacity;

        public bool IsValid => Positions != null && Positions.Length > 0;
    }

    /// <summary>
    /// ScriptableObject representing a single 4D Gaussian Splat frame.
    /// Data is streamed on demand from the original PLY file.
    /// </summary>
    public class GaussianFrame : ScriptableObject
    {
        [Tooltip("Frame index in the sequence.")]
        public int FrameIndex;

        [Tooltip("Time in seconds since the start of the animation.")]
        public float Timestamp;

        [Tooltip("Path to the PLY file on disk.")]
        public string plyFilePath;

        [Tooltip("Project-relative path to the generated runtime binary cache for this frame.")]
        public string binaryFramePath;

        [Tooltip("Splat count stored in the generated runtime binary cache.")]
        public int cachedSplatCount;

        [System.NonSerialized]
        public GaussianFrameData Data;

        [System.NonSerialized]
        private Task<GaussianFrameData> _loadTask;

        [System.NonSerialized]
        private string _loadTaskPath;

        public bool IsLoading => _loadTask != null && !_loadTask.IsCompleted;

        public bool IsLoaded => Data.IsValid;

        public bool TryApplyAsyncLoad()
        {
            if (_loadTask == null || !_loadTask.IsCompleted)
                return Data.IsValid;

            try
            {
                Data = _loadTask.Result;
                if (!Data.IsValid && !string.IsNullOrEmpty(_loadTaskPath))
                    Debug.LogWarning($"[4DGS Frame] Failed to async load binary cache: {_loadTaskPath}");
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[4DGS Frame] Error completing async binary load {_loadTaskPath}: {ex.Message}");
            }
            finally
            {
                _loadTask = null;
                _loadTaskPath = null;
            }

            return Data.IsValid;
        }

        public void BeginLoadDataAsync()
        {
            if (Data.IsValid || IsLoading)
                return;

            string binaryPath = ResolvePath(binaryFramePath);
            if (string.IsNullOrEmpty(binaryPath) || !File.Exists(binaryPath))
                return;

            _loadTaskPath = binaryPath;
            _loadTask = Task.Run(() => LoadBinaryFast(binaryPath));
        }

        public bool EnsureLoadStarted()
        {
            if (TryApplyAsyncLoad())
                return true;

            BeginLoadDataAsync();
            return false;
        }

        public void LoadData()
        {
            if (Data.IsValid)
                return;

            if (TryApplyAsyncLoad())
                return;

            string binaryPath = ResolvePath(binaryFramePath);
            if (!string.IsNullOrEmpty(binaryPath) && File.Exists(binaryPath))
            {
                Data = LoadBinaryFast(binaryPath);

                if (!Data.IsValid)
                    Debug.LogWarning($"[4DGS Frame] Failed to load binary cache: {binaryPath}");

                return;
            }

            string path = plyFilePath;

            if (!string.IsNullOrEmpty(path))
            {
                if (!System.IO.File.Exists(path))
                {
                    string relativePath = System.IO.Path.Combine(Application.dataPath, "..", path);

                    if (System.IO.File.Exists(relativePath))
                        path = relativePath;
                }
            }

            if (!string.IsNullOrEmpty(path) && System.IO.File.Exists(path))
            {
                Data = FourDPLYLoader.Load(path);

                if (!Data.IsValid)
                    Debug.LogWarning($"[4DGS Frame] Failed to load PLY: {path}");
            }
            else if (!string.IsNullOrEmpty(path))
            {
                Debug.LogWarning($"[4DGS Frame] PLY file not found: {path} | original: {plyFilePath}");
            }
        }

        public static void SaveBinary(string filePath, GaussianFrameData data)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(filePath));

            using (BinaryWriter writer = new BinaryWriter(File.Open(filePath, FileMode.Create, FileAccess.Write, FileShare.None)))
            {
                int count = data.Positions?.Length ?? 0;
                writer.Write(count);

                for (int i = 0; i < count; i++)
                {
                    Vector3 p = data.Positions[i];
                    writer.Write(p.x);
                    writer.Write(p.y);
                    writer.Write(p.z);

                    Quaternion r = data.Rotations[i];
                    writer.Write(r.x);
                    writer.Write(r.y);
                    writer.Write(r.z);
                    writer.Write(r.w);

                    Vector3 s = data.Covariance[i];
                    writer.Write(s.x);
                    writer.Write(s.y);
                    writer.Write(s.z);

                    Color c = data.SHColors[i];
                    writer.Write(c.r);
                    writer.Write(c.g);
                    writer.Write(c.b);
                    writer.Write(c.a);

                    writer.Write(data.Opacity[i]);
                }
            }
        }

        private static GaussianFrameData LoadBinaryFast(string filePath)
        {
            GaussianFrameData data = new GaussianFrameData();

            try
            {
                byte[] bytes = File.ReadAllBytes(filePath);
                if (bytes.Length < 4)
                    return data;

                int count = System.BitConverter.ToInt32(bytes, 0);
                if (count <= 0)
                    return data;

                const int floatsPerSplat = 15;
                const int bytesPerSplat = floatsPerSplat * sizeof(float);
                int expectedBytes = 4 + count * bytesPerSplat;
                if (bytes.Length < expectedBytes)
                {
                    return data;
                }

                data.Positions = new Vector3[count];
                data.Rotations = new Quaternion[count];
                data.Covariance = new Vector3[count];
                data.SHColors = new Color[count];
                data.Opacity = new float[count];

                ReadOnlySpan<float> values = MemoryMarshal.Cast<byte, float>(bytes.AsSpan(4, count * bytesPerSplat));
                int offset = 0;
                for (int i = 0; i < count; i++)
                {
                    float px = values[offset++];
                    float py = values[offset++];
                    float pz = values[offset++];
                    data.Positions[i] = new Vector3(px, py, pz);

                    float rx = values[offset++];
                    float ry = values[offset++];
                    float rz = values[offset++];
                    float rw = values[offset++];
                    data.Rotations[i] = new Quaternion(rx, ry, rz, rw);

                    float sx = values[offset++];
                    float sy = values[offset++];
                    float sz = values[offset++];
                    data.Covariance[i] = new Vector3(sx, sy, sz);

                    float cr = values[offset++];
                    float cg = values[offset++];
                    float cb = values[offset++];
                    float ca = values[offset++];
                    data.SHColors[i] = new Color(cr, cg, cb, ca);

                    data.Opacity[i] = values[offset++];
                }
            }
            catch (System.Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[4DGS Frame] Error loading binary frame cache {filePath}: {ex.Message}");
            }

            return data;
        }

        private static string ResolvePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return "";

            if (Path.IsPathRooted(path))
                return path;

            string projectPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", path));
            return projectPath;
        }

        public void UnloadData()
        {
            TryApplyAsyncLoad();
            _loadTask = null;
            _loadTaskPath = null;
            Data.Positions = null;
            Data.Rotations = null;
            Data.Covariance = null;
            Data.SHColors = null;
            Data.SHRest = null;
            Data.Opacity = null;
        }
    }
}
