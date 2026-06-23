using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using UnityEngine;

namespace UnityScanLab.FourD
{
    public static class FourDPLYLoader
    {
        public static GaussianFrameData Load(string filePath, bool includeSHRest = false)
        {
            GaussianFrameData data = new GaussianFrameData();
            if (!File.Exists(filePath))
            {
                Debug.LogError($"[4DGS Loader] File not found: {filePath}");
                return data;
            }

            try
            {
                using (FileStream fs = File.OpenRead(filePath))
                {
                    using (BinaryReader br = new BinaryReader(fs))
                    {
                        // Parse header
                        string header = ReadHeader(br, out int vertexCount, out Dictionary<string, int> propOffsets, out int stride);
                        if (vertexCount <= 0 || stride <= 0)
                        {
                            Debug.LogError($"[4DGS Loader] Invalid header in PLY: {filePath}");
                            return data;
                        }

                        data.Positions = new Vector3[vertexCount];
                        data.Rotations = new Quaternion[vertexCount];
                        data.Covariance = new Vector3[vertexCount]; // Using scale_0, scale_1, scale_2
                        data.SHColors = new Color[vertexCount];
                        data.Opacity = new float[vertexCount];
                        data.SHRest = includeSHRest ? new float[vertexCount * 45] : null; // SH degree 3 rest coefficients

                        byte[] buffer = new byte[stride];
                        for (int i = 0; i < vertexCount; i++)
                        {
                            if (br.Read(buffer, 0, stride) < stride)
                            {
                                Debug.LogWarning($"[4DGS Loader] Unexpected EOF in PLY at vertex {i}");
                                break;
                            }

                            try
                            {
                                // Read positions (convert from OpenCV/COLMAP right-handed coordinate system to Unity left-handed)
                                float x = GetFloat(buffer, "x", propOffsets);
                                float y = GetFloat(buffer, "y", propOffsets);
                                float z = GetFloat(buffer, "z", propOffsets);
                                data.Positions[i] = new Vector3(x, -y, z);

                                // Read scaling (covariance representation)
                                float sx = GetFloat(buffer, "scale_0", propOffsets);
                                float sy = GetFloat(buffer, "scale_1", propOffsets);
                                float sz = GetFloat(buffer, "scale_2", propOffsets);
                                data.Covariance[i] = new Vector3(Mathf.Exp(sx), Mathf.Exp(sy), Mathf.Exp(sz));

                                // Read rotation (quaternion)
                                float r0 = GetFloat(buffer, "rot_0", propOffsets); // w
                                float r1 = GetFloat(buffer, "rot_1", propOffsets); // x
                                float r2 = GetFloat(buffer, "rot_2", propOffsets); // y
                                float r3 = GetFloat(buffer, "rot_3", propOffsets); // z
                                data.Rotations[i] = new Quaternion(-r1, r2, -r3, r0).normalized;

                                // Read opacity (sigmoid)
                                float rawOpacity = GetFloat(buffer, "opacity", propOffsets);
                                data.Opacity[i] = 1f / (1f + Mathf.Exp(-rawOpacity));

                                // Read color (SH f_dc_0, f_dc_1, f_dc_2)
                                float sh0 = GetFloat(buffer, "f_dc_0", propOffsets);
                                float sh1 = GetFloat(buffer, "f_dc_1", propOffsets);
                                float sh2 = GetFloat(buffer, "f_dc_2", propOffsets);

                                // Standard SH to RGB conversion for DC component
                                float r = (sh0 * 0.28209479177387814f) + 0.5f;
                                float g = (sh1 * 0.28209479177387814f) + 0.5f;
                                float b = (sh2 * 0.28209479177387814f) + 0.5f;
                                data.SHColors[i] = new Color(Mathf.Clamp01(r), Mathf.Clamp01(g), Mathf.Clamp01(b), data.Opacity[i]);

                                // Read SH rest coefficients (f_rest_0 through f_rest_44) only when requested.
                                for (int j = 0; includeSHRest && j < 45; j++)
                                {
                                    data.SHRest[i * 45 + j] = GetFloat(buffer, $"f_rest_{j}", propOffsets);
                                }
                            }
                            catch (Exception vertexEx)
                            {
                                Debug.LogWarning($"[4DGS Loader] Error parsing vertex {i} in PLY {filePath}: {vertexEx.Message}");
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[4DGS Loader] Error parsing PLY {filePath}: {ex.Message}");
            }

            return data;
        }

        private static string ReadHeader(BinaryReader br, out int vertexCount, out Dictionary<string, int> propOffsets, out int stride)
        {
            vertexCount = 0;
            stride = 0;
            propOffsets = new Dictionary<string, int>();

            StringBuilder sb = new StringBuilder();
            bool inHeader = true;
            int currentOffset = 0;

            while (inHeader)
            {
                string line = "";
                while (br.BaseStream.Position < br.BaseStream.Length)
                {
                    byte b = br.ReadByte();
                    if (b == '\n') break;
                    if (b != '\r') line += (char)b;
                }
                
                line = line.Trim();
                sb.AppendLine(line);

                if (line == "end_header")
                {
                    inHeader = false;
                    break;
                }

                string[] parts = line.Split(' ');
                if (parts[0] == "element" && parts.Length >= 3 && parts[1] == "vertex")
                {
                    int.TryParse(parts[2], out vertexCount);
                }
                else if (parts[0] == "property" && parts.Length >= 3)
                {
                    string propName = parts[parts.Length - 1];
                    string propType = parts[1];
                    int size = GetTypeSize(propType);
                    
                    propOffsets[propName] = currentOffset;
                    currentOffset += size;
                }
            }

            stride = currentOffset;
            return sb.ToString();
        }

        private static int GetTypeSize(string type)
        {
            switch (type)
            {
                case "float": case "float32": case "int": case "uint": return 4;
                case "double": case "float64": return 8;
                case "uchar": case "char": case "uint8": case "int8": return 1;
                case "short": case "ushort": case "int16": return 2;
                default: return 4;
            }
        }

        private static float GetFloat(byte[] buffer, string propName, Dictionary<string, int> offsets)
        {
            if (offsets.TryGetValue(propName, out int offset))
            {
                if (offset + 4 <= buffer.Length)
                {
                    return BitConverter.ToSingle(buffer, offset);
                }
            }
            return 0f;
        }
    }
}
