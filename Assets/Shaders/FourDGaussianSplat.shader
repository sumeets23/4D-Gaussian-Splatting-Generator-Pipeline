Shader "UnityScanLab/FourDGaussianSplat"
{
    /*
     * 4D Gaussian Splatting Billboard Shader
     *
     * Renders Gaussian splats as screen-aligned quads with soft radial falloff.
     * Data comes from ComputeBuffers uploaded by GaussianPlaybackController.
     *
     * Aligned with UnityGaussianSplatting rendering approach:
     *   https://github.com/aras-p/UnityGaussianSplatting
     *
     * Property layout:
     *   _PositionBuffer : float3  world positions
     *   _RotationBuffer : float4  quaternion (w,x,y,z)
     *   _ScaleBuffer    : float3  world-space half-extents (already exp'd by loader)
     *   _ColorBuffer    : float4  RGBA (SH DC component -> RGB + opacity as alpha)
     *   _OpacityBuffer  : float   per-splat opacity [0..1]
     */
    Properties
    {
        _PointSize   ("Base Splat Size",   Float)       = 1.0
        _OpacityScale("Opacity Scale",     Range(0, 1)) = 1.0
        _MinSize     ("Min Screen Size",   Float)       = 0.002
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent+1" }
        LOD 100
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex   vert
            #pragma geometry geom
            #pragma fragment frag
            #include "UnityCG.cginc"

            // ── Buffers ───────────────────────────────────────────────────────────
            StructuredBuffer<float3> _PositionBufferA;
            StructuredBuffer<float3> _PositionBufferB;
            StructuredBuffer<float4> _RotationBufferA;
            StructuredBuffer<float4> _RotationBufferB;
            StructuredBuffer<float3> _ScaleBufferA;
            StructuredBuffer<float3> _ScaleBufferB;
            StructuredBuffer<float4> _ColorBufferA;
            StructuredBuffer<float4> _ColorBufferB;
            StructuredBuffer<float>  _OpacityBufferA;
            StructuredBuffer<float>  _OpacityBufferB;
            int      _SplatCount;
            int      _SplatStride;
            float4x4 _LocalToWorld;
            float3   _PositionOffset;

            float _PointSize;
            float _OpacityScale;
            float _MinSize;
            float _InterpolationFactor;

            // ── Structs ───────────────────────────────────────────────────────────
            struct v2g
            {
                float4 worldPos : POSITION;
                float4 color    : COLOR;
                float3 scale    : TEXCOORD0;
                float4 rotation : TEXCOORD1;
            };

            struct g2f
            {
                float4 clipPos : SV_POSITION;
                float4 color   : COLOR;
                float2 uv      : TEXCOORD0;
            };

            // ── Vertex Shader ─────────────────────────────────────────────────────
            v2g vert(uint id : SV_VertexID)
            {
                v2g o;
                int sourceId = (int)id * max(_SplatStride, 1);

                if (sourceId >= _SplatCount)
                {
                    o.worldPos = float4(0,0,0,0);
                    o.color    = float4(0,0,0,0);
                    o.scale    = float3(0,0,0);
                    return o;
                }

                float3 posA   = _PositionBufferA[sourceId];
                float3 posB   = _PositionBufferB[sourceId];
                float3 pos    = lerp(posA, posB, _InterpolationFactor) + _PositionOffset;

                float4 colA   = _ColorBufferA[sourceId];
                float4 colB   = _ColorBufferB[sourceId];
                float4 col    = lerp(colA, colB, _InterpolationFactor);

                float  alphaA = _OpacityBufferA[sourceId];
                float  alphaB = _OpacityBufferB[sourceId];
                float  alpha  = lerp(alphaA, alphaB, _InterpolationFactor);

                float3 scA    = _ScaleBufferA[sourceId];
                float3 scB    = _ScaleBufferB[sourceId];
                float3 sc     = lerp(scA, scB, _InterpolationFactor);

                // Transform position to world space
                o.worldPos = mul(_LocalToWorld, float4(pos, 1.0));
                o.color    = float4(col.rgb, saturate(alpha * _OpacityScale));

                // Scale is stored as actual world-space half-extents (already exp'd by FourDPLYLoader)
                // _PointSize acts as a global multiplier so the user can tune visibility in the editor
                o.scale = max(sc * _PointSize, float3(_MinSize, _MinSize, _MinSize));

                float4 rotA = _RotationBufferA[sourceId];
                float4 rotB = _RotationBufferB[sourceId];
                if (dot(rotA, rotB) < 0.0)
                    rotB = -rotB;
                o.rotation = normalize(lerp(rotA, rotB, _InterpolationFactor));

                return o;
            }

            float3 RotateByQuaternion(float3 v, float4 q)
            {
                float3 t = 2.0 * cross(q.xyz, v);
                return v + q.w * t + cross(q.xyz, t);
            }

            // ── Geometry Shader ───────────────────────────────────────────────────
            // Expands each point into a view-aligned quad (billboard)
            [maxvertexcount(4)]
            void geom(point v2g input[1], inout TriangleStream<g2f> outStream)
            {
                v2g vi = input[0];

                // Discard fully transparent or degenerate splats
                if (vi.color.a < 0.001) return;

                // View-space position of the splat centre
                float4 viewPos = mul(UNITY_MATRIX_V, vi.worldPos);

                float3 localAxisX = RotateByQuaternion(float3(vi.scale.x, 0, 0), vi.rotation);
                float3 localAxisY = RotateByQuaternion(float3(0, vi.scale.y, 0), vi.rotation);
                float3 worldAxisX = mul((float3x3)_LocalToWorld, localAxisX);
                float3 worldAxisY = mul((float3x3)_LocalToWorld, localAxisY);
                float3 viewAxisX = mul((float3x3)UNITY_MATRIX_V, worldAxisX);
                float3 viewAxisY = mul((float3x3)UNITY_MATRIX_V, worldAxisY);

                g2f o;
                o.color = vi.color;

                // Top-Right
                o.clipPos = mul(UNITY_MATRIX_P, viewPos + float4(viewAxisX + viewAxisY, 0));
                o.uv = float2(1, 1);
                outStream.Append(o);

                // Bottom-Right
                o.clipPos = mul(UNITY_MATRIX_P, viewPos + float4(viewAxisX - viewAxisY, 0));
                o.uv = float2(1, -1);
                outStream.Append(o);

                // Top-Left
                o.clipPos = mul(UNITY_MATRIX_P, viewPos + float4(-viewAxisX + viewAxisY, 0));
                o.uv = float2(-1, 1);
                outStream.Append(o);

                // Bottom-Left
                o.clipPos = mul(UNITY_MATRIX_P, viewPos + float4(-viewAxisX - viewAxisY, 0));
                o.uv = float2(-1, -1);
                outStream.Append(o);
            }

            // ── Fragment Shader ───────────────────────────────────────────────────
            fixed4 frag(g2f i) : SV_Target
            {
                // Radial distance from splat centre (uv is [-1..1])
                float distSq = dot(i.uv, i.uv);

                // Discard corners so the quad reads as a circle
                if (distSq > 1.0) discard;

                // Gaussian-like soft falloff: exp(-2*r^2)
                float alpha = exp(-distSq * 2.5) * i.color.a;
                return fixed4(i.color.rgb, alpha);
            }
            ENDCG
        }
    }
    FallBack "Sprites/Default"
}
