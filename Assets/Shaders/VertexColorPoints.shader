Shader "UnityScanLab/VertexColorPoints"
{
    Properties
    {
        _PointSize("Point Size", Float) = 0.01
        _Softness("Edge Softness", Range(0, 1)) = 0.3
        _MainColor("Base Color", Color) = (0.5, 0.5, 0.5, 1.0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
            };

            struct v2g
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float size : TEXCOORD0;
            };

            struct g2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            float _PointSize;
            float _Softness;
            float4 _MainColor;

            v2g vert (appdata v)
            {
                v2g o;
                o.pos = v.vertex; // Keep in object space
                o.color = v.color;
                o.size = _PointSize;
                return o;
            }

            [maxvertexcount(4)]
            void geom(point v2g input[1], inout TriangleStream<g2f> outStream)
            {
                v2g i = input[0];
                float halfS = i.size * 0.5f;

                // Transform to View Space to create perfectly camera-facing billboards
                float4 viewPos = mul(UNITY_MATRIX_MV, i.pos);

                g2f o;
                o.color = i.color;

                // Top Right
                o.pos = mul(UNITY_MATRIX_P, viewPos + float4(halfS, halfS, 0, 0));
                o.uv = float2(1, 1);
                outStream.Append(o);
                
                // Bottom Right
                o.pos = mul(UNITY_MATRIX_P, viewPos + float4(halfS, -halfS, 0, 0));
                o.uv = float2(1, -1);
                outStream.Append(o);
                
                // Top Left
                o.pos = mul(UNITY_MATRIX_P, viewPos + float4(-halfS, halfS, 0, 0));
                o.uv = float2(-1, 1);
                outStream.Append(o);
                
                // Bottom Left
                o.pos = mul(UNITY_MATRIX_P, viewPos + float4(-halfS, -halfS, 0, 0));
                o.uv = float2(-1, -1);
                outStream.Append(o);
            }

            fixed4 frag (g2f i) : SV_Target
            {
                float distSq = dot(i.uv, i.uv);
                
                // Cut the square into a perfect flat circle
                // We do NOT add fake 3D shading here, otherwise it looks like "big balls"
                if (distSq > 1.0) discard;

                // Pure flat color for authentic surface splatting
                return fixed4(i.color.rgb, 1.0);
            }
            ENDCG
        }
    }
}
