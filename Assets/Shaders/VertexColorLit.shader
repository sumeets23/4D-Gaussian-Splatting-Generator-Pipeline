Shader "UnityScanLab/VertexColorLit"
{
    Properties
    {
        _Color ("Tint", Color) = (1, 1, 1, 1)
        _Ambient ("Ambient", Range(0, 1)) = 0.35
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 100

        Pass
        {
            Tags { "LightMode"="UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _Color;
                float _Ambient;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float4 color : COLOR;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                float4 color : COLOR;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                VertexPositionInputs pos = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = pos.positionCS;
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.color = input.color * _Color;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                Light mainLight = GetMainLight();
                float ndotl = saturate(dot(normalize(input.normalWS), normalize(mainLight.direction)));
                float lighting = max(ndotl, _Ambient);
                return half4(input.color.rgb * lighting, input.color.a);
            }
            ENDHLSL
        }
    }
    FallBack "Diffuse"
}
