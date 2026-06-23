Shader "UnityScanLab/MeshVertexColor"
{
    Properties
    {
        _Smoothness("Smoothness", Range(0.0, 1.0)) = 0.5
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        Cull Off
        
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _SHADOWS_SOFT
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float3 normalOS     : NORMAL;
                float4 color        : COLOR;
            };

            struct Varyings
            {
                float4 positionCS               : SV_POSITION;
                float3 positionWS               : TEXCOORD0;
                float3 normalWS                 : NORMAL;
                float4 color                    : COLOR;
            };

            float _Smoothness;

            Varyings vert(Attributes input)
            {
                Varyings output;
                
                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS);

                output.positionCS = vertexInput.positionCS;
                output.positionWS = vertexInput.positionWS;
                output.normalWS = normalInput.normalWS;
                output.color = input.color;

                return output;
            }

            half4 frag(Varyings input, FRONT_FACE_TYPE facing : FRONT_FACE_SEMANTIC) : SV_Target
            {
                // Ensure normal is normalized
                float3 normalWS = NormalizeNormalPerPixel(input.normalWS);
                normalWS *= IS_FRONT_VFACE(facing, 1.0, -1.0);

                // Main Light
                Light mainLight = GetMainLight(TransformWorldToShadowCoord(input.positionWS));
                
                // Diffuse
                half NdotL = saturate(dot(normalWS, mainLight.direction));
                half3 diffuse = mainLight.color * NdotL;

                // Ambient
                half3 ambient = SampleSH(normalWS);

                // Combine color
                half3 finalColor = input.color.rgb * (diffuse + ambient);
                
                return half4(finalColor, input.color.a);
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
