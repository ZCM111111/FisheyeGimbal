#include <metal_stdlib>
using namespace metal;

struct FEUniforms {
    float4 rotation0;
    float4 rotation1;
    float4 rotation2;
    float4 sourceView;
    float4 lens0;
    float4 lens1;
    float4 distortion;
};

struct FEVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex FEVertexOut FisheyeVertex(uint vertexID [[vertex_id]]) {
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    const float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };

    FEVertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}

static float4 sampleSource(texture2d<float> plane0,
                           texture2d<float> plane1,
                           sampler sourceSampler,
                           float2 uv,
                           uint sourceFormat) {
    if (sourceFormat == 0u) {
        float4 bgra = plane0.sample(sourceSampler, uv);
        return float4(bgra.b, bgra.g, bgra.r, 1.0);
    }

    float y = plane0.sample(sourceSampler, uv).r;
    float2 chroma = plane1.sample(sourceSampler, uv).rg;
    float cb;
    float cr;
    float yValue;
    if (sourceFormat == 1u) {
        yValue = y;
        cb = chroma.x - 0.5;
        cr = chroma.y - 0.5;
    } else {
        yValue = (y - 16.0 / 255.0) * (255.0 / 219.0);
        cb = (chroma.x - 128.0 / 255.0) * (255.0 / 224.0);
        cr = (chroma.y - 128.0 / 255.0) * (255.0 / 224.0);
    }

    float red = yValue + 1.5748 * cr;
    float green = yValue - 0.1873 * cb - 0.4681 * cr;
    float blue = yValue + 1.8556 * cb;
    return float4(clamp(float3(red, green, blue), 0.0, 1.0), 1.0);
}

fragment float4 FisheyeFragment(FEVertexOut in [[stage_in]],
                                constant FEUniforms &u [[buffer(0)]],
                                texture2d<float> plane0 [[texture(0)]],
                                texture2d<float> plane1 [[texture(1)]]) {
    constexpr sampler sourceSampler(coord::normalized,
                                    address::clamp_to_edge,
                                    filter::linear,
                                    mip_filter::none);

    float2 sourceSize = max(u.sourceView.xy, float2(1.0));
    float2 viewSize = max(u.sourceView.zw, float2(1.0));
    float2 center = u.lens0.xy;
    float focal = max(u.lens0.z, 0.001);
    float maxRadius = max(u.lens0.w, 0.001);
    float maxTheta = clamp(u.lens1.x, 0.01, 3.13);
    float outputFov = clamp(u.lens1.y, 0.1, 3.05);
    float projection = u.lens1.z;
    float edgeFeather = clamp(u.lens1.w, 0.0, 0.5);
    float k1 = u.distortion.x;
    float k2 = u.distortion.y;
    uint sourceFormat = uint(max(u.distortion.z, 0.0));

    // The output is a pinhole camera in locked-camera coordinates.
    float2 pixel = in.uv * viewSize;
    float focalOut = (viewSize.x * 0.5) / max(tan(outputFov * 0.5), 0.001);
    float2 xy = (pixel - viewSize * 0.5) / focalOut;
    float3 rayLocked = normalize(float3(xy.x, xy.y, 1.0));

    float3x3 cameraFromLocked = float3x3(u.rotation0.xyz,
                                         u.rotation1.xyz,
                                         u.rotation2.xyz);
    float3 raySource = normalize(cameraFromLocked * rayLocked);
    float theta = acos(clamp(raySource.z, -1.0, 1.0));
    if (theta > maxTheta) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float radial = length(raySource.xy);
    float radius;
    if (projection < 0.5) {
        radius = focal * theta;
    } else {
        radius = 2.0 * focal * sin(theta * 0.5);
    }

    float normalizedRadius = radius / maxRadius;
    radius *= 1.0 + k1 * normalizedRadius * normalizedRadius
                    + k2 * normalizedRadius * normalizedRadius
                          * normalizedRadius * normalizedRadius;
    float2 direction = radial > 0.000001 ? raySource.xy / radial : float2(0.0);
    float2 sourcePixel = center + direction * radius;

    if (sourcePixel.x < 0.0 || sourcePixel.y < 0.0 ||
        sourcePixel.x > sourceSize.x || sourcePixel.y > sourceSize.y) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    if (radius > maxRadius) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float2 sourceUV = sourcePixel / sourceSize;
    float4 color = sampleSource(plane0, plane1, sourceSampler, sourceUV, sourceFormat);

    float featherStart = maxRadius * (1.0 - edgeFeather);
    if (edgeFeather > 0.0 && radius > featherStart) {
        float alpha = 1.0 - (radius - featherStart) / max(maxRadius - featherStart, 0.001);
        color.rgb *= clamp(alpha, 0.0, 1.0);
    }
    return color;
}
