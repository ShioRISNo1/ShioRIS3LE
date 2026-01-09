#include <metal_stdlib>
using namespace metal;

/**
 * Metal Compute Shaders for CyberKnife Dose Calculation
 *
 * These shaders implement the ray tracing dose model:
 * Dose = 0.01 × OF × TMR × OCR × (800/SAD)²
 */

// Constants
constant float REFERENCE_SAD = 800.0f;
constant float RAY_TRACING_DOSE_SCALE = 0.01f;
constant float MAX_TMR_DEPTH = 449.99999f;
constant float MAX_OCR_DEPTH = 249.99999f;
constant float MAX_OCR_RADIUS = 59.99999f;

// Parameter structures to reduce buffer count (Metal limit: 31 buffers)
struct VolumeParams {
    int width;
    int height;
    int depth;
    float spacingX;
    float spacingY;
    float spacingZ;
    float originX;
    float originY;
    float originZ;
    float3 orientationX;
    float3 orientationY;
    float3 orientationZ;
    int stepX;
    int stepY;
    int stepZ;
    int gridCountX;
    int gridCountY;
    int gridCountZ;
    // Precomputed depth profile (optional)
    float depthEntryDistance;
    float depthStepSize;
    int depthSampleCount;
    int depthValid;
};

struct BeamParams {
    float3 source;
    float3 target;
    float3 basisX;
    float3 basisY;
    float3 basisZ;
    float collimatorSize;
};

struct LookupTableCounts {
    int ofDepthCount;
    int ofCollimatorCount;
    int tmrDepthCount;
    int tmrFieldSizeCount;
    int ocrDepthCount;
    int ocrRadiusCount;
    int ocrCollimatorCount;
};

/**
 * @brief 2D bilinear interpolation
 */
float interpolate2D(device const float* table,
                    device const float* xValues,
                    device const float* yValues,
                    int xCount,
                    int yCount,
                    float x,
                    float y)
{
    // Find indices
    int x0 = 0, x1 = 0;
    int y0 = 0, y1 = 0;

    // Linear search for x (could optimize with binary search)
    for (int i = 0; i < xCount - 1; i++) {
        if (x >= xValues[i] && x < xValues[i + 1]) {
            x0 = i;
            x1 = i + 1;
            break;
        }
    }
    if (x >= xValues[xCount - 1]) {
        x0 = x1 = xCount - 1;
    }

    // Linear search for y
    for (int i = 0; i < yCount - 1; i++) {
        if (y >= yValues[i] && y < yValues[i + 1]) {
            y0 = i;
            y1 = i + 1;
            break;
        }
    }
    if (y >= yValues[yCount - 1]) {
        y0 = y1 = yCount - 1;
    }

    // Get corner values
    // Table is stored in row-major order: [x][y], so index = x * yCount + y
    float q00 = table[x0 * yCount + y0];
    float q01 = table[x0 * yCount + y1];
    float q10 = table[x1 * yCount + y0];
    float q11 = table[x1 * yCount + y1];

    // Bilinear interpolation
    if (x0 == x1 && y0 == y1) {
        return q00;
    } else if (x0 == x1) {
        float ty = (y - yValues[y0]) / (yValues[y1] - yValues[y0]);
        return q00 * (1.0f - ty) + q01 * ty;
    } else if (y0 == y1) {
        float tx = (x - xValues[x0]) / (xValues[x1] - xValues[x0]);
        return q00 * (1.0f - tx) + q10 * tx;
    } else {
        float tx = (x - xValues[x0]) / (xValues[x1] - xValues[x0]);
        float ty = (y - yValues[y0]) / (yValues[y1] - yValues[y0]);

        float r0 = q00 * (1.0f - tx) + q10 * tx;
        float r1 = q01 * (1.0f - tx) + q11 * tx;

        return r0 * (1.0f - ty) + r1 * ty;
    }
}

/**
 * @brief 3D trilinear interpolation for OCR table
 */
float interpolate3D(device const float* table,
                    device const float* xValues,
                    device const float* yValues,
                    device const float* zValues,
                    int xCount,
                    int yCount,
                    int zCount,
                    float x,
                    float y,
                    float z)
{
    // Find indices
    int x0 = 0, x1 = 0;
    int y0 = 0, y1 = 0;
    int z0 = 0, z1 = 0;

    // Find x bounds
    for (int i = 0; i < xCount - 1; i++) {
        if (x >= xValues[i] && x < xValues[i + 1]) {
            x0 = i;
            x1 = i + 1;
            break;
        }
    }
    if (x >= xValues[xCount - 1]) x0 = x1 = xCount - 1;

    // Find y bounds
    for (int i = 0; i < yCount - 1; i++) {
        if (y >= yValues[i] && y < yValues[i + 1]) {
            y0 = i;
            y1 = i + 1;
            break;
        }
    }
    if (y >= yValues[yCount - 1]) y0 = y1 = yCount - 1;

    // Find z bounds
    for (int i = 0; i < zCount - 1; i++) {
        if (z >= zValues[i] && z < zValues[i + 1]) {
            z0 = i;
            z1 = i + 1;
            break;
        }
    }
    if (z >= zValues[zCount - 1]) z0 = z1 = zCount - 1;

    // Get 8 corner values
    // Table is stored in row-major order: [x][y][z], so index = (x * yCount + y) * zCount + z
    int idx000 = (x0 * yCount + y0) * zCount + z0;
    int idx001 = (x0 * yCount + y0) * zCount + z1;
    int idx010 = (x0 * yCount + y1) * zCount + z0;
    int idx011 = (x0 * yCount + y1) * zCount + z1;
    int idx100 = (x1 * yCount + y0) * zCount + z0;
    int idx101 = (x1 * yCount + y0) * zCount + z1;
    int idx110 = (x1 * yCount + y1) * zCount + z0;
    int idx111 = (x1 * yCount + y1) * zCount + z1;

    float c000 = table[idx000];
    float c001 = table[idx001];
    float c010 = table[idx010];
    float c011 = table[idx011];
    float c100 = table[idx100];
    float c101 = table[idx101];
    float c110 = table[idx110];
    float c111 = table[idx111];

    // Trilinear interpolation
    float tx = (x0 == x1) ? 0.0f : (x - xValues[x0]) / (xValues[x1] - xValues[x0]);
    float ty = (y0 == y1) ? 0.0f : (y - yValues[y0]) / (yValues[y1] - yValues[y0]);
    float tz = (z0 == z1) ? 0.0f : (z - zValues[z0]) / (zValues[z1] - zValues[z0]);

    float c00 = c000 * (1.0f - tx) + c001 * tx;
    float c01 = c010 * (1.0f - tx) + c011 * tx;
    float c10 = c100 * (1.0f - tx) + c101 * tx;
    float c11 = c110 * (1.0f - tx) + c111 * tx;

    float c0 = c00 * (1.0f - ty) + c01 * ty;
    float c1 = c10 * (1.0f - ty) + c11 * ty;

    return c0 * (1.0f - tz) + c1 * tz;
}

/**
 * @brief Dose calculation parameters structure
 */
struct DoseParams {
    // Volume dimensions
    int width;
    int height;
    int depth;

    // Voxel spacing (mm)
    float spacingX;
    float spacingY;
    float spacingZ;

    // Volume origin
    float originX;
    float originY;
    float originZ;

    // Image orientation (direction cosines)
    float orientationX[3];
    float orientationY[3];
    float orientationZ[3];

    // Calculation step size
    int stepX;
    int stepY;
    int stepZ;

    // Number of coarse grid samples per axis
    int gridCountX;
    int gridCountY;
    int gridCountZ;

    // Beam geometry
    float beamSourceX, beamSourceY, beamSourceZ;
    float beamTargetX, beamTargetY, beamTargetZ;
    float beamBasisX[3], beamBasisY[3], beamBasisZ[3];
    float collimatorSize;

    // Reference dose
    float referenceDose;

    // Precomputed depth profile (optional)
    float depthEntryDistance;
    float depthStepSize;
    int depthSampleCount;
    int depthValid;

    // Table dimensions
    int ofDepthCount;
    int ofCollimatorCount;
    int tmrDepthCount;
    int tmrFieldSizeCount;
    int ocrDepthCount;
    int ocrRadiusCount;
    int ocrCollimatorCount;

    // Accumulate mode
    int accumulate;
};

inline int coarseIndex(int index, int count, int step, int size)
{
    if (size <= 0) {
        return 0;
    }

    if (count <= 1 || size == 1) {
        return 0;
    }

    if (index <= 0) {
        return 0;
    }

    if (index >= count - 1) {
        return max(size - 1, 0);
    }

    int value = index * step;
    if (value >= size) {
        return size - 1;
    }

    return value;
}

/**
 * @brief Main dose calculation kernel
 *
 * Computes dose at each voxel using ray tracing model
 */
kernel void calculateDoseKernel(
    device const short* ctVolume [[buffer(0)]],
    device float* doseVolume [[buffer(1)]],
    device uchar* computedMask [[buffer(2)]],
    constant DoseParams& params [[buffer(3)]],
    device const float* ofTable [[buffer(4)]],
    device const float* ofDepths [[buffer(5)]],
    device const float* ofCollimators [[buffer(6)]],
    device const float* tmrTable [[buffer(7)]],
    device const float* tmrDepths [[buffer(8)]],
    device const float* tmrFieldSizes [[buffer(9)]],
    device const float* ocrTable [[buffer(10)]],
    device const float* ocrDepths [[buffer(11)]],
    device const float* ocrRadii [[buffer(12)]],
    device const float* ocrCollimators [[buffer(13)]],
    device const float* depthProfile [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.gridCountX || gid.y >= params.gridCountY || gid.z >= params.gridCountZ) {
        return;
    }

    int x = coarseIndex(int(gid.x), params.gridCountX, params.stepX, params.width);
    int y = coarseIndex(int(gid.y), params.gridCountY, params.stepY, params.height);
    int z = coarseIndex(int(gid.z), params.gridCountZ, params.stepZ, params.depth);

    if (x < 0 || y < 0 || z < 0 || x >= params.width || y >= params.height || z >= params.depth) {
        return;
    }

    // Calculate voxel position in patient coordinates
    float voxelX = params.originX +
                   x * params.spacingX * params.orientationX[0] +
                   y * params.spacingY * params.orientationY[0] +
                   z * params.spacingZ * params.orientationZ[0];

    float voxelY = params.originY +
                   x * params.spacingX * params.orientationX[1] +
                   y * params.spacingY * params.orientationY[1] +
                   z * params.spacingZ * params.orientationZ[1];

    float voxelZ = params.originZ +
                   x * params.spacingX * params.orientationX[2] +
                   y * params.spacingY * params.orientationY[2] +
                   z * params.spacingZ * params.orientationZ[2];

    // Vector from source to voxel
    float dx = voxelX - params.beamSourceX;
    float dy = voxelY - params.beamSourceY;
    float dz = voxelZ - params.beamSourceZ;

    // Calculate SAD (source-to-axis distance)
    float sad = sqrt(dx * dx + dy * dy + dz * dz);

    // Transform to beam coordinates
    float beamLocalX = dx * params.beamBasisX[0] +
                       dy * params.beamBasisX[1] +
                       dz * params.beamBasisX[2];

    float beamLocalY = dx * params.beamBasisY[0] +
                       dy * params.beamBasisY[1] +
                       dz * params.beamBasisY[2];

    float beamLocalZ = dx * params.beamBasisZ[0] +
                       dy * params.beamBasisZ[1] +
                       dz * params.beamBasisZ[2];

    // Calculate depth using precomputed water-equivalent profile if available
    float depth = 0.0f;
    if (params.depthValid && params.depthSampleCount > 0 && params.depthStepSize > 0.0f) {
        float relative = beamLocalZ - params.depthEntryDistance;
        if (relative <= 0.0f) {
            depth = 0.0f;
        } else {
            float index = relative / params.depthStepSize;
            if (index < 1.0f) {
                depth = depthProfile[0] * index;
            } else {
                int sampleCount = params.depthSampleCount;
                if (sampleCount == 1) {
                    depth = depthProfile[0];
                } else {
                    int lower = int(floor(index));
                    if (lower >= sampleCount - 1) {
                        float lastValue = depthProfile[sampleCount - 1];
                        float prevValue = depthProfile[sampleCount - 2];
                        float slope = lastValue - prevValue;
                        float extra = index - float(sampleCount - 1);
                        depth = lastValue + slope * extra;
                    } else {
                        int upper = min(lower + 1, sampleCount - 1);
                        float lowerValue = depthProfile[lower];
                        float upperValue = depthProfile[upper];
                        float fraction = index - float(lower);
                        depth = lowerValue + (upperValue - lowerValue) * fraction;
                    }
                }
            }
        }
    } else {
        depth = max(beamLocalZ, 0.0f);
    }

    // Calculate off-axis distance
    float offAxis = sqrt(beamLocalX * beamLocalX + beamLocalY * beamLocalY);

    // Calculate radius at 800mm reference distance
    float radius800 = offAxis * (REFERENCE_SAD / sad);

    // Clamp values to table limits
    depth = min(depth, MAX_TMR_DEPTH);
    float ocrDepth = min(depth, MAX_OCR_DEPTH);
    float ocrRadius = min(radius800, MAX_OCR_RADIUS);

    // Lookup Output Factor (OF)
    float outputFactor = interpolate2D(ofTable, ofDepths, ofCollimators,
                                       params.ofDepthCount, params.ofCollimatorCount,
                                       depth, params.collimatorSize);

    // Lookup TMR
    float fieldSize = params.collimatorSize;  // Simplified
    float tmr = interpolate2D(tmrTable, tmrDepths, tmrFieldSizes,
                              params.tmrDepthCount, params.tmrFieldSizeCount,
                              depth, fieldSize);

    // Lookup OCR (3D interpolation)
    float ocr = interpolate3D(ocrTable, ocrDepths, ocrRadii, ocrCollimators,
                              params.ocrDepthCount, params.ocrRadiusCount, params.ocrCollimatorCount,
                              ocrDepth, ocrRadius, params.collimatorSize);

    // Calculate dose using ray tracing formula:
    // Dose = 0.01 × OF × TMR × OCR × (800/SAD)²
    float sadFactor = REFERENCE_SAD / sad;
    float sadFactorSquared = sadFactor * sadFactor;

    float dose = RAY_TRACING_DOSE_SCALE * outputFactor * tmr * ocr *
                 sadFactorSquared * params.referenceDose;

    // Write result
    int index = z * params.width * params.height + y * params.width + x;

    if (params.accumulate) {
        doseVolume[index] += dose;
    } else {
        doseVolume[index] = dose;
    }

    // Mark as computed
    computedMask[index] = 1;
}

/**
 * @brief Trilinear interpolation kernel
 *
 * Fills non-computed voxels by interpolating from grid points
 * Only interpolates voxels on the target grid (determined by step parameter)
 */
kernel void interpolateVolumeKernel(
    device float* doseVolume [[buffer(0)]],
    device const uchar* computedMask [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& depth [[buffer(4)]],
    constant int& step [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = int(gid.x);
    int y = int(gid.y);
    int z = int(gid.z);

    // Bounds check
    if (x >= width || y >= height || z >= depth) {
        return;
    }

    int index = z * width * height + y * width + x;

    // Skip if already computed
    if (computedMask[index] != 0) {
        return;
    }

    // IMPORTANT: Only interpolate voxels on the target grid
    // Skip voxels that are not on step boundaries
    if (x % step != 0 || y % step != 0 || z % step != 0) {
        return;
    }

    // Find 8 neighboring grid points
    // The source grid has step size = step * 2 (e.g., step=2 -> source grid has step=4)
    int sourceStep = step * 2;
    int x0 = (x / sourceStep) * sourceStep;
    int x1 = x0 + sourceStep;
    int y0 = (y / sourceStep) * sourceStep;
    int y1 = y0 + sourceStep;
    int z0 = (z / sourceStep) * sourceStep;
    int z1 = z0 + sourceStep;

    // Clamp to volume bounds
    x1 = min(x1, width - 1);
    y1 = min(y1, height - 1);
    z1 = min(z1, depth - 1);

    // Get 8 corner values
    float c000 = doseVolume[z0 * width * height + y0 * width + x0];
    float c001 = doseVolume[z0 * width * height + y0 * width + x1];
    float c010 = doseVolume[z0 * width * height + y1 * width + x0];
    float c011 = doseVolume[z0 * width * height + y1 * width + x1];
    float c100 = doseVolume[z1 * width * height + y0 * width + x0];
    float c101 = doseVolume[z1 * width * height + y0 * width + x1];
    float c110 = doseVolume[z1 * width * height + y1 * width + x0];
    float c111 = doseVolume[z1 * width * height + y1 * width + x1];

    // Calculate interpolation weights
    float tx = (x0 == x1) ? 0.0f : float(x - x0) / float(x1 - x0);
    float ty = (y0 == y1) ? 0.0f : float(y - y0) / float(y1 - y0);
    float tz = (z0 == z1) ? 0.0f : float(z - z0) / float(z1 - z0);

    // Trilinear interpolation
    float c00 = c000 * (1.0f - tx) + c001 * tx;
    float c01 = c010 * (1.0f - tx) + c011 * tx;
    float c10 = c100 * (1.0f - tx) + c101 * tx;
    float c11 = c110 * (1.0f - tx) + c111 * tx;

    float c0 = c00 * (1.0f - ty) + c01 * ty;
    float c1 = c10 * (1.0f - ty) + c11 * ty;

    float result = c0 * (1.0f - tz) + c1 * tz;

    // Write interpolated value
    doseVolume[index] = result;
}

/**
 * @brief Recalculate dose for voxels above threshold
 *
 * Only updates voxels where current dose >= threshold
 * Used for selective refinement in Pass 2/3
 */
kernel void recalculateDoseWithThresholdKernel(
    device const short* ctVolume [[buffer(0)]],
    device float* doseVolume [[buffer(1)]],
    device uchar* computedMask [[buffer(2)]],

    constant VolumeParams& params [[buffer(3)]],
    constant BeamParams& beam [[buffer(4)]],
    constant float& referenceDose [[buffer(5)]],

    device const float* ofTable [[buffer(6)]],
    device const float* ofDepths [[buffer(7)]],
    device const float* ofCollimators [[buffer(8)]],

    device const float* tmrTable [[buffer(9)]],
    device const float* tmrDepths [[buffer(10)]],
    device const float* tmrFieldSizes [[buffer(11)]],

    device const float* ocrTable [[buffer(12)]],
    device const float* ocrDepths [[buffer(13)]],
    device const float* ocrRadii [[buffer(14)]],
    device const float* ocrCollimators [[buffer(15)]],

    constant LookupTableCounts& counts [[buffer(16)]],
    constant float& threshold [[buffer(17)]],
    device const float* depthProfile [[buffer(18)]],
    constant int& skipStep [[buffer(19)]],

    uint3 tid [[thread_position_in_grid]])
{
    int gidX = tid.x;
    int gidY = tid.y;
    int gidZ = tid.z;

    if (gidX >= params.gridCountX || gidY >= params.gridCountY || gidZ >= params.gridCountZ) {
        return;
    }

    int x;
    if (params.gridCountX <= 1 || params.width <= 1 || gidX == 0) {
        x = 0;
    } else if (gidX >= params.gridCountX - 1) {
        x = params.width - 1;
    } else {
        x = gidX * params.stepX;
        if (x >= params.width) x = params.width - 1;
    }

    int y;
    if (params.gridCountY <= 1 || params.height <= 1 || gidY == 0) {
        y = 0;
    } else if (gidY >= params.gridCountY - 1) {
        y = params.height - 1;
    } else {
        y = gidY * params.stepY;
        if (y >= params.height) y = params.height - 1;
    }

    int z;
    if (params.gridCountZ <= 1 || params.depth <= 1 || gidZ == 0) {
        z = 0;
    } else if (gidZ >= params.gridCountZ - 1) {
        z = params.depth - 1;
    } else {
        z = gidZ * params.stepZ;
        if (z >= params.depth) z = params.depth - 1;
    }

    if (x < 0 || y < 0 || z < 0 || x >= params.width || y >= params.height || z >= params.depth) {
        return;
    }

    int index = z * params.width * params.height + y * params.width + x;

    // IMPORTANT: Skip grid points on skipStep grid
    // Pass 2: skipStep=4 -> Skip Step 4 grid (directly calculated in Pass 1)
    // Pass 3: skipStep=2 -> Skip Step 2 grid (already calculated/recalculated)
    if (skipStep > 0 && x % skipStep == 0 && y % skipStep == 0 && z % skipStep == 0) {
        return;
    }

    // Check threshold - only recalculate if current dose >= threshold
    float currentDose = doseVolume[index];
    if (currentDose < threshold) {
        return;
    }

    // Calculate voxel position in patient coordinates
    float voxelX = params.originX + x * params.spacingX * params.orientationX.x + y * params.spacingY * params.orientationY.x + z * params.spacingZ * params.orientationZ.x;
    float voxelY = params.originY + x * params.spacingX * params.orientationX.y + y * params.spacingY * params.orientationY.y + z * params.spacingZ * params.orientationZ.y;
    float voxelZ = params.originZ + x * params.spacingX * params.orientationX.z + y * params.spacingY * params.orientationY.z + z * params.spacingZ * params.orientationZ.z;

    // Vector from source to voxel
    float dx = voxelX - beam.source.x;
    float dy = voxelY - beam.source.y;
    float dz = voxelZ - beam.source.z;

    // Calculate SAD
    float sad = sqrt(dx * dx + dy * dy + dz * dz);

    // Transform to beam coordinates
    float beamLocalX = dx * beam.basisX.x + dy * beam.basisX.y + dz * beam.basisX.z;
    float beamLocalY = dx * beam.basisY.x + dy * beam.basisY.y + dz * beam.basisY.z;
    float beamLocalZ = dx * beam.basisZ.x + dy * beam.basisZ.y + dz * beam.basisZ.z;

    // Calculate depth using precomputed water-equivalent profile if available
    float beamDepth = 0.0f;
    if (params.depthValid && params.depthSampleCount > 0 && params.depthStepSize > 0.0f) {
        float relative = beamLocalZ - params.depthEntryDistance;
        if (relative <= 0.0f) {
            beamDepth = 0.0f;
        } else {
            float index = relative / params.depthStepSize;
            if (index < 1.0f) {
                beamDepth = depthProfile[0] * index;
            } else {
                int sampleCount = params.depthSampleCount;
                if (sampleCount == 1) {
                    beamDepth = depthProfile[0];
                } else {
                    int lower = int(floor(index));
                    if (lower >= sampleCount - 1) {
                        float lastValue = depthProfile[sampleCount - 1];
                        float prevValue = depthProfile[sampleCount - 2];
                        float slope = lastValue - prevValue;
                        float extra = index - float(sampleCount - 1);
                        beamDepth = lastValue + slope * extra;
                    } else {
                        int upper = min(lower + 1, sampleCount - 1);
                        float lowerValue = depthProfile[lower];
                        float upperValue = depthProfile[upper];
                        float fraction = index - float(lower);
                        beamDepth = lowerValue + (upperValue - lowerValue) * fraction;
                    }
                }
            }
        }
    } else {
        beamDepth = max(beamLocalZ, 0.0f);
    }

    // Calculate off-axis distance
    float offAxis = sqrt(beamLocalX * beamLocalX + beamLocalY * beamLocalY);

    // Calculate radius at 800mm reference distance
    float radius800 = offAxis * (REFERENCE_SAD / sad);

    // Clamp values
    beamDepth = fmin(beamDepth, MAX_TMR_DEPTH);
    float ocrDepth = fmin(beamDepth, MAX_OCR_DEPTH);
    float ocrRadius = fmin(radius800, MAX_OCR_RADIUS);

    // Lookup OF
    float outputFactor = interpolate2D(ofTable, ofDepths, ofCollimators,
                                       counts.ofDepthCount, counts.ofCollimatorCount,
                                       beamDepth, beam.collimatorSize);

    // Lookup TMR
    float fieldSize = beam.collimatorSize;
    float tmr = interpolate2D(tmrTable, tmrDepths, tmrFieldSizes,
                              counts.tmrDepthCount, counts.tmrFieldSizeCount,
                              beamDepth, fieldSize);

    // Lookup OCR
    float ocr = interpolate3D(ocrTable, ocrDepths, ocrRadii, ocrCollimators,
                              counts.ocrDepthCount, counts.ocrRadiusCount, counts.ocrCollimatorCount,
                              ocrDepth, ocrRadius, beam.collimatorSize);

    // Calculate dose
    float sadFactor = (REFERENCE_SAD / sad);
    float sadFactorSquared = sadFactor * sadFactor;

    float dose = RAY_TRACING_DOSE_SCALE * outputFactor * tmr * ocr * sadFactorSquared * referenceDose;

    // Update dose
    doseVolume[index] = dose;

    // Mark as computed
    if (computedMask != nullptr) {
        computedMask[index] = 1;
    }
}
