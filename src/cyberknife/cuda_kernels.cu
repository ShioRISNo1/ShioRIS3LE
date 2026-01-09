/**
 * CUDA kernels for CyberKnife dose calculation
 *
 * Optimized for NVIDIA RTX 3090 (Compute Capability 8.6)
 * These kernels implement the ray tracing dose model:
 * Dose = 0.01 × OF × TMR × OCR × (800/SAD)²
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants
#define REFERENCE_SAD 800.0f
#define RAY_TRACING_DOSE_SCALE 0.01f
#define MAX_TMR_DEPTH 449.99999f
#define MAX_OCR_DEPTH 249.99999f
#define MAX_OCR_RADIUS 59.99999f

/**
 * @brief Interpolate beam depth from cumulative depth profile (device function)
 * Matches CPU implementation in BeamDepthProfile::interpolate()
 */
__device__ float interpolateBeamDepth(const float* depthCumulative,
                                      int depthSampleCount,
                                      float depthEntryDistance,
                                      float depthStepSize,
                                      float sad)
{
    if (depthSampleCount == 0 || !isfinite(depthEntryDistance) || depthStepSize <= 0.0f) {
        return sad;  // Fallback to geometric depth
    }

    if (!isfinite(sad) || sad <= depthEntryDistance) {
        return 0.0f;
    }

    const float relative = sad - depthEntryDistance;
    if (relative <= 0.0f) {
        return 0.0f;
    }

    const float index = relative / depthStepSize;
    if (index < 1.0f) {
        return depthCumulative[0] * index;
    }

    if (depthSampleCount == 1) {
        return depthCumulative[0];
    }

    const int lower = static_cast<int>(floorf(index));
    if (lower >= depthSampleCount - 1) {
        const float lastValue = depthCumulative[depthSampleCount - 1];
        const float prevValue = depthCumulative[depthSampleCount - 2];
        const float slope = (lastValue - prevValue);
        const float extra = index - (depthSampleCount - 1);
        return lastValue + slope * extra;
    }

    const int upper = lower + 1;
    const float lowerValue = depthCumulative[lower];
    const float upperValue = depthCumulative[upper];
    const float fraction = index - lower;

    return lowerValue + (upperValue - lowerValue) * fraction;
}

/**
 * @brief 2D bilinear interpolation (device function)
 */
__device__ float interpolate2D(const float* table,
                               const float* xValues,
                               const float* yValues,
                               int xCount,
                               int yCount,
                               float x,
                               float y)
{
    // Find indices
    int x0 = 0, x1 = 0;
    int y0 = 0, y1 = 0;

    // Binary search for x
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

    // Binary search for y
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
 * @brief 3D trilinear interpolation for OCR table (device function)
 */
__device__ float interpolate3D(const float* table,
                               const float* xValues,
                               const float* yValues,
                               const float* zValues,
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
 * @brief Calculate point dose using ray tracing model
 *
 * CUDA kernel for computing dose at each voxel in parallel
 */
__global__ void calculateDoseKernel(
    const short* ctVolume,        // CT volume in HU
    float* doseVolume,            // Output dose volume
    unsigned char* computedMask,  // Mask of computed voxels

    // Volume dimensions
    int width,
    int height,
    int depth,

    // Voxel spacing (mm)
    float spacingX,
    float spacingY,
    float spacingZ,

    // Volume origin in patient coordinates
    float originX,
    float originY,
    float originZ,

    // Image orientation (direction cosines)
    float orientationX0, float orientationX1, float orientationX2,
    float orientationY0, float orientationY1, float orientationY2,
    float orientationZ0, float orientationZ1, float orientationZ2,

    // Calculation step size
    int stepX,
    int stepY,
    int stepZ,

    // Number of coarse grid samples per axis
    int gridCountX,
    int gridCountY,
    int gridCountZ,

    // Beam geometry
    float beamSourceX, float beamSourceY, float beamSourceZ,
    float beamTargetX, float beamTargetY, float beamTargetZ,
    float beamBasisX0, float beamBasisX1, float beamBasisX2,
    float beamBasisY0, float beamBasisY1, float beamBasisY2,
    float beamBasisZ0, float beamBasisZ1, float beamBasisZ2,
    float collimatorSize,

    // Reference dose
    float referenceDose,

    // Beam data tables
    const float* ofTable,
    const float* ofDepths,
    const float* ofCollimators,
    int ofDepthCount,
    int ofCollimatorCount,

    const float* tmrTable,
    const float* tmrDepths,
    const float* tmrFieldSizes,
    int tmrDepthCount,
    int tmrFieldSizeCount,

    const float* ocrTable,
    const float* ocrDepths,
    const float* ocrRadii,
    const float* ocrCollimators,
    int ocrDepthCount,
    int ocrRadiusCount,
    int ocrCollimatorCount,

    // Depth profile for CT-density corrected depth calculation
    const float* depthCumulative,
    int depthSampleCount,
    float depthEntryDistance,
    float depthStepSize,

    // Accumulate mode
    int accumulate
)
{
    int gidX = blockIdx.x * blockDim.x + threadIdx.x;
    int gidY = blockIdx.y * blockDim.y + threadIdx.y;
    int gidZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (gidX >= gridCountX || gidY >= gridCountY || gidZ >= gridCountZ) {
        return;
    }

    int x;
    if (gridCountX <= 1 || width <= 1 || gidX == 0) {
        x = 0;
    } else if (gidX >= gridCountX - 1) {
        x = width - 1;
    } else {
        x = gidX * stepX;
        if (x >= width) x = width - 1;
    }

    int y;
    if (gridCountY <= 1 || height <= 1 || gidY == 0) {
        y = 0;
    } else if (gidY >= gridCountY - 1) {
        y = height - 1;
    } else {
        y = gidY * stepY;
        if (y >= height) y = height - 1;
    }

    int z;
    if (gridCountZ <= 1 || depth <= 1 || gidZ == 0) {
        z = 0;
    } else if (gidZ >= gridCountZ - 1) {
        z = depth - 1;
    } else {
        z = gidZ * stepZ;
        if (z >= depth) z = depth - 1;
    }

    if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) {
        return;
    }

    // Calculate voxel position in patient coordinates
    float voxelX = originX + x * spacingX * orientationX0 + y * spacingY * orientationY0 + z * spacingZ * orientationZ0;
    float voxelY = originY + x * spacingX * orientationX1 + y * spacingY * orientationY1 + z * spacingZ * orientationZ1;
    float voxelZ = originZ + x * spacingX * orientationX2 + y * spacingY * orientationY2 + z * spacingZ * orientationZ2;

    // Vector from source to voxel
    float dx = voxelX - beamSourceX;
    float dy = voxelY - beamSourceY;
    float dz = voxelZ - beamSourceZ;

    // Calculate SAD (source-to-axis distance)
    float sad = sqrtf(dx * dx + dy * dy + dz * dz);

    // Transform to beam coordinates
    float beamLocalX = dx * beamBasisX0 + dy * beamBasisX1 + dz * beamBasisX2;
    float beamLocalY = dx * beamBasisY0 + dy * beamBasisY1 + dz * beamBasisY2;
    float beamLocalZ = dx * beamBasisZ0 + dy * beamBasisZ1 + dz * beamBasisZ2;

    // Calculate depth - use CT-density corrected depth if available
    float beamDepth;
    if (depthSampleCount > 0 && isfinite(depthEntryDistance)) {
        beamDepth = interpolateBeamDepth(depthCumulative, depthSampleCount,
                                         depthEntryDistance, depthStepSize, beamLocalZ);
    } else {
        // Fallback to geometric depth
        beamDepth = beamLocalZ;
        if (beamDepth < 0.0f) beamDepth = 0.0f;
    }

    // Calculate off-axis distance
    float offAxis = sqrtf(beamLocalX * beamLocalX + beamLocalY * beamLocalY);

    // Calculate radius at 800mm reference distance
    float radius800 = offAxis * (REFERENCE_SAD / sad);

    // Clamp values to table limits
    beamDepth = fminf(beamDepth, MAX_TMR_DEPTH);
    float ocrDepth = fminf(beamDepth, MAX_OCR_DEPTH);
    float ocrRadius = fminf(radius800, MAX_OCR_RADIUS);

    // Lookup Output Factor (OF)
    float outputFactor = interpolate2D(ofTable, ofDepths, ofCollimators,
                                       ofDepthCount, ofCollimatorCount,
                                       beamDepth, collimatorSize);

    // Lookup TMR
    // Effective field size must be scaled by SAD/800 to match CPU implementation
    float fieldSize = collimatorSize * sad / REFERENCE_SAD;
    float tmr = interpolate2D(tmrTable, tmrDepths, tmrFieldSizes,
                              tmrDepthCount, tmrFieldSizeCount,
                              beamDepth, fieldSize);

    // Lookup OCR (2D interpolation - OCR table is depth x radius)
    // Note: ocrCollimatorCount is always 1 (single matched collimator table)
    float ocr = interpolate2D(ocrTable, ocrDepths, ocrRadii,
                              ocrDepthCount, ocrRadiusCount,
                              ocrDepth, ocrRadius);

    // Calculate dose using ray tracing formula:
    // Dose = 0.01 × OF × TMR × OCR × (800/SAD)²
    float sadFactor = (REFERENCE_SAD / sad);
    float sadFactorSquared = sadFactor * sadFactor;

    float dose = RAY_TRACING_DOSE_SCALE * outputFactor * tmr * ocr * sadFactorSquared * referenceDose;

    // Debug: Print first few voxels' calculations
    if (gidX < 3 && gidY < 3 && gidZ < 3) {
        printf("CUDA Kernel [%d,%d,%d]: depth=%.2f offAxis=%.2f radius800=%.2f sad=%.2f\n",
               gidX, gidY, gidZ, beamDepth, offAxis, radius800, sad);
        printf("  OF=%.6f TMR=%.6f OCR=%.6f sadFactor²=%.6f\n",
               outputFactor, tmr, ocr, sadFactorSquared);
        printf("  dose=%.8f (scale=%.2f)\n", dose, RAY_TRACING_DOSE_SCALE);
    }

    // Write result
    int index = z * width * height + y * width + x;

    if (accumulate) {
        atomicAdd(&doseVolume[index], dose);
    } else {
        doseVolume[index] = dose;
    }

    // Mark as computed
    if (computedMask != nullptr) {
        computedMask[index] = 1;
    }
}

/**
 * @brief Recalculate dose for voxels above threshold
 *
 * Only updates voxels where current dose >= threshold
 * Used for selective refinement in Pass 2/3
 */
__global__ void recalculateDoseWithThresholdKernel(
    const short* ctVolume,
    float* doseVolume,
    unsigned char* computedMask,

    // Volume dimensions
    int width,
    int height,
    int depth,

    // Voxel spacing (mm)
    float spacingX,
    float spacingY,
    float spacingZ,

    // Volume origin in patient coordinates
    float originX,
    float originY,
    float originZ,

    // Image orientation (direction cosines)
    float orientationX0, float orientationX1, float orientationX2,
    float orientationY0, float orientationY1, float orientationY2,
    float orientationZ0, float orientationZ1, float orientationZ2,

    // Calculation step size
    int stepX,
    int stepY,
    int stepZ,

    // Number of coarse grid samples per axis
    int gridCountX,
    int gridCountY,
    int gridCountZ,

    // Beam geometry
    float beamSourceX, float beamSourceY, float beamSourceZ,
    float beamTargetX, float beamTargetY, float beamTargetZ,
    float beamBasisX0, float beamBasisX1, float beamBasisX2,
    float beamBasisY0, float beamBasisY1, float beamBasisY2,
    float beamBasisZ0, float beamBasisZ1, float beamBasisZ2,
    float collimatorSize,

    // Reference dose
    float referenceDose,

    // Beam data tables
    const float* ofTable,
    const float* ofDepths,
    const float* ofCollimators,
    int ofDepthCount,
    int ofCollimatorCount,

    const float* tmrTable,
    const float* tmrDepths,
    const float* tmrFieldSizes,
    int tmrDepthCount,
    int tmrFieldSizeCount,

    const float* ocrTable,
    const float* ocrDepths,
    const float* ocrRadii,
    const float* ocrCollimators,
    int ocrDepthCount,
    int ocrRadiusCount,
    int ocrCollimatorCount,

    // Depth profile for CT-density corrected depth calculation
    const float* depthCumulative,
    int depthSampleCount,
    float depthEntryDistance,
    float depthStepSize,

    // Threshold for recalculation
    float threshold,

    // Skip step (4 for Pass 2, 2 for Pass 3)
    int skipStep
)
{
    int gidX = blockIdx.x * blockDim.x + threadIdx.x;
    int gidY = blockIdx.y * blockDim.y + threadIdx.y;
    int gidZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (gidX >= gridCountX || gidY >= gridCountY || gidZ >= gridCountZ) {
        return;
    }

    int x;
    if (gridCountX <= 1 || width <= 1 || gidX == 0) {
        x = 0;
    } else if (gidX >= gridCountX - 1) {
        x = width - 1;
    } else {
        x = gidX * stepX;
        if (x >= width) x = width - 1;
    }

    int y;
    if (gridCountY <= 1 || height <= 1 || gidY == 0) {
        y = 0;
    } else if (gidY >= gridCountY - 1) {
        y = height - 1;
    } else {
        y = gidY * stepY;
        if (y >= height) y = height - 1;
    }

    int z;
    if (gridCountZ <= 1 || depth <= 1 || gidZ == 0) {
        z = 0;
    } else if (gidZ >= gridCountZ - 1) {
        z = depth - 1;
    } else {
        z = gidZ * stepZ;
        if (z >= depth) z = depth - 1;
    }

    if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) {
        return;
    }

    int index = z * width * height + y * width + x;

    // Skip grid points on skipStep grid
    if (skipStep > 0 && x % skipStep == 0 && y % skipStep == 0 && z % skipStep == 0) {
        return;
    }

    // Check threshold - only recalculate if current dose >= threshold
    float currentDose = doseVolume[index];
    if (currentDose < threshold) {
        return;
    }

    // Calculate voxel position in patient coordinates
    float voxelX = originX + x * spacingX * orientationX0 + y * spacingY * orientationY0 + z * spacingZ * orientationZ0;
    float voxelY = originY + x * spacingX * orientationX1 + y * spacingY * orientationY1 + z * spacingZ * orientationZ1;
    float voxelZ = originZ + x * spacingX * orientationX2 + y * spacingY * orientationY2 + z * spacingZ * orientationZ2;

    // Vector from source to voxel
    float dx = voxelX - beamSourceX;
    float dy = voxelY - beamSourceY;
    float dz = voxelZ - beamSourceZ;

    // Calculate SAD
    float sad = sqrtf(dx * dx + dy * dy + dz * dz);

    // Transform to beam coordinates
    float beamLocalX = dx * beamBasisX0 + dy * beamBasisX1 + dz * beamBasisX2;
    float beamLocalY = dx * beamBasisY0 + dy * beamBasisY1 + dz * beamBasisY2;
    float beamLocalZ = dx * beamBasisZ0 + dy * beamBasisZ1 + dz * beamBasisZ2;

    // Calculate depth - use CT-density corrected depth if available
    float beamDepth;
    if (depthSampleCount > 0 && isfinite(depthEntryDistance)) {
        beamDepth = interpolateBeamDepth(depthCumulative, depthSampleCount,
                                         depthEntryDistance, depthStepSize, beamLocalZ);
    } else {
        // Fallback to geometric depth
        beamDepth = beamLocalZ;
        if (beamDepth < 0.0f) beamDepth = 0.0f;
    }

    // Calculate off-axis distance
    float offAxis = sqrtf(beamLocalX * beamLocalX + beamLocalY * beamLocalY);

    // Calculate radius at 800mm reference distance
    float radius800 = offAxis * (REFERENCE_SAD / sad);

    // Clamp values to table limits
    beamDepth = fminf(beamDepth, MAX_TMR_DEPTH);
    float ocrDepth = fminf(beamDepth, MAX_OCR_DEPTH);
    float ocrRadius = fminf(radius800, MAX_OCR_RADIUS);

    // Lookup Output Factor (OF)
    float outputFactor = interpolate2D(ofTable, ofDepths, ofCollimators,
                                       ofDepthCount, ofCollimatorCount,
                                       beamDepth, collimatorSize);

    // Lookup TMR
    // Effective field size must be scaled by SAD/800 to match CPU implementation
    float fieldSize = collimatorSize * sad / REFERENCE_SAD;
    float tmr = interpolate2D(tmrTable, tmrDepths, tmrFieldSizes,
                              tmrDepthCount, tmrFieldSizeCount,
                              beamDepth, fieldSize);

    // Lookup OCR (2D interpolation - OCR table is depth x radius)
    // Note: ocrCollimatorCount is always 1 (single matched collimator table)
    float ocr = interpolate2D(ocrTable, ocrDepths, ocrRadii,
                              ocrDepthCount, ocrRadiusCount,
                              ocrDepth, ocrRadius);

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

/**
 * @brief Trilinear interpolation kernel
 *
 * Fills non-computed voxels by interpolating from neighboring computed voxels
 */
__global__ void interpolateVolumeKernel(
    float* doseVolume,
    const unsigned char* computedMask,
    int width,
    int height,
    int depth,
    int step
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Bounds check
    if (x >= width || y >= height || z >= depth) {
        return;
    }

    int index = z * width * height + y * width + x;

    // Skip if already computed
    if (computedMask[index] != 0) {
        return;
    }

    // Only interpolate voxels on the target grid
    if (x % step != 0 || y % step != 0 || z % step != 0) {
        return;
    }

    // Find 8 neighboring grid points
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
    float tx = (x0 == x1) ? 0.0f : (float)(x - x0) / (float)(x1 - x0);
    float ty = (y0 == y1) ? 0.0f : (float)(y - y0) / (float)(y1 - y0);
    float tz = (z0 == z1) ? 0.0f : (float)(z - z0) / (float)(z1 - z0);

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
