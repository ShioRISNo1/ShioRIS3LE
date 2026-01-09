#pragma once

#include <QString>
#include <vector>

namespace Brachy {

/**
 * @brief Ir-192 source dose data loader
 *
 * Binary format: Dose[z][r] stored in row-major order (float32, 113KB)
 * - Physical interpretation: Dose as function of (z, r) where:
 *   - z: axial distance along source axis [-100mm, +100mm], center at z=0
 *   - r: radial distance from axis [0mm, 139mm]
 * - Array storage: Dose[z][r] means binary file contains:
 *   - z=-100: all r values from 0 to 139mm (140 floats)
 *   - z=-99: all r values from 0 to 139mm (140 floats)
 *   - ...
 *   - z=+100: all r values from 0 to 139mm (140 floats)
 * - Access pattern: m_doseData[(z+100) * 140 + r]
 * - Center position: Dose[100][0] corresponds to (z=0mm, r=0mm)
 */
class IrSourceData {
public:
    IrSourceData() = default;

    /**
     * @brief Load binary dose data from file
     * @param filePath Path to binary file containing dose data
     * @return true if loaded successfully
     */
    bool loadFromFile(const QString &filePath);

    /**
     * @brief Check if data is loaded
     */
    bool isLoaded() const { return m_isLoaded; }

    /**
     * @brief Get dose at specific axial and radial distance
     * @param axialDistanceMm Distance along source axis (mm), 0 = center
     * @param radialDistanceMm Distance perpendicular to axis (mm), 0 = on axis
     * @return Dose value (Gy) with bilinear interpolation
     */
    double getDose(double axialDistanceMm, double radialDistanceMm) const;

    /**
     * @brief Get raw data dimensions
     */
    int getAxialSize() const { return kAxialSize; }
    int getRadialSize() const { return kRadialSize; }

    /**
     * @brief Get data bounds
     */
    double getMinAxialMm() const { return -100.0; }
    double getMaxAxialMm() const { return 100.0; }
    double getMaxRadialMm() const { return 139.0; }

    /**
     * @brief Get raw dose value at array indices (no interpolation)
     * @param axialIndex Index [0, 200] representing z from -100mm to +100mm
     * @param radialIndex Index [0, 139] representing r from 0mm to 139mm
     * @return Dose value (Gy)
     */
    double getRawDose(int axialIndex, int radialIndex) const;

private:
    static constexpr int kAxialSize = 201;   // z-axis: -100mm to +100mm
    static constexpr int kRadialSize = 140;  // r-axis: 0mm to 139mm
    static constexpr int kCenterIndex = 100; // z=0 is at index 100

    bool m_isLoaded = false;
    std::vector<float> m_doseData;  // Flattened array stored as Dose[z][r]: m_doseData[z*140 + r]

    /**
     * @brief Bilinear interpolation helper
     */
    double interpolate(double axialDistanceMm, double radialDistanceMm) const;
};

} // namespace Brachy
