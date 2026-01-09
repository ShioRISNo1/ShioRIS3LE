#include "brachy/ir_source_data.h"
#include <QFile>
#include <QDebug>
#include <cmath>

namespace Brachy {

bool IrSourceData::loadFromFile(const QString &filePath) {
    m_isLoaded = false;
    m_doseData.clear();

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open Ir source data file:" << filePath;
        return false;
    }

    // Expected size: 201 * 140 * 4 bytes = 112,560 bytes
    qint64 expectedSize = kAxialSize * kRadialSize * sizeof(float);
    qint64 fileSize = file.size();

    if (fileSize != expectedSize) {
        qWarning() << "Ir source data file size mismatch. Expected:" << expectedSize
                   << "Got:" << fileSize;
        // Continue anyway - try to read what we can
    }

    // Read binary data
    m_doseData.resize(kAxialSize * kRadialSize, 0.0f);
    qint64 bytesRead = file.read(reinterpret_cast<char*>(m_doseData.data()),
                                  kAxialSize * kRadialSize * sizeof(float));

    file.close();

    if (bytesRead < expectedSize) {
        qWarning() << "Could not read full Ir source data. Read:" << bytesRead
                   << "Expected:" << expectedSize;
        // Still mark as loaded if we got some data
    }

    m_isLoaded = true;
    qDebug() << "Ir source data loaded successfully from" << filePath;
    qDebug() << "Data dimensions:" << kAxialSize << "x" << kRadialSize;

    // DEBUG: Check data layout
    qDebug() << "=== DEBUG: Checking dose data layout ===";
    qDebug() << "Center dose (z=0, r=0) [index 100,0]:" << getRawDose(kCenterIndex, 0);
    qDebug() << "Dose at (z=0, r=1) [index 100,1]:" << getRawDose(kCenterIndex, 1);
    qDebug() << "Dose at (z=1, r=0) [index 101,0]:" << getRawDose(kCenterIndex + 1, 0);
    qDebug() << "Dose at (z=0, r=10) [index 100,10]:" << getRawDose(kCenterIndex, 10);
    qDebug() << "Dose at (z=10, r=0) [index 110,0]:" << getRawDose(kCenterIndex + 10, 0);

    // Check raw data values
    qDebug() << "Raw m_doseData[0]:" << m_doseData[0];
    qDebug() << "Raw m_doseData[1]:" << m_doseData[1];
    qDebug() << "Raw m_doseData[140]:" << m_doseData[140];
    qDebug() << "Raw m_doseData[14000] (center):" << m_doseData[kCenterIndex * kRadialSize];
    qDebug() << "======================================";

    return true;
}

double IrSourceData::getRawDose(int axialIndex, int radialIndex) const {
    if (!m_isLoaded) {
        return 0.0;
    }

    if (axialIndex < 0 || axialIndex >= kAxialSize ||
        radialIndex < 0 || radialIndex >= kRadialSize) {
        return 0.0;
    }

    // Binary file is stored as Dose[z][r] (row-major format)
    // Access: m_doseData[z * kRadialSize + r]
    return static_cast<double>(m_doseData[axialIndex * kRadialSize + radialIndex]);
}

double IrSourceData::getDose(double axialDistanceMm, double radialDistanceMm) const {
    if (!m_isLoaded) {
        return 0.0;
    }

    // Take absolute value of radial distance (symmetric)
    double r = std::abs(radialDistanceMm);

    // Check bounds
    if (r >= kRadialSize) {
        return 0.0;  // Out of radial bounds
    }

    // Convert axial distance to index: z_index = 100 + z(mm)
    double axialIndex = kCenterIndex + axialDistanceMm;

    if (axialIndex < 0 || axialIndex >= kAxialSize) {
        return 0.0;  // Out of axial bounds
    }

    // DEBUG: Log a few samples
    static int debugCount = 0;
    if (debugCount < 5) {
        qDebug() << "getDose: axialDist=" << axialDistanceMm << "mm, radialDist=" << radialDistanceMm
                 << "mm -> axialIndex=" << axialIndex << ", radialIndex=" << r;
        debugCount++;
    }

    // Perform bilinear interpolation
    return interpolate(axialDistanceMm, r);
}

double IrSourceData::interpolate(double axialDistanceMm, double radialDistanceMm) const {
    // Convert to indices
    double axialIndex = kCenterIndex + axialDistanceMm;
    double radialIndex = radialDistanceMm;

    // Get integer indices
    int z0 = static_cast<int>(std::floor(axialIndex));
    int z1 = z0 + 1;
    int r0 = static_cast<int>(std::floor(radialIndex));
    int r1 = r0 + 1;

    // Clamp to valid range
    z0 = std::max(0, std::min(z0, kAxialSize - 1));
    z1 = std::max(0, std::min(z1, kAxialSize - 1));
    r0 = std::max(0, std::min(r0, kRadialSize - 1));
    r1 = std::max(0, std::min(r1, kRadialSize - 1));

    // Get fractional parts
    double fz = axialIndex - std::floor(axialIndex);
    double fr = radialIndex - std::floor(radialIndex);

    // Clamp fractions
    fz = std::max(0.0, std::min(1.0, fz));
    fr = std::max(0.0, std::min(1.0, fr));

    // Get corner values
    double d00 = getRawDose(z0, r0);
    double d01 = getRawDose(z0, r1);
    double d10 = getRawDose(z1, r0);
    double d11 = getRawDose(z1, r1);

    // Bilinear interpolation
    double d0 = d00 * (1.0 - fr) + d01 * fr;
    double d1 = d10 * (1.0 - fr) + d11 * fr;
    double dose = d0 * (1.0 - fz) + d1 * fz;

    return dose;
}

} // namespace Brachy
