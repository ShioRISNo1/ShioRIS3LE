#include "brachy/brachy_dose_calculator.h"
#include <QDebug>
#include <QVector3D>
#include <cmath>
#include <limits>

namespace Brachy {

bool BrachyDoseCalculator::initialize(const QString &sourceDataPath) {
    qDebug() << "Initializing Brachy dose calculator with data:" << sourceDataPath;

    bool success = m_sourceData.loadFromFile(sourceDataPath);

    if (success) {
        qDebug() << "Brachy dose calculator initialized successfully";
    } else {
        qWarning() << "Failed to initialize Brachy dose calculator";
    }

    return success;
}

void BrachyDoseCalculator::setCtVolume(const DicomVolume *ctVolume) {
    m_ctVolume = ctVolume;
}

double BrachyDoseCalculator::calculatePointDose(const QVector3D &point,
                                                const BrachyPlan &plan) const {
    if (!m_sourceData.isLoaded()) {
        return 0.0;
    }

    double totalDose = 0.0;
    const auto &sources = plan.sources();

    // DEBUG: Log dwell times and ratios for verification
    static bool loggedDwellTimes = false;
    if (!loggedDwellTimes && !sources.isEmpty()) {
        qDebug() << "=== Dwell Times Used in Dose Calculation ===";

        // Group by channel and show ratios
        QMap<int, QVector<int>> channelIndices;
        for (int i = 0; i < sources.size(); ++i) {
            channelIndices[sources[i].channel()].append(i);
        }

        for (auto it = channelIndices.cbegin(); it != channelIndices.cend(); ++it) {
            int channel = it.key();
            const auto& indices = it.value();

            // Calculate total time for this channel
            double totalTime = 0.0;
            for (int idx : indices) {
                totalTime += sources[idx].dwellTime();
            }

            qDebug() << QString("Channel %1 (Total: %2 s):").arg(channel).arg(totalTime, 0, 'f', 3);

            // Log first few sources with ratios
            for (int j = 0; j < qMin(5, indices.size()); ++j) {
                int idx = indices[j];
                double dwellTime = sources[idx].dwellTime();
                double ratio = (totalTime > 0.0) ? (dwellTime / totalTime) : 0.0;
                qDebug() << QString("  Source %1: %2 s (%3%)")
                    .arg(idx).arg(dwellTime, 0, 'f', 3).arg(ratio * 100.0, 0, 'f', 1);
            }
            if (indices.size() > 5) {
                qDebug() << QString("  ... (%1 more sources)").arg(indices.size() - 5);
            }
        }
        qDebug() << "==========================================";
        loggedDwellTimes = true;
    }

    for (const auto &source : sources) {
        double dose = calculateSingleSourceDose(point, source);
        totalDose += dose;
    }

    return totalDose;
}

double BrachyDoseCalculator::calculateSingleSourceDose(const QVector3D &point,
                                                       const BrachySource &source) const {
    if (!m_sourceData.isLoaded()) {
        return 0.0;
    }

    // Skip if dwell time is zero
    if (source.dwellTime() <= 0.0) {
        return 0.0;
    }

    // Transform point to source coordinate system
    double axialDist = 0.0;
    double radialDist = 0.0;
    transformToSourceCoordinates(point, source, axialDist, radialDist);

    // DEBUG: Log coordinate transformation for first few calculations
    static int debugCount = 0;
    if (debugCount < 3) {
        qDebug() << "=== DEBUG: Dose calculation sample" << debugCount << "===";
        qDebug() << "Source position:" << source.position();
        qDebug() << "Source direction:" << source.direction();
        qDebug() << "Point:" << point;
        qDebug() << "Vector (point-source):" << (point - source.position());
        qDebug() << "Axial distance (z):" << axialDist << "mm";
        qDebug() << "Radial distance (r):" << radialDist << "mm";
        qDebug() << "Dose rate:" << m_sourceData.getDose(axialDist, radialDist);
        debugCount++;
    }

    // Get dose rate from lookup table (Gy per unit time)
    double doseRate = m_sourceData.getDose(axialDist, radialDist);

    // Calculate total dose: dose_rate * dwell_time
    double dose = doseRate * source.dwellTime();

    return dose;
}

void BrachyDoseCalculator::transformToSourceCoordinates(const QVector3D &point,
                                                        const BrachySource &source,
                                                        double &axialDist,
                                                        double &radialDist) const {
    // Vector from source to point
    QVector3D toPoint = point - source.position();

    // Get source direction (axis direction)
    QVector3D axis = source.direction();

    // If direction is not set, assume Z-axis
    if (axis.lengthSquared() < 0.0001f) {
        axis = QVector3D(0, 0, 1);
    } else {
        axis.normalize();
    }

    // Project onto axis to get axial distance
    axialDist = QVector3D::dotProduct(toPoint, axis);

    // Calculate radial distance (perpendicular to axis)
    QVector3D axialComponent = axis * axialDist;
    QVector3D radialComponent = toPoint - axialComponent;
    radialDist = radialComponent.length();
}

std::vector<double> BrachyDoseCalculator::calculateAutoBounds(const BrachyPlan &plan,
                                                              double margin) const {
    const auto &sources = plan.sources();

    if (sources.isEmpty()) {
        // Default bounds if no sources
        return {-50.0, -50.0, -50.0, 50.0, 50.0, 50.0};
    }

    // Find min/max coordinates
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double minZ = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();
    double maxZ = std::numeric_limits<double>::lowest();

    for (const auto &source : sources) {
        const QVector3D &pos = source.position();
        minX = std::min(minX, static_cast<double>(pos.x()));
        minY = std::min(minY, static_cast<double>(pos.y()));
        minZ = std::min(minZ, static_cast<double>(pos.z()));
        maxX = std::max(maxX, static_cast<double>(pos.x()));
        maxY = std::max(maxY, static_cast<double>(pos.y()));
        maxZ = std::max(maxZ, static_cast<double>(pos.z()));
    }

    // Add margin
    minX -= margin;
    minY -= margin;
    minZ -= margin;
    maxX += margin;
    maxY += margin;
    maxZ += margin;

    return {minX, minY, minZ, maxX, maxY, maxZ};
}

RTDoseVolume BrachyDoseCalculator::calculateVolumeDose(
    const BrachyPlan &plan,
    double voxelSize,
    const std::vector<double> &bounds,
    std::function<void(int, int)> progressCallback) const {

    RTDoseVolume doseVolume;

    if (!m_sourceData.isLoaded()) {
        qWarning() << "Cannot calculate dose: source data not loaded";
        return doseVolume;
    }

    // Determine bounds
    std::vector<double> useBounds = bounds;
    if (useBounds.empty()) {
        useBounds = calculateAutoBounds(plan);
        qDebug() << "Auto-calculated bounds:"
                 << useBounds[0] << useBounds[1] << useBounds[2]
                 << useBounds[3] << useBounds[4] << useBounds[5];
    }

    double minX = useBounds[0];
    double minY = useBounds[1];
    double minZ = useBounds[2];
    double maxX = useBounds[3];
    double maxY = useBounds[4];
    double maxZ = useBounds[5];

    // Calculate grid dimensions
    int nx = static_cast<int>(std::ceil((maxX - minX) / voxelSize));
    int ny = static_cast<int>(std::ceil((maxY - minY) / voxelSize));
    int nz = static_cast<int>(std::ceil((maxZ - minZ) / voxelSize));

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        qWarning() << "Invalid grid dimensions:" << nx << ny << nz;
        return doseVolume;
    }

    qDebug() << "Calculating dose on grid:" << nx << "x" << ny << "x" << nz;
    qDebug() << "Voxel size:" << voxelSize << "mm";

    // Create OpenCV volume (depth x height x width)
    int dims[3] = {nz, ny, nx};
    cv::Mat volume(3, dims, CV_32F, cv::Scalar(0.0f));

    // Calculate dose for each voxel
    int totalVoxels = nx * ny * nz;
    int processedVoxels = 0;

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                // Calculate patient coordinate
                QVector3D point(
                    minX + (ix + 0.5) * voxelSize,
                    minY + (iy + 0.5) * voxelSize,
                    minZ + (iz + 0.5) * voxelSize
                );

                // Calculate dose at this point
                double dose = calculatePointDose(point, plan);

                // Store in volume (in Gy)
                volume.at<float>(iz, iy, ix) = static_cast<float>(dose);

                ++processedVoxels;
            }
        }

        // Report progress
        if (progressCallback && (iz % 5 == 0 || iz == nz - 1)) {
            progressCallback(processedVoxels, totalVoxels);
        }
    }

    // Configure RTDoseVolume
    qDebug() << "Created cv::Mat with dims:" << volume.dims
             << "size[0]=" << volume.size[0]
             << "size[1]=" << volume.size[1]
             << "size[2]=" << volume.size[2]
             << "total elements:" << volume.total();

    if (volume.empty() || volume.dims != 3) {
        qWarning() << "Invalid volume created: empty=" << volume.empty()
                   << "dims=" << volume.dims;
        return doseVolume;  // Return empty volume
    }

    doseVolume.setVolume(volume);

    qDebug() << "After setVolume: width=" << doseVolume.width()
             << "height=" << doseVolume.height()
             << "depth=" << doseVolume.depth();

    if (doseVolume.width() == 0 || doseVolume.height() == 0 || doseVolume.depth() == 0) {
        qWarning() << "setVolume() failed to set dimensions correctly!";
        return doseVolume;  // Return empty volume
    }

    doseVolume.setSpacing(voxelSize, voxelSize, voxelSize);
    // Set origin to voxel center (DICOM standard: ImagePositionPatient is center of first voxel)
    // This aligns with dose calculation which uses voxel centers: minX + (ix + 0.5) * voxelSize
    doseVolume.setOrigin(
        minX + 0.5 * voxelSize,
        minY + 0.5 * voxelSize,
        minZ + 0.5 * voxelSize
    );

    // Set identity direction cosines (aligned with patient coordinate system)
    double rowDir[3] = {1.0, 0.0, 0.0};
    double colDir[3] = {0.0, 1.0, 0.0};
    double sliceDir[3] = {0.0, 0.0, 1.0};
    doseVolume.setDirectionCosines(rowDir, colDir, sliceDir);

    // Compute max dose for the volume
    doseVolume.computeMaxDose();

    qDebug() << "Dose calculation completed. Max dose:" << doseVolume.maxDose() << "Gy";

    return doseVolume;
}

double BrachyDoseCalculator::calculateNormalizationFactor(const BrachyPlan &plan) const {
    if (!m_sourceData.isLoaded()) {
        qWarning() << "Cannot calculate normalization: source data not loaded";
        return 1.0;
    }

    const auto &refPoints = plan.referencePoints();
    if (refPoints.isEmpty()) {
        qWarning() << "No reference points found for normalization";
        return 1.0;
    }

    qDebug() << "=== Calculating Normalization Factor ===";
    qDebug() << "Number of reference points:" << refPoints.size();

    // Calculate dose at each reference point
    QVector<double> calculatedDoses;
    QVector<double> prescribedDoses;
    QVector<double> factors;

    for (int i = 0; i < refPoints.size(); ++i) {
        const auto &refPoint = refPoints[i];

        // Skip if no prescribed dose
        if (refPoint.prescribedDose <= 0.0) {
            qDebug() << "Skipping reference point" << i
                     << "(no prescribed dose):" << refPoint.label;
            continue;
        }

        // Calculate dose at this reference point
        double calculatedDose = calculatePointDose(refPoint.position, plan);

        if (calculatedDose <= 0.0) {
            qWarning() << "Reference point" << i << "has zero calculated dose at"
                       << refPoint.position;
            continue;
        }

        double factor = refPoint.prescribedDose / calculatedDose;

        qDebug() << "Reference point" << i << ":" << refPoint.label;
        qDebug() << "  Position:" << refPoint.position;
        qDebug() << "  Prescribed dose:" << refPoint.prescribedDose << "Gy";
        qDebug() << "  Calculated dose:" << calculatedDose << "Gy";
        qDebug() << "  Normalization factor:" << factor;

        calculatedDoses.append(calculatedDose);
        prescribedDoses.append(refPoint.prescribedDose);
        factors.append(factor);
    }

    if (factors.isEmpty()) {
        qWarning() << "No valid reference points for normalization";
        return 1.0;
    }

    // Strategy: Use the reference point with the highest prescribed dose
    // This is typically the primary prescription point
    double maxPrescribedDose = 0.0;
    double selectedFactor = 1.0;
    int selectedIndex = 0;

    for (int i = 0; i < prescribedDoses.size(); ++i) {
        if (prescribedDoses[i] > maxPrescribedDose) {
            maxPrescribedDose = prescribedDoses[i];
            selectedFactor = factors[i];
            selectedIndex = i;
        }
    }

    qDebug() << "Selected reference point" << selectedIndex
             << "with prescribed dose" << maxPrescribedDose << "Gy";
    qDebug() << "Final normalization factor:" << selectedFactor;
    qDebug() << "=======================================";

    return selectedFactor;
}

QVector<ReferencePointError> BrachyDoseCalculator::verifyReferencePointDoses(
    const BrachyPlan &plan,
    double normalizationFactor) const {

    QVector<ReferencePointError> errors;

    if (!m_sourceData.isLoaded()) {
        qWarning() << "Cannot verify reference points: source data not loaded";
        return errors;
    }

    const auto &refPoints = plan.referencePoints();
    if (refPoints.isEmpty()) {
        qDebug() << "No reference points to verify";
        return errors;
    }

    qDebug() << "=== Verifying Reference Point Doses (After Normalization) ===";
    qDebug() << "Normalization factor:" << normalizationFactor;
    qDebug() << "Number of reference points:" << refPoints.size();

    for (int i = 0; i < refPoints.size(); ++i) {
        const auto &refPoint = refPoints[i];

        ReferencePointError error;
        error.label = refPoint.label.isEmpty() ? QString("Point %1").arg(i) : refPoint.label;
        error.position = refPoint.position;
        error.prescribedDose = refPoint.prescribedDose;

        // Calculate dose at this reference point (without normalization)
        double rawDose = calculatePointDose(refPoint.position, plan);

        // Apply normalization factor
        error.calculatedDose = rawDose * normalizationFactor;

        // Calculate errors
        if (error.prescribedDose > 0.0) {
            error.absoluteError = error.calculatedDose - error.prescribedDose;
            error.relativeError = (error.absoluteError / error.prescribedDose) * 100.0;
        } else {
            error.absoluteError = 0.0;
            error.relativeError = 0.0;
        }

        errors.append(error);

        // Log details
        qDebug() << "---";
        qDebug() << "Reference Point:" << error.label;
        qDebug() << "  Position:" << error.position;
        qDebug() << "  Prescribed dose:" << error.prescribedDose << "Gy";
        qDebug() << "  Calculated dose (normalized):" << error.calculatedDose << "Gy";
        qDebug() << "  Absolute error:" << error.absoluteError << "Gy";
        qDebug() << "  Relative error:" << error.relativeError << "%";
    }

    qDebug() << "=========================================================";

    return errors;
}

RTDoseVolume BrachyDoseCalculator::calculateVolumeDoseNormalized(
    const BrachyPlan &plan,
    double normalizationFactor,
    double voxelSize,
    const std::vector<double> &bounds,
    std::function<void(int, int)> progressCallback) const {

    RTDoseVolume doseVolume;

    if (!m_sourceData.isLoaded()) {
        qWarning() << "Cannot calculate dose: source data not loaded";
        return doseVolume;
    }

    qDebug() << "Calculating normalized dose with factor:" << normalizationFactor;

    // Determine bounds
    std::vector<double> useBounds = bounds;
    if (useBounds.empty()) {
        useBounds = calculateAutoBounds(plan);
        qDebug() << "Auto-calculated bounds:"
                 << useBounds[0] << useBounds[1] << useBounds[2]
                 << useBounds[3] << useBounds[4] << useBounds[5];
    }

    double minX = useBounds[0];
    double minY = useBounds[1];
    double minZ = useBounds[2];
    double maxX = useBounds[3];
    double maxY = useBounds[4];
    double maxZ = useBounds[5];

    // Calculate grid dimensions
    int nx = static_cast<int>(std::ceil((maxX - minX) / voxelSize));
    int ny = static_cast<int>(std::ceil((maxY - minY) / voxelSize));
    int nz = static_cast<int>(std::ceil((maxZ - minZ) / voxelSize));

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        qWarning() << "Invalid grid dimensions:" << nx << ny << nz;
        return doseVolume;
    }

    qDebug() << "Calculating dose on grid:" << nx << "x" << ny << "x" << nz;
    qDebug() << "Voxel size:" << voxelSize << "mm";

    // Create OpenCV volume (depth x height x width)
    int dims[3] = {nz, ny, nx};
    cv::Mat volume(3, dims, CV_32F, cv::Scalar(0.0f));

    // Calculate dose for each voxel with normalization
    int totalVoxels = nx * ny * nz;
    int processedVoxels = 0;

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                // Calculate patient coordinate
                QVector3D point(
                    minX + (ix + 0.5) * voxelSize,
                    minY + (iy + 0.5) * voxelSize,
                    minZ + (iz + 0.5) * voxelSize
                );

                // Calculate dose at this point
                double dose = calculatePointDose(point, plan);

                // Apply normalization factor
                double normalizedDose = dose * normalizationFactor;

                // Store in volume (in Gy)
                volume.at<float>(iz, iy, ix) = static_cast<float>(normalizedDose);

                ++processedVoxels;
            }
        }

        // Report progress
        if (progressCallback && (iz % 5 == 0 || iz == nz - 1)) {
            progressCallback(processedVoxels, totalVoxels);
        }
    }

    // Configure RTDoseVolume
    qDebug() << "Created cv::Mat with dims:" << volume.dims
             << "size[0]=" << volume.size[0]
             << "size[1]=" << volume.size[1]
             << "size[2]=" << volume.size[2]
             << "total elements:" << volume.total();

    if (volume.empty() || volume.dims != 3) {
        qWarning() << "Invalid volume created: empty=" << volume.empty()
                   << "dims=" << volume.dims;
        return doseVolume;
    }

    doseVolume.setVolume(volume);

    qDebug() << "After setVolume: width=" << doseVolume.width()
             << "height=" << doseVolume.height()
             << "depth=" << doseVolume.depth();

    if (doseVolume.width() == 0 || doseVolume.height() == 0 || doseVolume.depth() == 0) {
        qWarning() << "setVolume() failed to set dimensions correctly!";
        return doseVolume;
    }

    doseVolume.setSpacing(voxelSize, voxelSize, voxelSize);
    // Set origin to voxel center
    doseVolume.setOrigin(
        minX + 0.5 * voxelSize,
        minY + 0.5 * voxelSize,
        minZ + 0.5 * voxelSize
    );

    // Set identity direction cosines
    double rowDir[3] = {1.0, 0.0, 0.0};
    double colDir[3] = {0.0, 1.0, 0.0};
    double sliceDir[3] = {0.0, 0.0, 1.0};
    doseVolume.setDirectionCosines(rowDir, colDir, sliceDir);

    // Compute max dose for the volume
    doseVolume.computeMaxDose();

    qDebug() << "Normalized dose calculation completed. Max dose:"
             << doseVolume.maxDose() << "Gy";

    return doseVolume;
}

} // namespace Brachy
