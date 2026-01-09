#pragma once

#include "brachy/ir_source_data.h"
#include "dicom/brachy_plan.h"
#include "dicom/dicom_volume.h"
#include "dicom/rtdose_volume.h"
#include <QString>
#include <QVector3D>
#include <functional>

namespace Brachy {

/**
 * @brief Reference point dose error information
 */
struct ReferencePointError {
    QString label;              // Reference point label
    QVector3D position;         // Position in patient coordinates (mm)
    double prescribedDose;      // Prescribed dose (Gy)
    double calculatedDose;      // Calculated dose after normalization (Gy)
    double absoluteError;       // Absolute error (Gy)
    double relativeError;       // Relative error (%)
};

/**
 * @brief Brachytherapy dose calculator for Ir-192 sources
 *
 * Calculates dose distribution from multiple dwell positions
 * using pre-computed dose kernel data.
 */
class BrachyDoseCalculator {
public:
    BrachyDoseCalculator() = default;

    /**
     * @brief Initialize calculator with Ir source data
     * @param sourceDataPath Path to binary dose data file
     * @return true if initialization succeeded
     */
    bool initialize(const QString &sourceDataPath);

    /**
     * @brief Check if calculator is ready
     */
    bool isInitialized() const { return m_sourceData.isLoaded(); }

    /**
     * @brief Set CT volume for dose calculation grid
     * @param ctVolume Pointer to CT volume (optional, for density correction)
     */
    void setCtVolume(const DicomVolume *ctVolume);

    /**
     * @brief Calculate dose at a single point from all sources
     * @param point Patient coordinate (mm)
     * @param plan Brachy plan with source positions and dwell times
     * @return Total dose (Gy) at the point
     */
    double calculatePointDose(const QVector3D &point, const BrachyPlan &plan) const;

    /**
     * @brief Calculate normalization factor from reference points
     * @param plan Brachy plan with source positions and reference points
     * @return Normalization factor (prescribedDose / calculatedDose)
     *         Returns 1.0 if no reference points or calculation fails
     */
    double calculateNormalizationFactor(const BrachyPlan &plan) const;

    /**
     * @brief Verify normalized doses at reference points
     * @param plan Brachy plan with source positions and reference points
     * @param normalizationFactor The normalization factor that was applied
     * @return Vector of error information for each reference point
     */
    QVector<ReferencePointError> verifyReferencePointDoses(
        const BrachyPlan &plan,
        double normalizationFactor) const;

    /**
     * @brief Calculate volume dose distribution
     * @param plan Brachy plan with source positions and dwell times
     * @param voxelSize Grid resolution (mm)
     * @param bounds Optional bounds [minX, minY, minZ, maxX, maxY, maxZ]
     * @param progressCallback Optional progress callback (current, total)
     * @return RTDoseVolume with calculated dose distribution
     */
    RTDoseVolume calculateVolumeDose(
        const BrachyPlan &plan,
        double voxelSize = 2.0,
        const std::vector<double> &bounds = {},
        std::function<void(int, int)> progressCallback = nullptr) const;

    /**
     * @brief Calculate volume dose distribution with normalization
     * @param plan Brachy plan with source positions and dwell times
     * @param normalizationFactor Factor to multiply all doses by
     * @param voxelSize Grid resolution (mm)
     * @param bounds Optional bounds [minX, minY, minZ, maxX, maxY, maxZ]
     * @param progressCallback Optional progress callback (current, total)
     * @return RTDoseVolume with normalized dose distribution
     */
    RTDoseVolume calculateVolumeDoseNormalized(
        const BrachyPlan &plan,
        double normalizationFactor,
        double voxelSize = 2.0,
        const std::vector<double> &bounds = {},
        std::function<void(int, int)> progressCallback = nullptr) const;

    /**
     * @brief Calculate dose contribution from a single source
     * @param point Patient coordinate (mm)
     * @param source Single brachy source
     * @return Dose (Gy) from this source
     */
    double calculateSingleSourceDose(const QVector3D &point,
                                     const BrachySource &source) const;

    /**
     * @brief Get source data (for debugging/validation)
     */
    const IrSourceData& sourceData() const { return m_sourceData; }

private:
    IrSourceData m_sourceData;
    const DicomVolume *m_ctVolume = nullptr;

    /**
     * @brief Transform point to source coordinate system
     * @param point Patient coordinate
     * @param source Source with position and direction
     * @param axialDist Output: distance along source axis (mm)
     * @param radialDist Output: distance perpendicular to axis (mm)
     */
    void transformToSourceCoordinates(const QVector3D &point,
                                     const BrachySource &source,
                                     double &axialDist,
                                     double &radialDist) const;

    /**
     * @brief Calculate automatic bounds from source positions
     */
    std::vector<double> calculateAutoBounds(const BrachyPlan &plan,
                                           double margin = 30.0) const;
};

} // namespace Brachy
