#ifndef BRACHY_PLAN_H
#define BRACHY_PLAN_H

#include <QString>
#include <QStringList>
#include <QVector>
#include <QVector3D>
#include "dicom/brachy_source.h"
#include "dicom/dose_evaluation_point.h"
#include "dicom/rtstruct.h"  // for StructurePointList

/**
 * @brief Represents a reference point from RTPLAN with coordinates and prescribed dose
 */
struct ReferencePoint {
    QVector3D position;        // Position in patient coordinates (mm)
    double prescribedDose;     // Prescribed dose in Gy
    QString label;             // Optional label/description

    ReferencePoint()
        : position(0.0, 0.0, 0.0)
        , prescribedDose(0.0)
    {}

    ReferencePoint(const QVector3D& pos, double dose, const QString& lbl = QString())
        : position(pos)
        , prescribedDose(dose)
        , label(lbl)
    {}
};

class BrachyPlan {
public:
    bool loadFromFile(const QString &filename);
    const QVector<BrachySource>& sources() const { return m_sources; }
    QVector<BrachySource>& sources() { return m_sources; }
    QStringList dwellTimeStrings() const;

    // Dose evaluation points
    const QVector<DoseEvaluationPoint>& evaluationPoints() const { return m_evaluationPoints; }
    QVector<DoseEvaluationPoint>& evaluationPoints() { return m_evaluationPoints; }
    void addEvaluationPoint(const DoseEvaluationPoint& point);
    void clearEvaluationPoints();
    void setEvaluationPoints(const QVector<DoseEvaluationPoint>& points) { m_evaluationPoints = points; }

    // Reference points
    const QVector<ReferencePoint>& referencePoints() const { return m_referencePoints; }
    QVector<ReferencePoint>& referencePoints() { return m_referencePoints; }
    void addReferencePoint(const ReferencePoint& point);
    void clearReferencePoints();

    // Dwell time management
    void setDwellTimes(const QVector<double>& dwellTimes);
    QVector<double> getDwellTimes() const;

    /**
     * @brief Generate random sources for testing
     * @param count Number of sources to generate
     * @param spatialRange Spatial range in mm (e.g., 50.0 means -50 to +50 mm)
     * @param minDwellTime Minimum dwell time in seconds
     * @param maxDwellTime Maximum dwell time in seconds
     */
    void generateRandomSources(int count = 10, double spatialRange = 50.0,
                               double minDwellTime = 1.0, double maxDwellTime = 10.0);

    /**
     * @brief Clear all sources
     */
    void clearSources();

    /**
     * @brief Add a single source
     * @param source Source to add
     */
    void addSource(const BrachySource &source);

    /**
     * @brief Generate a test source at origin for alignment verification
     */
    void generateTestSourceAtOrigin();

private:
    QVector<BrachySource> m_sources;
    QVector<DoseEvaluationPoint> m_evaluationPoints;
    QVector<ReferencePoint> m_referencePoints;
};

#endif // BRACHY_PLAN_H
