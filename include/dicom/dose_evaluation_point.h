#ifndef DOSE_EVALUATION_POINT_H
#define DOSE_EVALUATION_POINT_H

#include <QVector3D>
#include <QString>

/**
 * @brief Represents a dose evaluation point for brachytherapy optimization
 *
 * Dose evaluation points specify target dose values at specific locations.
 * These are used to optimize dwell times to achieve the prescribed dose distribution.
 */
class DoseEvaluationPoint {
public:
    DoseEvaluationPoint()
        : m_position(0.0, 0.0, 0.0)
        , m_targetDose(0.0)
        , m_weight(1.0)
        , m_calculatedDose(0.0)
    {}

    DoseEvaluationPoint(const QVector3D& position, double targetDose, double weight = 1.0)
        : m_position(position)
        , m_targetDose(targetDose)
        , m_weight(weight)
        , m_calculatedDose(0.0)
    {}

    // Getters
    const QVector3D& position() const { return m_position; }
    double targetDose() const { return m_targetDose; }
    double weight() const { return m_weight; }
    double calculatedDose() const { return m_calculatedDose; }
    const QString& label() const { return m_label; }

    // Setters
    void setPosition(const QVector3D& position) { m_position = position; }
    void setTargetDose(double dose) { m_targetDose = dose; }
    void setWeight(double weight) { m_weight = weight; }
    void setCalculatedDose(double dose) { m_calculatedDose = dose; }
    void setLabel(const QString& label) { m_label = label; }

    // Computed values
    double doseDifference() const { return m_calculatedDose - m_targetDose; }
    double relativeError() const {
        return m_targetDose > 0.0 ? (doseDifference() / m_targetDose) : 0.0;
    }
    double weightedSquaredError() const {
        double diff = doseDifference();
        return m_weight * diff * diff;
    }

private:
    QVector3D m_position;          // Position in patient coordinates (mm)
    double m_targetDose;           // Target dose in Gy
    double m_weight;               // Weight for optimization (higher = more important)
    double m_calculatedDose;       // Calculated dose in Gy (updated during optimization)
    QString m_label;               // Optional label for the point
};

#endif // DOSE_EVALUATION_POINT_H
