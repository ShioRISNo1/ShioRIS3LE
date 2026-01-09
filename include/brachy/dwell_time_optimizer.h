#ifndef DWELL_TIME_OPTIMIZER_H
#define DWELL_TIME_OPTIMIZER_H

#include <QVector>
#include <functional>
#include "dicom/dose_evaluation_point.h"
#include "dicom/brachy_plan.h"

namespace Brachy {

class BrachyDoseCalculator;

/**
 * @brief Optimizes brachytherapy dwell times to match target doses at evaluation points
 *
 * Uses a least-squares optimization approach to adjust source dwell times
 * such that the calculated dose at evaluation points matches the target doses.
 */
class DwellTimeOptimizer {
public:
    struct OptimizationSettings {
        int maxIterations;           // Maximum number of optimization iterations
        double convergenceTolerance; // Relative change threshold for convergence
        double minDwellTime;          // Minimum allowed dwell time (seconds)
        double maxDwellTime;        // Maximum allowed dwell time (seconds)
        double learningRate;          // Learning rate for gradient descent
        bool normalizeToTotalTime;  // If true, maintain total dwell time
        double totalTime;             // Total time to maintain (if normalizing)
        bool normalizeToReferencePoint;  // If true, normalize dose to reference point
        int referencePointIndex;         // Index of reference point to use for normalization

        OptimizationSettings()
            : maxIterations(100)
            , convergenceTolerance(1e-4)
            , minDwellTime(0.0)
            , maxDwellTime(100.0)
            , learningRate(0.5)
            , normalizeToTotalTime(false)
            , totalTime(0.0)
            , normalizeToReferencePoint(false)
            , referencePointIndex(0)
        {}
    };

    struct OptimizationResult {
        bool converged;
        int iterations;
        double finalError;            // RMS error in Gy
        double initialError;          // Initial RMS error
        QVector<double> optimizedDwellTimes; // Optimized dwell times (seconds)
        QString message;

        OptimizationResult()
            : converged(false)
            , iterations(0)
            , finalError(0.0)
            , initialError(0.0)
        {}
    };

    DwellTimeOptimizer(const BrachyDoseCalculator* calculator);

    /**
     * @brief Optimize dwell times to match target doses
     * @param plan Input plan with sources (dwell times will be used as initial guess)
     * @param evaluationPoints Target points with prescribed doses
     * @param settings Optimization settings
     * @return Optimization result with optimized dwell times
     */
    OptimizationResult optimize(
        const BrachyPlan& plan,
        const QVector<DoseEvaluationPoint>& evaluationPoints,
        const OptimizationSettings& settings = OptimizationSettings()
    );

    /**
     * @brief Calculate dose at all evaluation points for given dwell times
     * @param plan Plan with sources
     * @param dwellTimes Dwell times for each source
     * @param evaluationPoints Points to evaluate
     * @return Updated evaluation points with calculated doses
     */
    QVector<DoseEvaluationPoint> calculateDosesAtPoints(
        const BrachyPlan& plan,
        const QVector<double>& dwellTimes,
        const QVector<DoseEvaluationPoint>& evaluationPoints
    ) const;

    /**
     * @brief Calculate root mean square error between calculated and target doses
     * @param evaluationPoints Points with calculated and target doses
     * @return RMS error in Gy
     */
    static double calculateRMSError(const QVector<DoseEvaluationPoint>& evaluationPoints);

    /**
     * @brief Set progress callback for long optimizations
     * @param callback Function called with (current iteration, max iterations, current error)
     */
    void setProgressCallback(std::function<void(int, int, double)> callback) {
        m_progressCallback = callback;
    }

    /**
     * @brief Normalize dwell times to achieve prescribed dose at reference point
     * @param plan Plan with sources (dwell times will be scaled)
     * @param referencePoint Reference point with position and prescribed dose
     * @return Normalized dwell times, or empty vector if normalization failed
     */
    QVector<double> normalizeToReferencePointDose(
        const BrachyPlan& plan,
        const ReferencePoint& referencePoint
    ) const;

    /**
     * @brief Calculate dose at reference point for given dwell times
     * @param plan Plan with sources
     * @param dwellTimes Dwell times for each source
     * @param referencePoint Reference point to evaluate
     * @return Calculated dose at reference point in Gy
     */
    double calculateDoseAtReferencePoint(
        const BrachyPlan& plan,
        const QVector<double>& dwellTimes,
        const ReferencePoint& referencePoint
    ) const;

private:
    const BrachyDoseCalculator* m_calculator;
    std::function<void(int, int, double)> m_progressCallback;

    /**
     * @brief Compute dose influence matrix (dose per unit dwell time)
     * Each row represents an evaluation point, each column a source
     * @param plan Plan with sources
     * @param evaluationPoints Points to evaluate
     * @return Matrix of size (numPoints x numSources)
     */
    QVector<QVector<double>> computeDoseInfluenceMatrix(
        const BrachyPlan& plan,
        const QVector<DoseEvaluationPoint>& evaluationPoints
    ) const;

    /**
     * @brief Apply constraints to dwell times
     * @param dwellTimes Dwell times to constrain
     * @param settings Constraint settings
     */
    void applyConstraints(
        QVector<double>& dwellTimes,
        const OptimizationSettings& settings
    ) const;
};

} // namespace Brachy

#endif // DWELL_TIME_OPTIMIZER_H
