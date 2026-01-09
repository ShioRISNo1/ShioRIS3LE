#include "brachy/dwell_time_optimizer.h"
#include "brachy/brachy_dose_calculator.h"
#include "dicom/brachy_source.h"
#include <QDebug>
#include <cmath>
#include <limits>

namespace Brachy {

DwellTimeOptimizer::DwellTimeOptimizer(const BrachyDoseCalculator* calculator)
    : m_calculator(calculator)
{
}

QVector<QVector<double>> DwellTimeOptimizer::computeDoseInfluenceMatrix(
    const BrachyPlan& plan,
    const QVector<DoseEvaluationPoint>& evaluationPoints) const
{
    const auto& sources = plan.sources();
    int numPoints = evaluationPoints.size();
    int numSources = sources.size();

    // Initialize matrix
    QVector<QVector<double>> matrix(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        matrix[i].resize(numSources);
    }

    // Compute dose from each source (with unit dwell time) at each point
    for (int j = 0; j < numSources; ++j) {
        // Create a temporary source with 1 second dwell time
        BrachySource unitSource = sources[j];
        unitSource.setDwellTime(1.0);

        for (int i = 0; i < numPoints; ++i) {
            const QVector3D& point = evaluationPoints[i].position();
            // Calculate dose from this single source with 1 second dwell time
            double dose = m_calculator->calculateSingleSourceDose(point, unitSource);
            matrix[i][j] = dose;
        }
    }

    return matrix;
}

QVector<DoseEvaluationPoint> DwellTimeOptimizer::calculateDosesAtPoints(
    const BrachyPlan& plan,
    const QVector<double>& dwellTimes,
    const QVector<DoseEvaluationPoint>& evaluationPoints) const
{
    QVector<DoseEvaluationPoint> result = evaluationPoints;
    const auto& sources = plan.sources();

    if (dwellTimes.size() != sources.size()) {
        qWarning() << "Dwell times size mismatch:" << dwellTimes.size() << "vs" << sources.size();
        return result;
    }

    // Create a temporary plan with updated dwell times
    BrachyPlan tempPlan;
    for (int i = 0; i < sources.size(); ++i) {
        BrachySource source = sources[i];
        source.setDwellTime(dwellTimes[i]);
        tempPlan.addSource(source);
    }

    // Calculate dose at each point
    for (int i = 0; i < result.size(); ++i) {
        const QVector3D& point = result[i].position();
        double dose = m_calculator->calculatePointDose(point, tempPlan);
        result[i].setCalculatedDose(dose);
    }

    return result;
}

double DwellTimeOptimizer::calculateRMSError(const QVector<DoseEvaluationPoint>& evaluationPoints)
{
    if (evaluationPoints.isEmpty()) {
        return 0.0;
    }

    double sumWeightedSquaredError = 0.0;
    double sumWeights = 0.0;

    for (const auto& point : evaluationPoints) {
        sumWeightedSquaredError += point.weightedSquaredError();
        sumWeights += point.weight();
    }

    if (sumWeights <= 0.0) {
        return 0.0;
    }

    return std::sqrt(sumWeightedSquaredError / sumWeights);
}

void DwellTimeOptimizer::applyConstraints(
    QVector<double>& dwellTimes,
    const OptimizationSettings& settings) const
{
    // Clamp to min/max bounds
    for (int i = 0; i < dwellTimes.size(); ++i) {
        if (dwellTimes[i] < settings.minDwellTime) {
            dwellTimes[i] = settings.minDwellTime;
        }
        if (dwellTimes[i] > settings.maxDwellTime) {
            dwellTimes[i] = settings.maxDwellTime;
        }
    }

    // Normalize to total time if requested
    // Note: normalizeToReferencePoint takes precedence over normalizeToTotalTime
    if (!settings.normalizeToReferencePoint && settings.normalizeToTotalTime && settings.totalTime > 0.0) {
        double currentTotal = 0.0;
        for (double t : dwellTimes) {
            currentTotal += t;
        }

        if (currentTotal > 0.0) {
            double scale = settings.totalTime / currentTotal;
            for (int i = 0; i < dwellTimes.size(); ++i) {
                dwellTimes[i] *= scale;
            }
        }
    }
}

DwellTimeOptimizer::OptimizationResult DwellTimeOptimizer::optimize(
    const BrachyPlan& plan,
    const QVector<DoseEvaluationPoint>& evaluationPoints,
    const OptimizationSettings& settings)
{
    OptimizationResult result;
    const auto& sources = plan.sources();
    int numSources = sources.size();
    int numPoints = evaluationPoints.size();

    if (numSources == 0) {
        result.message = "No sources in plan";
        return result;
    }

    if (numPoints == 0) {
        result.message = "No evaluation points specified";
        return result;
    }

    qDebug() << "Starting dwell time optimization:";
    qDebug() << "  Sources:" << numSources;
    qDebug() << "  Evaluation points:" << numPoints;
    qDebug() << "  Max iterations:" << settings.maxIterations;

    // Initialize dwell times from plan
    QVector<double> dwellTimes(numSources);
    for (int i = 0; i < numSources; ++i) {
        dwellTimes[i] = sources[i].dwellTime();
        // If initial dwell time is 0, start with a small positive value
        if (dwellTimes[i] <= 0.0) {
            dwellTimes[i] = 1.0;
        }
    }

    // Compute dose influence matrix (A matrix)
    qDebug() << "Computing dose influence matrix...";
    auto influenceMatrix = computeDoseInfluenceMatrix(plan, evaluationPoints);

    // Calculate initial error
    auto currentPoints = calculateDosesAtPoints(plan, dwellTimes, evaluationPoints);
    result.initialError = calculateRMSError(currentPoints);
    result.finalError = result.initialError;

    qDebug() << "Initial RMS error:" << result.initialError << "Gy";

    // Optimization loop using gradient descent
    double previousError = result.initialError;

    for (int iter = 0; iter < settings.maxIterations; ++iter) {
        // Calculate current doses
        currentPoints = calculateDosesAtPoints(plan, dwellTimes, evaluationPoints);

        // Compute gradient: dE/dt = 2 * A^T * (A*t - D_target) * weight
        QVector<double> gradient(numSources, 0.0);

        for (int j = 0; j < numSources; ++j) {
            double gradSum = 0.0;
            for (int i = 0; i < numPoints; ++i) {
                double error = currentPoints[i].doseDifference();
                double weight = currentPoints[i].weight();
                gradSum += 2.0 * influenceMatrix[i][j] * error * weight;
            }
            gradient[j] = gradSum / numPoints;
        }

        // Update dwell times using gradient descent
        for (int j = 0; j < numSources; ++j) {
            dwellTimes[j] -= settings.learningRate * gradient[j];
        }

        // Apply constraints
        applyConstraints(dwellTimes, settings);

        // Calculate new error
        currentPoints = calculateDosesAtPoints(plan, dwellTimes, evaluationPoints);
        double currentError = calculateRMSError(currentPoints);

        // Check convergence
        double relativeChange = std::abs(currentError - previousError) / (previousError + 1e-10);

        if (m_progressCallback && (iter % 10 == 0 || iter == settings.maxIterations - 1)) {
            m_progressCallback(iter + 1, settings.maxIterations, currentError);
        }

        if (iter % 10 == 0) {
            qDebug() << QString("Iteration %1: RMS error = %2 Gy, relative change = %3")
                        .arg(iter).arg(currentError, 0, 'f', 6).arg(relativeChange, 0, 'e', 2);
        }

        result.finalError = currentError;
        result.iterations = iter + 1;

        if (relativeChange < settings.convergenceTolerance) {
            result.converged = true;
            result.message = QString("Converged after %1 iterations").arg(iter + 1);
            qDebug() << result.message;
            break;
        }

        previousError = currentError;
    }

    if (!result.converged) {
        result.message = QString("Reached maximum iterations (%1)").arg(settings.maxIterations);
        qDebug() << result.message;
    }

    result.optimizedDwellTimes = dwellTimes;

    // Apply reference point normalization if requested
    bool normalizedApplied = false;
    if (settings.normalizeToReferencePoint) {
        const auto& refPoints = plan.referencePoints();
        if (!refPoints.isEmpty() && settings.referencePointIndex >= 0 &&
            settings.referencePointIndex < refPoints.size()) {

            const ReferencePoint& refPoint = refPoints[settings.referencePointIndex];

            qDebug() << "Applying reference point normalization...";
            QVector<double> normalizedTimes = normalizeToReferencePointDose(plan, refPoint);

            if (!normalizedTimes.isEmpty()) {
                // Update optimized dwell times with normalized values
                result.optimizedDwellTimes = normalizedTimes;

                // Recalculate error with normalized dwell times
                currentPoints = calculateDosesAtPoints(plan, normalizedTimes, evaluationPoints);
                result.finalError = calculateRMSError(currentPoints);
                normalizedApplied = true;
            } else {
                qWarning() << "Reference point normalization failed, using unnormalized dwell times";
            }
        } else {
            qWarning() << "Reference point normalization requested but no valid reference point available";
        }
    }

    // Print summary
    qDebug() << "Optimization complete:";
    qDebug() << "  Initial error:" << result.initialError << "Gy";
    qDebug() << "  Final error:" << result.finalError << "Gy";
    qDebug() << "  Improvement:" << (result.initialError - result.finalError) << "Gy";
    qDebug() << "  Iterations:" << result.iterations;

    // Print dose at each evaluation point (reuse calculation if normalization was applied)
    if (!normalizedApplied) {
        currentPoints = calculateDosesAtPoints(plan, result.optimizedDwellTimes, evaluationPoints);
    }
    for (int i = 0; i < currentPoints.size(); ++i) {
        const auto& pt = currentPoints[i];
        qDebug() << QString("  Point %1: target=%2 Gy, calculated=%3 Gy, error=%4 Gy (%5%)")
                    .arg(i)
                    .arg(pt.targetDose(), 0, 'f', 3)
                    .arg(pt.calculatedDose(), 0, 'f', 3)
                    .arg(pt.doseDifference(), 0, 'f', 3)
                    .arg(pt.relativeError() * 100.0, 0, 'f', 1);
    }

    return result;
}

double DwellTimeOptimizer::calculateDoseAtReferencePoint(
    const BrachyPlan& plan,
    const QVector<double>& dwellTimes,
    const ReferencePoint& referencePoint) const
{
    if (!m_calculator || !m_calculator->isInitialized()) {
        qWarning() << "Dose calculator not initialized";
        return 0.0;
    }

    const auto& sources = plan.sources();
    if (dwellTimes.size() != sources.size()) {
        qWarning() << "Dwell times size mismatch:" << dwellTimes.size() << "vs" << sources.size();
        return 0.0;
    }

    // Create a temporary plan with specified dwell times
    BrachyPlan tempPlan;
    for (int i = 0; i < sources.size(); ++i) {
        BrachySource source = sources[i];
        source.setDwellTime(dwellTimes[i]);
        tempPlan.addSource(source);
    }

    // Calculate dose at reference point
    double dose = m_calculator->calculatePointDose(referencePoint.position, tempPlan);
    return dose;
}

QVector<double> DwellTimeOptimizer::normalizeToReferencePointDose(
    const BrachyPlan& plan,
    const ReferencePoint& referencePoint) const
{
    if (!m_calculator || !m_calculator->isInitialized()) {
        qWarning() << "Dose calculator not initialized";
        return QVector<double>();
    }

    const auto& sources = plan.sources();
    if (sources.isEmpty()) {
        qWarning() << "No sources in plan";
        return QVector<double>();
    }

    if (referencePoint.prescribedDose <= 0.0) {
        qWarning() << "Invalid prescribed dose:" << referencePoint.prescribedDose;
        return QVector<double>();
    }

    // Get current dwell times
    QVector<double> dwellTimes = plan.getDwellTimes();

    // Calculate current dose at reference point
    double calculatedDose = calculateDoseAtReferencePoint(plan, dwellTimes, referencePoint);

    if (calculatedDose <= 0.0) {
        qWarning() << "Calculated dose at reference point is zero or negative:" << calculatedDose;
        return QVector<double>();
    }

    // Calculate normalization factor
    double normalizationFactor = referencePoint.prescribedDose / calculatedDose;

    qDebug() << "Reference point normalization:";
    qDebug() << "  Position:" << referencePoint.position;
    qDebug() << "  Prescribed dose:" << referencePoint.prescribedDose << "Gy";
    qDebug() << "  Calculated dose:" << calculatedDose << "Gy";
    qDebug() << "  Normalization factor:" << normalizationFactor;

    // Apply normalization to all dwell times
    QVector<double> normalizedDwellTimes(dwellTimes.size());
    for (int i = 0; i < dwellTimes.size(); ++i) {
        normalizedDwellTimes[i] = dwellTimes[i] * normalizationFactor;
    }

    return normalizedDwellTimes;
}

} // namespace Brachy
