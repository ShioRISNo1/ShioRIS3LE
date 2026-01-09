#include "cyberknife/geometry_calculator.h"

#include <QtMath>

namespace CyberKnife {

void GeometryCalculator::buildBeamBasis(const BeamGeometry &beam, QVector3D &beamX, QVector3D &beamY, QVector3D &beamZ)
{
    beamZ = (beam.targetPosition - beam.sourcePosition).normalized();

    QVector3D globalUp(0.0f, 1.0f, 0.0f);
    if (qFabs(QVector3D::dotProduct(beamZ, globalUp)) > 0.99f) {
        globalUp = QVector3D(1.0f, 0.0f, 0.0f);
    }

    beamX = QVector3D::crossProduct(globalUp, beamZ).normalized();
    beamY = QVector3D::crossProduct(beamZ, beamX).normalized();
}

void GeometryCalculator::BeamGeometry::initializeBasis()
{
    GeometryCalculator::buildBeamBasis(*this, beamX, beamY, beamZ);
    basisInitialized = true;
}

double GeometryCalculator::calculateSSD(const BeamGeometry &beam, const QVector3D &point)
{
    return (point - beam.sourcePosition).length();
}

double GeometryCalculator::calculateDepth(const BeamGeometry &beam, const QVector3D &point)
{
    // Use precomputed basis if available, otherwise compute on-the-fly
    if (beam.basisInitialized) {
        QVector3D relative = point - beam.sourcePosition;
        return QVector3D::dotProduct(relative, beam.beamZ);
    } else {
        QVector3D beamX, beamY, beamZ;
        buildBeamBasis(beam, beamX, beamY, beamZ);
        QVector3D relative = point - beam.sourcePosition;
        return QVector3D::dotProduct(relative, beamZ);
    }
}

double GeometryCalculator::calculateOffAxisDistance(const BeamGeometry &beam, const QVector3D &point)
{
    // Use precomputed basis if available, otherwise compute on-the-fly
    if (beam.basisInitialized) {
        QVector3D relative = point - beam.targetPosition;
        double x = QVector3D::dotProduct(relative, beam.beamX);
        double y = QVector3D::dotProduct(relative, beam.beamY);
        return qSqrt(x * x + y * y);
    } else {
        QVector3D beamX, beamY, beamZ;
        buildBeamBasis(beam, beamX, beamY, beamZ);
        QVector3D relative = point - beam.targetPosition;
        double x = QVector3D::dotProduct(relative, beamX);
        double y = QVector3D::dotProduct(relative, beamY);
        return qSqrt(x * x + y * y);
    }
}

QVector3D GeometryCalculator::patientToBeamCoordinate(const QVector3D &patientPoint, const BeamGeometry &beam)
{
    // Use precomputed basis if available, otherwise compute on-the-fly
    if (beam.basisInitialized) {
        QVector3D relative = patientPoint - beam.targetPosition;
        return QVector3D(QVector3D::dotProduct(relative, beam.beamX),
                         QVector3D::dotProduct(relative, beam.beamY),
                         QVector3D::dotProduct(relative, beam.beamZ));
    } else {
        QVector3D beamX, beamY, beamZ;
        buildBeamBasis(beam, beamX, beamY, beamZ);
        QVector3D relative = patientPoint - beam.targetPosition;
        return QVector3D(QVector3D::dotProduct(relative, beamX),
                         QVector3D::dotProduct(relative, beamY),
                         QVector3D::dotProduct(relative, beamZ));
    }
}

QVector3D GeometryCalculator::beamToPatientCoordinate(const QVector3D &beamPoint, const BeamGeometry &beam)
{
    // Use precomputed basis if available, otherwise compute on-the-fly
    if (beam.basisInitialized) {
        QVector3D patientVector = beam.beamX * beamPoint.x() + beam.beamY * beamPoint.y() + beam.beamZ * beamPoint.z();
        return beam.targetPosition + patientVector;
    } else {
        QVector3D beamX, beamY, beamZ;
        buildBeamBasis(beam, beamX, beamY, beamZ);
        QVector3D patientVector = beamX * beamPoint.x() + beamY * beamPoint.y() + beamZ * beamPoint.z();
        return beam.targetPosition + patientVector;
    }
}

} // namespace CyberKnife

