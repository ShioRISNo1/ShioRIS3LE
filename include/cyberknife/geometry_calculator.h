#pragma once

#include <QVector3D>

namespace CyberKnife {

class GeometryCalculator {
public:
    struct BeamGeometry {
        QVector3D sourcePosition;
        QVector3D targetPosition;
        double SAD = 0.0;
        double collimatorSize = 0.0;

        // Precomputed beam coordinate system basis vectors
        // These should be initialized via initializeBasis() to avoid redundant calculations
        QVector3D beamX;
        QVector3D beamY;
        QVector3D beamZ;
        bool basisInitialized = false;

        // Initialize the beam basis vectors (should be called once per beam)
        void initializeBasis();
    };

    static double calculateSSD(const BeamGeometry &beam, const QVector3D &point);
    static double calculateDepth(const BeamGeometry &beam, const QVector3D &point);
    static double calculateOffAxisDistance(const BeamGeometry &beam, const QVector3D &point);

    static QVector3D patientToBeamCoordinate(const QVector3D &patientPoint, const BeamGeometry &beam);
    static QVector3D beamToPatientCoordinate(const QVector3D &beamPoint, const BeamGeometry &beam);

    // Build beam coordinate system basis vectors
    static void buildBeamBasis(const BeamGeometry &beam, QVector3D &beamX, QVector3D &beamY, QVector3D &beamZ);
};

} // namespace CyberKnife

