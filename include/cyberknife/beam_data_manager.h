#pragma once

#include "cyberknife/beam_data_parser.h"

#include <QString>
#include <QStringList>

#include <map>

namespace CyberKnife {

class BeamDataManager {
private:
    BeamDataParser::DMData m_dmData;
    std::map<double, BeamDataParser::OCRData> m_ocrData;
    std::map<int, double> m_ocrIndexToCollimator;
    BeamDataParser::TMRData m_tmrData;
    QStringList m_validationErrors;
    bool m_isDataLoaded = false;
    int m_ocrTableIndexOffset = 0;

public:
    bool loadBeamData(const QString &dataDirectory);
    bool isDataLoaded() const;

    double getOutputFactor(double collimatorSize, double depth = 15.0) const;
    double getOCRRatio(double collimatorSize, double depth, double radius) const;
    double getTMRValue(double fieldSize, double depth) const;

    const BeamDataParser::OCRData *findClosestOcrTable(double collimatorSize,
                                                       double *matchedCollimator = nullptr) const;

    const BeamDataParser::DMData &dmData() const { return m_dmData; }
    const BeamDataParser::TMRData &tmrData() const { return m_tmrData; }
    const std::map<double, BeamDataParser::OCRData> &ocrData() const { return m_ocrData; }

    bool validateData();
    QStringList getAvailableCollimatorSizes() const;
    QStringList getValidationErrors() const;

    bool exportDmDataToCsv(const QString &filePath) const;
    bool exportTmrDataToCsv(const QString &filePath) const;
    bool exportOcrDataToCsv(const QString &filePath,
                            double minCollimator = 2.5,
                            double maxCollimator = 62.5) const;
    bool exportAllDataToCsv(const QString &directoryPath,
                            QStringList *exportedFiles = nullptr) const;
};

} // namespace CyberKnife
