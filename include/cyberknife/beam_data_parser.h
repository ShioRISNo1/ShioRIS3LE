#pragma once

#include <QString>
#include <vector>

namespace CyberKnife {

class BeamDataParser {
public:
    struct DMData {
        std::vector<double> collimatorSizes;
        double depth = 0.0;
        std::vector<double> outputFactors;
        std::vector<double> depths;
        std::vector<std::vector<double>> outputFactorMatrix;
    };

    struct OCRData {
        double collimatorSize = 0.0;
        std::vector<double> depths;
        std::vector<double> radii;
        std::vector<std::vector<double>> ratios;
        QString sourceFileName;
        int nominalIndex = -1;
    };

    struct TMRData {
        std::vector<double> fieldSizes;
        std::vector<double> depths;
        std::vector<std::vector<double>> tmrValues;
    };

    static DMData parseDMTable(const QString &filePath);
    static OCRData parseOCRTable(const QString &filePath);
    static TMRData parseTMRTable(const QString &filePath);
};

} // namespace CyberKnife

