#include "cyberknife/beam_data_parser.h"

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>
#include <QtDebug>
#include <QtGlobal>
#include <QVector>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace CyberKnife {
namespace {

struct NumericLine {
    QString rawLine;
    std::vector<double> values;
};

QString readNextDataLine(QTextStream &stream)
{
    while (!stream.atEnd()) {
        const QString line = stream.readLine().trimmed();
        if (line.isEmpty()) {
            continue;
        }
        if (line.startsWith('#')) {
            continue;
        }
        return line;
    }
    return {};
}

std::vector<double> extractNumbers(const QString &line)
{
    std::vector<double> values;
    static const QRegularExpression numberPattern(
        QStringLiteral(R"((-?\d*\.?\d+(?:[eE][-+]?\d+)?))"));
    QRegularExpressionMatchIterator it = numberPattern.globalMatch(line);
    while (it.hasNext()) {
        const QRegularExpressionMatch match = it.next();
        bool ok = false;
        const double value = match.captured(1).toDouble(&ok);
        if (!ok) {
            continue;
        }
        values.push_back(value);
    }
    return values;
}

QVector<double> toQVector(const std::vector<double> &values)
{
    QVector<double> result;
    result.reserve(static_cast<int>(values.size()));
    for (double value : values) {
        result.append(value);
    }
    return result;
}

bool readNextNumericLine(QTextStream &stream, std::vector<double> &values, QString *rawLine = nullptr)
{
    while (!stream.atEnd()) {
        const QString line = readNextDataLine(stream);
        if (line.isEmpty()) {
            continue;
        }
        std::vector<double> numbers = extractNumbers(line);
        if (numbers.empty()) {
            continue;
        }
        if (rawLine) {
            *rawLine = line;
        }
        values = std::move(numbers);
        return true;
    }
    values.clear();
    if (rawLine) {
        rawLine->clear();
    }
    return false;
}

} // namespace

BeamDataParser::DMData BeamDataParser::parseDMTable(const QString &filePath)
{
    qDebug() << "★★★ Parsing DM Table:" << filePath;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(QStringLiteral("Failed to open DM table: %1").arg(filePath).toStdString());
    }

    QTextStream stream(&file);

    const double defaultDepth = 15.0;
    int expectedCollimatorCount = -1;
    int expectedDepthCount = -1;
    std::vector<double> collimatorSizes;
    std::vector<double> depths;
    std::vector<std::vector<double>> outputFactorMatrix;

    std::vector<NumericLine> numericLines;
    std::vector<double> numbers;
    QString rawLine;
    while (readNextNumericLine(stream, numbers, &rawLine)) {
        numericLines.push_back({ rawLine, numbers });
    }

    std::vector<NumericLine> dataLines;
    std::vector<double> explicitDepths;
    for (const NumericLine &line : numericLines) {
        const QString lowerLine = line.rawLine.toLower();
        qDebug() << "★★★ DM candidate line:" << line.rawLine << "values:" << toQVector(line.values);

        if (lowerLine.contains(QStringLiteral("version"))
            || lowerLine.contains(QStringLiteral("sample"))) {
            qDebug() << "★★★ Skipping metadata line.";
            continue;
        }

        if (!line.values.empty() && expectedCollimatorCount < 0
            && (lowerLine.contains(QStringLiteral("collimator"))
                || lowerLine.contains(QStringLiteral("field")))) {
            expectedCollimatorCount = static_cast<int>(line.values.front());
            qDebug() << "★★★ Expected collimator count from header:" << expectedCollimatorCount;
            if (line.values.size() > 1) {
                std::vector<double> possibleSizes(line.values.begin() + 1, line.values.end());
                if (!possibleSizes.empty()) {
                    dataLines.push_back({ line.rawLine, std::move(possibleSizes) });
                    qDebug() << "★★★ Header line also contained collimator sizes.";
                }
            }
            continue;
        }
        if (!line.values.empty() && expectedDepthCount < 0
            && (lowerLine.contains(QStringLiteral("depth"))
                || lowerLine.contains(QStringLiteral("sad")))) {
            expectedDepthCount = static_cast<int>(line.values.front());
            qDebug() << "★★★ Expected depth count from header:" << expectedDepthCount;
            if (line.values.size() > 1) {
                explicitDepths.assign(line.values.begin() + 1, line.values.end());
                if (!explicitDepths.empty()) {
                    qDebug() << "★★★ Header line provided explicit depths:" << toQVector(explicitDepths);
                }
            }
            continue;
        }

        dataLines.push_back(line);
    }

    if (dataLines.empty()) {
        throw std::runtime_error("DM table is missing usable data lines.");
    }

    auto isIncreasingSequence = [](const std::vector<double> &values) {
        if (values.size() < 2) {
            return false;
        }
        for (size_t i = 1; i < values.size(); ++i) {
            if (!(values[i] > values[i - 1])) {
                return false;
            }
        }
        return true;
    };

    auto collimatorLineIt = dataLines.end();
    for (auto it = dataLines.begin(); it != dataLines.end(); ++it) {
        const auto &values = it->values;
        if (values.size() < 2) {
            continue;
        }
        if (expectedCollimatorCount > 0
            && static_cast<int>(values.size()) < expectedCollimatorCount) {
            continue;
        }
        if (!isIncreasingSequence(values)) {
            continue;
        }
        collimatorLineIt = it;
        break;
    }

    if (collimatorLineIt == dataLines.end()) {
        if (dataLines.empty()) {
            throw std::runtime_error("DM table is missing collimator sizes line.");
        }
        collimatorLineIt = dataLines.begin();
    }

    collimatorSizes = collimatorLineIt->values;
    dataLines.erase(collimatorLineIt);

    if (collimatorSizes.empty()) {
        throw std::runtime_error("DM table is missing collimator sizes line.");
    }

    if (expectedCollimatorCount <= 0) {
        expectedCollimatorCount = static_cast<int>(collimatorSizes.size());
    }
    if (static_cast<int>(collimatorSizes.size()) > expectedCollimatorCount) {
        collimatorSizes.resize(expectedCollimatorCount);
    }
    if (static_cast<int>(collimatorSizes.size()) != expectedCollimatorCount) {
        qWarning() << "★★★ Collimator count mismatch:" << expectedCollimatorCount
                   << collimatorSizes.size();
    }
    qDebug() << "★★★ Collimator sizes read:" << toQVector(collimatorSizes);

    outputFactorMatrix.reserve(static_cast<size_t>(std::max(expectedDepthCount, 0)));

    for (const NumericLine &line : dataLines) {
        if (line.values.empty()) {
            continue;
        }

        if (explicitDepths.empty() && expectedDepthCount > 0
            && static_cast<int>(line.values.size()) == expectedDepthCount
            && expectedDepthCount != expectedCollimatorCount) {
            explicitDepths = line.values;
            qDebug() << "★★★ Found standalone depth list:" << toQVector(explicitDepths);
            continue;
        }

        std::vector<double> rowValues;
        double depthValue = std::numeric_limits<double>::quiet_NaN();

        if (static_cast<int>(line.values.size()) >= expectedCollimatorCount + 1) {
            depthValue = line.values.front();
            rowValues.assign(line.values.begin() + 1,
                             line.values.begin() + 1 + expectedCollimatorCount);
            if (static_cast<int>(line.values.size()) > expectedCollimatorCount + 1) {
                qWarning() << "★★★ DM row has extra columns; ignoring trailing values.";
            }
        } else if (static_cast<int>(line.values.size()) >= expectedCollimatorCount) {
            rowValues.assign(line.values.begin(),
                             line.values.begin() + expectedCollimatorCount);
            if (!explicitDepths.empty()
                && static_cast<int>(depths.size()) < static_cast<int>(explicitDepths.size())) {
                depthValue = explicitDepths[depths.size()];
            }
        } else {
            qWarning() << "★★★ Unrecognized DM data line; insufficient columns.";
            continue;
        }

        if (static_cast<int>(rowValues.size()) != expectedCollimatorCount) {
            qWarning() << "★★★ DM row column count mismatch:" << rowValues.size()
                       << expectedCollimatorCount;
            continue;
        }

        outputFactorMatrix.push_back(std::move(rowValues));
        if (std::isfinite(depthValue)) {
            depths.push_back(depthValue);
        }
    }

    if (outputFactorMatrix.empty()) {
        throw std::runtime_error("DM table is missing output factor rows.");
    }

    if (!depths.empty()
        && static_cast<int>(depths.size()) != static_cast<int>(outputFactorMatrix.size())) {
        qWarning() << "★★★ Depth row count mismatch:" << depths.size()
                   << outputFactorMatrix.size();
    }

    if (expectedDepthCount > 0 && expectedDepthCount != static_cast<int>(outputFactorMatrix.size())) {
        qWarning() << "★★★ Expected depth rows" << expectedDepthCount
                   << "but found" << outputFactorMatrix.size();
    }

    DMData result;
    result.collimatorSizes = std::move(collimatorSizes);
    if (!depths.empty()) {
        result.depths = std::move(depths);
    } else if (!explicitDepths.empty()) {
        result.depths = std::move(explicitDepths);
    }
    result.outputFactorMatrix = std::move(outputFactorMatrix);
    if (!result.depths.empty()) {
        result.depth = result.depths.front();
    } else {
        if (expectedDepthCount == 1) {
            qDebug() << "★★★ Depth count reported as 1 but no explicit depth values present.";
        }
        result.depth = defaultDepth;
    }
    if (!result.outputFactorMatrix.empty()) {
        result.outputFactors = result.outputFactorMatrix.front();
    }

    qDebug() << "★★★ DM Table parsed successfully";
    qDebug() << "★★★ Final collimator sizes:" << toQVector(result.collimatorSizes);
    qDebug() << "★★★ Final output factors (primary row):" << toQVector(result.outputFactors);
    qDebug() << "★★★ Depths available:" << toQVector(result.depths);

    return result;
}

BeamDataParser::OCRData BeamDataParser::parseOCRTable(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(QStringLiteral("Failed to open OCR table: %1").arg(filePath).toStdString());
    }

    QTextStream stream(&file);
    std::vector<NumericLine> numericLines;
    std::vector<double> values;
    QString rawLine;
    while (readNextNumericLine(stream, values, &rawLine)) {
        numericLines.push_back({ rawLine, values });
    }

    double collimatorSize = 0.0;
    int depthCount = -1;
    int radiusCount = -1;
    std::vector<NumericLine> dataLines;

    for (const NumericLine &line : numericLines) {
        const QString lowerLine = line.rawLine.toLower();
        if (lowerLine.contains(QStringLiteral("version"))
            || lowerLine.contains(QStringLiteral("sample"))) {
            continue;
        }

        bool consumed = false;
        if (!line.values.empty() && collimatorSize <= 0.0
            && lowerLine.contains(QStringLiteral("collimator"))) {
            collimatorSize = line.values.front();
            qDebug() << "[OCR] Header collimator size:" << collimatorSize;
            if (line.values.size() > 1) {
                std::vector<double> remainder;
                remainder.assign(line.values.begin() + 1, line.values.end());
                if (!remainder.empty()) {
                    dataLines.push_back({ line.rawLine, std::move(remainder) });
                }
            }
            consumed = true;
        }
        if (!line.values.empty() && depthCount < 0
            && lowerLine.contains(QStringLiteral("depth"))) {
            depthCount = static_cast<int>(line.values.front());
            qDebug() << "[OCR] Header depth count:" << depthCount;
            if (line.values.size() > 1) {
                std::vector<double> remainder;
                remainder.assign(line.values.begin() + 1, line.values.end());
                if (!remainder.empty()) {
                    dataLines.push_back({ line.rawLine, std::move(remainder) });
                }
            }
            consumed = true;
        }
        if (!line.values.empty() && radiusCount < 0
            && (lowerLine.contains(QStringLiteral("radius"))
                || lowerLine.contains(QStringLiteral("radial")))) {
            radiusCount = static_cast<int>(line.values.front());
            qDebug() << "[OCR] Header radius count:" << radiusCount;
            if (line.values.size() > 1) {
                std::vector<double> remainder;
                remainder.assign(line.values.begin() + 1, line.values.end());
                if (!remainder.empty()) {
                    dataLines.push_back({ line.rawLine, std::move(remainder) });
                }
            }
            consumed = true;
        }

        if (!consumed) {
            dataLines.push_back(line);
        }
    }

    if (collimatorSize <= 0.0 && !numericLines.empty() && numericLines.front().values.size() >= 3) {
        const NumericLine &line = numericLines.front();
        collimatorSize = line.values[0];
        if (depthCount < 0) {
            depthCount = static_cast<int>(line.values[1]);
        }
        if (radiusCount < 0) {
            radiusCount = static_cast<int>(line.values[2]);
        }
        if (line.values.size() > 3) {
            std::vector<double> remainder;
            remainder.assign(line.values.begin() + 3, line.values.end());
            if (!remainder.empty()) {
                dataLines.push_back({ line.rawLine, std::move(remainder) });
            }
        }
        qDebug() << "[OCR] Fallback header -> collimator:" << collimatorSize << "depths:" << depthCount
                 << "radii:" << radiusCount;
    }

    if (collimatorSize <= 0.0 || depthCount <= 0 || radiusCount <= 0) {
        throw std::runtime_error(QStringLiteral("OCR table headerが不完全です: %1")
                                     .arg(filePath)
                                     .toStdString());
    }

    if (dataLines.empty()) {
        throw std::runtime_error(QStringLiteral("OCR tableのデータ行が不足しています: %1")
                                     .arg(filePath)
                                     .toStdString());
    }

    auto extractDepths = [&](NumericLine &line) -> std::vector<double> {
        std::vector<double> depthValues = line.values;
        if (!depthValues.empty() && qFuzzyCompare(depthValues.front(), collimatorSize)
            && static_cast<int>(depthValues.size()) >= depthCount + 1) {
            depthValues.erase(depthValues.begin());
        }
        if (static_cast<int>(depthValues.size()) > depthCount) {
            depthValues.resize(depthCount);
        }
        return depthValues;
    };

    std::vector<double> depths = extractDepths(dataLines.front());
    QString depthLineRaw = dataLines.front().rawLine;
    dataLines.erase(dataLines.begin());

    if (static_cast<int>(depths.size()) != depthCount) {
        bool replaced = false;
        for (size_t index = 0; index < dataLines.size(); ++index) {
            std::vector<double> candidate = extractDepths(dataLines[index]);
            if (static_cast<int>(candidate.size()) == depthCount) {
                depths = std::move(candidate);
                depthLineRaw = dataLines[index].rawLine;
                dataLines.erase(dataLines.begin() + static_cast<long>(index));
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            qWarning() << "[OCR] Depth count mismatch:" << depthCount << depths.size();
        }
    }
    qDebug() << "[OCR] Parsed depths from" << depthLineRaw << ':' << toQVector(depths);

    std::vector<double> radii;
    std::vector<std::vector<double>> ratios;

    if (!dataLines.empty() && radiusCount > 0
        && static_cast<int>(dataLines.front().values.size()) == radiusCount) {
        radii = dataLines.front().values;
        dataLines.erase(dataLines.begin());
        qDebug() << "[OCR] Radii line detected:" << toQVector(radii);

        ratios.reserve(depthCount);
        for (int depthIndex = 0; depthIndex < depthCount; ++depthIndex) {
            if (dataLines.empty()) {
                throw std::runtime_error(QStringLiteral("OCR tableの比率行が不足しています: %1")
                                             .arg(filePath)
                                             .toStdString());
            }
            std::vector<double> row = dataLines.front().values;
            dataLines.erase(dataLines.begin());
            if (static_cast<int>(row.size()) == radiusCount + 1) {
                row.erase(row.begin());
            }
            if (static_cast<int>(row.size()) < radiusCount) {
                throw std::runtime_error(QStringLiteral("OCR ratio rowの列数が不足しています: %1")
                                             .arg(filePath)
                                             .toStdString());
            }
            if (static_cast<int>(row.size()) > radiusCount) {
                row.resize(radiusCount);
                qWarning() << "[OCR] Ratio row has extra columns; trimming.";
            }
            ratios.push_back(std::move(row));
        }
    } else {
        radii.reserve(radiusCount);
        ratios.assign(depthCount, std::vector<double>());
        for (int depthIndex = 0; depthIndex < depthCount; ++depthIndex) {
            ratios[depthIndex].reserve(radiusCount);
        }

        for (const NumericLine &line : dataLines) {
            if (static_cast<int>(radii.size()) >= radiusCount) {
                break;
            }
            if (line.values.empty()) {
                continue;
            }

            std::vector<double> row = line.values;
            if (static_cast<int>(row.size()) == depthCount + 2) {
                row.erase(row.begin());
            }
            if (static_cast<int>(row.size()) < depthCount + 1) {
                qWarning() << "[OCR] Ratio row has insufficient columns.";
                continue;
            }
            if (static_cast<int>(row.size()) > depthCount + 1) {
                row.resize(depthCount + 1);
                qWarning() << "[OCR] Ratio row has extra columns; trimming.";
            }

            const double radius = row.front();
            std::vector<double> ratioValues(row.begin() + 1, row.end());
            if (static_cast<int>(ratioValues.size()) != depthCount) {
                qWarning() << "[OCR] Ratio value count mismatch:" << ratioValues.size()
                           << depthCount;
                continue;
            }

            radii.push_back(radius);
            for (int depthIndex = 0; depthIndex < depthCount; ++depthIndex) {
                ratios[depthIndex].push_back(ratioValues[depthIndex]);
            }
        }
    }

    if (static_cast<int>(radii.size()) != radiusCount) {
        throw std::runtime_error(QStringLiteral("OCR table radius行数が期待値と一致しません: %1")
                                     .arg(filePath)
                                     .toStdString());
    }

    for (int depthIndex = 0; depthIndex < depthCount; ++depthIndex) {
        if (static_cast<int>(ratios.size()) <= depthIndex) {
            throw std::runtime_error(QStringLiteral("OCR table ratioデータが不足しています: %1")
                                         .arg(filePath)
                                         .toStdString());
        }
        if (static_cast<int>(ratios[depthIndex].size()) != radiusCount) {
            throw std::runtime_error(QStringLiteral("OCR ratio rowの列数が一致しません: %1")
                                         .arg(filePath)
                                         .toStdString());
        }
    }

    OCRData result;
    result.collimatorSize = collimatorSize;
    result.depths = std::move(depths);
    result.radii = std::move(radii);
    result.ratios = std::move(ratios);
    result.sourceFileName = QFileInfo(filePath).fileName();
    qDebug() << "[OCR] Completed parsing for" << result.sourceFileName;
    return result;
}

BeamDataParser::TMRData BeamDataParser::parseTMRTable(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(QStringLiteral("Failed to open TMR table: %1").arg(filePath).toStdString());
    }

    QTextStream stream(&file);
    std::vector<NumericLine> numericLines;
    std::vector<double> values;
    QString rawLine;
    while (readNextNumericLine(stream, values, &rawLine)) {
        numericLines.push_back({ rawLine, values });
    }

    int fieldSizeCount = -1;
    int depthCount = -1;
    std::vector<double> fieldSizes;
    std::vector<double> standaloneDepths;
    std::vector<NumericLine> dataLines;

    for (const NumericLine &line : numericLines) {
        const QString lowerLine = line.rawLine.toLower();
        if (lowerLine.contains(QStringLiteral("version"))
            || lowerLine.contains(QStringLiteral("sample"))) {
            continue;
        }

        bool consumed = false;
        if (!line.values.empty()
            && (lowerLine.contains(QStringLiteral("field"))
                || lowerLine.contains(QStringLiteral("size")))) {
            if (fieldSizeCount < 0) {
                fieldSizeCount = static_cast<int>(line.values.front());
                qDebug() << "[TMR] Header reported field size count:" << fieldSizeCount;
            }
            if (lowerLine.contains(QStringLiteral("depth")) && line.values.size() >= 2) {
                if (depthCount < 0) {
                    depthCount = static_cast<int>(line.values[1]);
                    qDebug() << "[TMR] Header reported depth count:" << depthCount;
                }
            }
            if (line.values.size() > 1) {
                std::vector<double> possibleFieldSizes(line.values.begin() + 1, line.values.end());
                if (!possibleFieldSizes.empty()) {
                    fieldSizes = std::move(possibleFieldSizes);
                    qDebug() << "[TMR] Header line included field sizes.";
                }
            }
            consumed = true;
        }

        if (!line.values.empty() && lowerLine.contains(QStringLiteral("depth"))) {
            if (depthCount < 0) {
                depthCount = static_cast<int>(line.values.front());
                qDebug() << "[TMR] Header depth count:" << depthCount;
            }
            if (line.values.size() > 1) {
                standaloneDepths.assign(line.values.begin() + 1, line.values.end());
                if (!standaloneDepths.empty()) {
                    qDebug() << "[TMR] Header provided explicit depths:" << toQVector(standaloneDepths);
                }
            }
            consumed = true;
        }

        if (!consumed) {
            if (line.values.size() == 1 && fieldSizeCount < 0 && depthCount < 0) {
                qDebug() << "[TMR] Skipping standalone numeric metadata line:" << line.rawLine;
                continue;
            }
            dataLines.push_back(line);
        }
    }

    auto isStrictlyIncreasing = [](const std::vector<double> &values) {
        if (values.size() < 2) {
            return false;
        }
        for (size_t i = 1; i < values.size(); ++i) {
            if (!(values[i] > values[i - 1])) {
                return false;
            }
        }
        return true;
    };

    if (fieldSizes.empty()) {
        int fieldLineIndex = -1;
        if (fieldSizeCount > 0) {
            for (int i = 0; i < static_cast<int>(dataLines.size()); ++i) {
                if (static_cast<int>(dataLines[i].values.size()) == fieldSizeCount) {
                    fieldLineIndex = i;
                    break;
                }
            }
        }

        if (fieldLineIndex < 0) {
            for (int i = 0; i < static_cast<int>(dataLines.size()); ++i) {
                if (isStrictlyIncreasing(dataLines[i].values)) {
                    fieldLineIndex = i;
                    break;
                }
            }
        }

        if (fieldLineIndex >= 0) {
            fieldSizes = dataLines[fieldLineIndex].values;
            dataLines.erase(dataLines.begin() + fieldLineIndex);
        }
    }

    if (fieldSizes.empty()) {
        if (dataLines.empty()) {
            throw std::runtime_error("TMR table is missing field sizes line.");
        }
        fieldSizes = dataLines.front().values;
        dataLines.erase(dataLines.begin());
    }

    if (fieldSizes.empty()) {
        throw std::runtime_error("TMR table is missing field sizes line.");
    }

    if (fieldSizeCount <= 0) {
        fieldSizeCount = static_cast<int>(fieldSizes.size());
    }
    if (fieldSizeCount != static_cast<int>(fieldSizes.size())) {
        qWarning() << "TMR table field size count mismatch:" << fieldSizeCount << fieldSizes.size();
    }

    if (depthCount <= 0) {
        depthCount = static_cast<int>(dataLines.size());
        qWarning() << "[TMR] Depth count unspecified; using data line count" << depthCount;
    }

    std::vector<double> depths;
    std::vector<std::vector<double>> tmrValues;
    std::vector<double> depthList = standaloneDepths;

    for (const NumericLine &line : dataLines) {
        if (static_cast<int>(tmrValues.size()) >= depthCount) {
            break;
        }
        if (line.values.empty()) {
            continue;
        }

        if (depthList.empty() && depthCount > 0
            && static_cast<int>(line.values.size()) == depthCount
            && depthCount != fieldSizeCount) {
            depthList = line.values;
            qDebug() << "[TMR] Found standalone depth list:" << toQVector(depthList);
            continue;
        }

        std::vector<double> rowValues;
        double depthValue = std::numeric_limits<double>::quiet_NaN();

        if (static_cast<int>(line.values.size()) >= fieldSizeCount + 1) {
            depthValue = line.values.front();
            rowValues.assign(line.values.begin() + 1,
                             line.values.begin() + 1 + fieldSizeCount);
            if (static_cast<int>(line.values.size()) > fieldSizeCount + 1) {
                qWarning() << "TMR table row has extra columns; trimming.";
            }
        } else if (static_cast<int>(line.values.size()) == fieldSizeCount) {
            rowValues = line.values;
            if (!depthList.empty()
                && static_cast<int>(depths.size()) < static_cast<int>(depthList.size())) {
                depthValue = depthList[depths.size()];
            }
        } else {
            qWarning() << "TMR table row has insufficient columns.";
            continue;
        }

        if (static_cast<int>(rowValues.size()) != fieldSizeCount) {
            qWarning() << "TMR table row length mismatch:" << rowValues.size()
                       << fieldSizeCount;
            continue;
        }

        tmrValues.push_back(std::move(rowValues));
        if (std::isfinite(depthValue)) {
            depths.push_back(depthValue);
        }
    }

    if (tmrValues.empty()) {
        throw std::runtime_error("TMR table is missing depth rows.");
    }

    if (depths.empty() && !depthList.empty()) {
        depths = depthList;
    }

    if (!depths.empty() && static_cast<int>(depths.size()) != static_cast<int>(tmrValues.size())) {
        qWarning() << "TMR table depth count mismatch:" << depths.size() << tmrValues.size();
    }

    if (depthCount > 0 && depthCount != static_cast<int>(tmrValues.size())) {
        qWarning() << "TMR table expected" << depthCount << "rows but found" << tmrValues.size();
    }

    TMRData result;
    result.fieldSizes = std::move(fieldSizes);
    result.depths = std::move(depths);
    result.tmrValues = std::move(tmrValues);
    return result;
}

} // namespace CyberKnife

