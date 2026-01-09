#include "cyberknife/beam_data_manager.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSaveFile>
#include <QTextStream>
#include <QtDebug>
#include <QtGlobal>
#include <QStringList>
#include <QLocale>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace CyberKnife {
namespace {

constexpr std::array<double, 12> kNominalCollimators = {
    5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0};

constexpr double kLowerTmrFieldSize = 2.5;
constexpr double kUpperTmrFieldSize = 62.5;
constexpr double kTmrEpsilon = 1e-4;

double clampIndex(const std::vector<double> &values, double target)
{
    if (values.empty()) {
        return 0.0;
    }
    if (target <= values.front()) {
        return values.front();
    }
    if (target >= values.back()) {
        return values.back();
    }
    return target;
}

int findLowerIndex(const std::vector<double> &values, double target)
{
    if (values.empty()) {
        return -1;
    }
    if (values.size() == 1) {
        return 0;
    }
    if (target <= values.front()) {
        return 0;
    }
    for (int i = 0; i < static_cast<int>(values.size()) - 1; ++i) {
        if (target >= values[i] && target <= values[i + 1]) {
            return i;
        }
    }
    return static_cast<int>(values.size()) - 2;
}

double linearInterpolate(double x0, double x1, double y0, double y1, double x)
{
    if (qFuzzyCompare(x0, x1)) {
        return y0;
    }
    const double ratio = (x - x0) / (x1 - x0);
    return y0 + ratio * (y1 - y0);
}

double bilinearInterpolate(
    double x,
    double y,
    double x0,
    double x1,
    double y0,
    double y1,
    double q11,
    double q21,
    double q12,
    double q22)
{
    if (qFuzzyCompare(x0, x1) && qFuzzyCompare(y0, y1)) {
        return q11;
    }
    if (qFuzzyCompare(x0, x1)) {
        return linearInterpolate(y0, y1, q11, q12, y);
    }
    if (qFuzzyCompare(y0, y1)) {
        return linearInterpolate(x0, x1, q11, q21, x);
    }

    const double r1 = ((x1 - x) / (x1 - x0)) * q11 + ((x - x0) / (x1 - x0)) * q21;
    const double r2 = ((x1 - x) / (x1 - x0)) * q12 + ((x - x0) / (x1 - x0)) * q22;
    return ((y1 - y) / (y1 - y0)) * r1 + ((y - y0) / (y1 - y0)) * r2;
}

int findNearestNominalIndex(double collimatorSize)
{
    if (collimatorSize <= kNominalCollimators.front()) {
        return 0;
    }
    if (collimatorSize >= kNominalCollimators.back()) {
        return static_cast<int>(kNominalCollimators.size()) - 1;
    }

    int bestIndex = 0;
    double bestDiff = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(kNominalCollimators.size()); ++i) {
        const double diff = std::abs(collimatorSize - kNominalCollimators[i]);
        if (diff < bestDiff) {
            bestDiff = diff;
            bestIndex = i;
        }
    }
    return bestIndex;
}

double nominalCollimatorForIndex(int index)
{
    if (index < 0 || index >= static_cast<int>(kNominalCollimators.size())) {
        return 0.0;
    }
    return kNominalCollimators[index];
}

bool ensureTmrColumnConsistency(const BeamDataParser::TMRData &data, int requiredColumns)
{
    for (const auto &row : data.tmrValues) {
        if (static_cast<int>(row.size()) < requiredColumns) {
            return false;
        }
    }
    return true;
}

bool extrapolateLowerTmrField(BeamDataParser::TMRData &data)
{
    if (data.fieldSizes.size() < 2 || data.tmrValues.empty()) {
        return false;
    }

    if (data.fieldSizes.front() <= kLowerTmrFieldSize + kTmrEpsilon) {
        return false;
    }

    const double firstSize = data.fieldSizes.front();
    const double secondSize = data.fieldSizes[1];
    const double denominator = secondSize - firstSize;
    if (qFuzzyIsNull(denominator)) {
        return false;
    }

    if (!ensureTmrColumnConsistency(data, 2)) {
        return false;
    }

    std::vector<double> extrapolatedColumn;
    extrapolatedColumn.reserve(data.tmrValues.size());
    for (const auto &row : data.tmrValues) {
        const double y0 = row.front();
        const double y1 = row[1];
        const double slope = (y1 - y0) / denominator;
        const double extrapolated = y0 + slope * (kLowerTmrFieldSize - firstSize);
        extrapolatedColumn.push_back(extrapolated);
    }

    data.fieldSizes.insert(data.fieldSizes.begin(), kLowerTmrFieldSize);
    for (size_t i = 0; i < data.tmrValues.size(); ++i) {
        data.tmrValues[i].insert(data.tmrValues[i].begin(), extrapolatedColumn[i]);
    }
    return true;
}

bool extrapolateUpperTmrField(BeamDataParser::TMRData &data)
{
    if (data.fieldSizes.size() < 2 || data.tmrValues.empty()) {
        return false;
    }

    if (data.fieldSizes.back() >= kUpperTmrFieldSize - kTmrEpsilon) {
        return false;
    }

    const int lastIndex = static_cast<int>(data.fieldSizes.size()) - 1;
    const double lastSize = data.fieldSizes[lastIndex];
    const double penultimateSize = data.fieldSizes[lastIndex - 1];
    const double denominator = lastSize - penultimateSize;
    if (qFuzzyIsNull(denominator)) {
        return false;
    }

    if (!ensureTmrColumnConsistency(data, lastIndex + 1)) {
        return false;
    }

    std::vector<double> extrapolatedColumn;
    extrapolatedColumn.reserve(data.tmrValues.size());
    for (const auto &row : data.tmrValues) {
        const double yPrev = row[lastIndex - 1];
        const double yLast = row[lastIndex];
        const double slope = (yLast - yPrev) / denominator;
        const double extrapolated = yLast + slope * (kUpperTmrFieldSize - lastSize);
        extrapolatedColumn.push_back(extrapolated);
    }

    data.fieldSizes.push_back(kUpperTmrFieldSize);
    for (size_t i = 0; i < data.tmrValues.size(); ++i) {
        data.tmrValues[i].push_back(extrapolatedColumn[i]);
    }
    return true;
}

void extendTmrFieldBounds(BeamDataParser::TMRData &data)
{
    const bool addedLower = extrapolateLowerTmrField(data);
    const bool addedUpper = extrapolateUpperTmrField(data);

    if (addedLower) {
        qDebug() << "[TMR] Extrapolated lower field size to" << kLowerTmrFieldSize << "mm.";
    }
    if (addedUpper) {
        qDebug() << "[TMR] Extrapolated upper field size to" << kUpperTmrFieldSize << "mm.";
    }
}

int extractTrailingInteger(const QString &text)
{
    static const QRegularExpression kIntegerPattern(QStringLiteral(R"((\d+))"));
    QRegularExpressionMatchIterator it = kIntegerPattern.globalMatch(text);
    int value = -1;
    while (it.hasNext()) {
        const QRegularExpressionMatch match = it.next();
        bool ok = false;
        const int candidate = match.captured(1).toInt(&ok);
        if (ok) {
            value = candidate;
        }
    }
    return value;
}

bool extractTrailingNumber(const QString &text, double *outValue)
{
    if (!outValue) {
        return false;
    }

    static const QRegularExpression kNumberPattern(QStringLiteral(R"((\d+(?:\.\d+)?))"));
    QRegularExpressionMatchIterator it = kNumberPattern.globalMatch(text);
    bool found = false;
    while (it.hasNext()) {
        const QRegularExpressionMatch match = it.next();
        bool ok = false;
        const double candidate = match.captured(1).toDouble(&ok);
        if (ok) {
            *outValue = candidate;
            found = true;
        }
    }
    return found;
}

int resolveIndexFromFileName(const QFileInfo &fileInfo)
{
    const QString baseName = fileInfo.completeBaseName();
    int index = extractTrailingInteger(baseName);
    if (index >= 0) {
        return index;
    }

    index = extractTrailingInteger(fileInfo.fileName());
    if (index >= 0) {
        return index;
    }

    double numericValue = 0.0;
    if (extractTrailingNumber(baseName, &numericValue)
        || extractTrailingNumber(fileInfo.fileName(), &numericValue)) {
        const int nearestIndex = findNearestNominalIndex(numericValue);
        const double nominalValue = nominalCollimatorForIndex(nearestIndex);
        constexpr double kTolerance = 0.25; // allow slight rounding differences
        if (std::abs(numericValue - nominalValue) <= kTolerance) {
            qDebug() << "OCR table" << fileInfo.fileName() << "numeric suffix interpreted as index"
                     << nearestIndex << "(" << nominalValue << "mm )";
            return nearestIndex;
        }
    }

    qWarning() << "OCR table file name does not contain an index:" << fileInfo.fileName();
    return -1;
}

int computeOcrSortKey(const QFileInfo &fileInfo, int rawIndex, int fallbackSeed)
{
    if (rawIndex >= 0) {
        return rawIndex;
    }

    const int trailingNumber = extractTrailingInteger(fileInfo.fileName());
    if (trailingNumber >= 0) {
        return trailingNumber;
    }

    double numericValue = 0.0;
    if (extractTrailingNumber(fileInfo.fileName(), &numericValue)) {
        // Use tenths of millimeter precision to keep ordering stable.
        return static_cast<int>(std::round(numericValue * 10.0));
    }

    return std::numeric_limits<int>::max() - 100 + fallbackSeed;
}

int detectTableIndexOffset(const QFileInfoList &ocrFiles)
{
    int minIndex = std::numeric_limits<int>::max();
    bool anyValid = false;
    bool hasZero = false;
    bool hasOne = false;
    for (const QFileInfo &info : ocrFiles) {
        const int rawIndex = resolveIndexFromFileName(info);
        if (rawIndex < 0) {
            continue;
        }
        anyValid = true;
        minIndex = std::min(minIndex, rawIndex);
        if (rawIndex == 0) {
            hasZero = true;
        }
        if (rawIndex == 1) {
            hasOne = true;
        }
    }

    if (!anyValid) {
        return 0;
    }
    if (hasZero) {
        return 0;
    }
    if (hasOne) {
        return 1;
    }
    if (minIndex <= 0) {
        return 0;
    }
    return minIndex;
}

std::map<double, BeamDataParser::OCRData>::const_iterator selectClosestOcrIterator(
    const std::map<double, BeamDataParser::OCRData> &ocrData,
    double collimatorSize)
{
    if (ocrData.empty()) {
        return ocrData.end();
    }

    auto upper = ocrData.lower_bound(collimatorSize);
    if (upper == ocrData.end()) {
        return std::prev(ocrData.end());
    }
    if (upper == ocrData.begin()) {
        return upper;
    }

    auto lower = std::prev(upper);
    if (std::abs(collimatorSize - lower->first) <= std::abs(upper->first - collimatorSize)) {
        return lower;
    }
    return upper;
}

double clampToRange(const std::vector<double> &values, int count, double target)
{
    if (values.empty() || count <= 0) {
        return target;
    }
    const int lastIndex = std::min(static_cast<int>(values.size()), count) - 1;
    if (lastIndex < 0) {
        return target;
    }
    const double minValue = values.front();
    const double maxValue = values[static_cast<size_t>(lastIndex)];
    if (target <= minValue) {
        return minValue;
    }
    if (target >= maxValue) {
        return maxValue;
    }
    return target;
}

double interpolateOutputFactorRow(const std::vector<double> &collimatorSizes,
                                  const std::vector<double> &row,
                                  double collimatorSize)
{
    const int columnCount = std::min(static_cast<int>(collimatorSizes.size()),
                                     static_cast<int>(row.size()));
    if (columnCount <= 0) {
        return 0.0;
    }
    if (columnCount == 1) {
        return row.front();
    }

    const double clampedCollimator = clampToRange(collimatorSizes, columnCount, collimatorSize);

    int index = 0;
    if (clampedCollimator > collimatorSizes.front()) {
        index = columnCount - 2;
        for (int i = 0; i < columnCount - 1; ++i) {
            const double current = collimatorSizes[static_cast<size_t>(i)];
            const double next = collimatorSizes[static_cast<size_t>(i + 1)];
            if (clampedCollimator >= current && clampedCollimator <= next) {
                index = i;
                break;
            }
        }
    }

    const int upperIndex = index + 1;
    const double x0 = collimatorSizes[static_cast<size_t>(index)];
    const double x1 = collimatorSizes[static_cast<size_t>(upperIndex)];
    const double y0 = row[static_cast<size_t>(index)];
    const double y1 = row[static_cast<size_t>(upperIndex)];
    return linearInterpolate(x0, x1, y0, y1, clampedCollimator);
}

void trimDmColumns(BeamDataParser::DMData &data)
{
    int columnCount = static_cast<int>(data.collimatorSizes.size());
    if (columnCount <= 0) {
        data.collimatorSizes.clear();
        data.outputFactors.clear();
        data.outputFactorMatrix.clear();
        return;
    }

    int effectiveColumnCount = columnCount;
    for (const auto &row : data.outputFactorMatrix) {
        if (row.empty()) {
            effectiveColumnCount = 0;
            break;
        }
        effectiveColumnCount = std::min(effectiveColumnCount, static_cast<int>(row.size()));
    }

    if (effectiveColumnCount <= 0) {
        data.collimatorSizes.clear();
        data.outputFactors.clear();
        data.outputFactorMatrix.clear();
        return;
    }

    if (effectiveColumnCount < columnCount) {
        data.collimatorSizes.resize(effectiveColumnCount);
    }

    if (static_cast<int>(data.outputFactors.size()) > effectiveColumnCount) {
        data.outputFactors.resize(effectiveColumnCount);
    }

    for (auto &row : data.outputFactorMatrix) {
        if (static_cast<int>(row.size()) > effectiveColumnCount) {
            row.resize(effectiveColumnCount);
        }
    }
}

void normalizeDmDepths(BeamDataParser::DMData &data)
{
    if (data.outputFactorMatrix.empty()) {
        return;
    }

    if (data.depths.empty()) {
        return;
    }

    const size_t rowCount = std::min(data.depths.size(), data.outputFactorMatrix.size());
    data.depths.resize(rowCount);
    data.outputFactorMatrix.resize(rowCount);

    std::vector<size_t> indices(rowCount);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
        return data.depths[lhs] < data.depths[rhs];
    });

    std::vector<double> sortedDepths;
    sortedDepths.reserve(rowCount);
    std::vector<std::vector<double>> sortedMatrix;
    sortedMatrix.reserve(rowCount);

    for (size_t idx : indices) {
        sortedDepths.push_back(data.depths[idx]);
        sortedMatrix.push_back(data.outputFactorMatrix[idx]);
    }

    data.depths = std::move(sortedDepths);
    data.outputFactorMatrix = std::move(sortedMatrix);
}

void refreshPrimaryOutputFactors(BeamDataParser::DMData &data)
{
    if (!data.outputFactorMatrix.empty()) {
        data.outputFactors = data.outputFactorMatrix.front();
    }
    if (!data.depths.empty()) {
        data.depth = data.depths.front();
    }
}

void normalizeDmData(BeamDataParser::DMData &data)
{
    trimDmColumns(data);
    normalizeDmDepths(data);
    refreshPrimaryOutputFactors(data);
}

} // namespace

bool BeamDataManager::loadBeamData(const QString &dataDirectory)
{
    m_validationErrors.clear();
    m_isDataLoaded = false;
    m_ocrData.clear();
    m_ocrIndexToCollimator.clear();
    m_ocrTableIndexOffset = 0;

    QDir dir(dataDirectory);
    if (!dir.exists()) {
        m_validationErrors << QStringLiteral("データディレクトリが存在しません: %1").arg(dataDirectory);
        return false;
    }

    qDebug() << "Loading CyberKnife beam data from" << dir.absolutePath();

    bool dmLoaded = false;
    bool tmrLoaded = false;
    bool ocrLoaded = false;

    const QString dmTablePath = dir.filePath(QStringLiteral("DMTable.dat"));
    try {
        m_dmData = BeamDataParser::parseDMTable(dmTablePath);
        normalizeDmData(m_dmData);
        dmLoaded = !m_dmData.collimatorSizes.empty();
        qDebug() << "DM table loaded with" << m_dmData.collimatorSizes.size() << "entries";
    } catch (const std::exception &ex) {
        const QString error = QStringLiteral("DMTable.dat の読み込みに失敗しました: %1").arg(ex.what());
        qWarning() << error;
        m_validationErrors << error;
    }

    const QString tmrTablePath = dir.filePath(QStringLiteral("TMRtable.dat"));
    try {
        m_tmrData = BeamDataParser::parseTMRTable(tmrTablePath);
        tmrLoaded = !m_tmrData.fieldSizes.empty();
        if (tmrLoaded) {
            extendTmrFieldBounds(m_tmrData);
        }
        qDebug() << "TMR table loaded with" << m_tmrData.fieldSizes.size() << "field sizes";
    } catch (const std::exception &ex) {
        const QString error = QStringLiteral("TMRtable.dat の読み込みに失敗しました: %1").arg(ex.what());
        qWarning() << error;
        m_validationErrors << error;
    }

    const QFileInfoList ocrFiles = dir.entryInfoList(QStringList() << QStringLiteral("OCRtable*.dat"), QDir::Files | QDir::Readable);
    m_ocrTableIndexOffset = detectTableIndexOffset(ocrFiles);
    if (!ocrFiles.isEmpty()) {
        qDebug() << "Detected OCR table index offset:" << m_ocrTableIndexOffset;
    }
    struct OcrRecord {
        QFileInfo fileInfo;
        BeamDataParser::OCRData data;
        double originalCollimator = 0.0;
        int rawIndex = -1;
        int normalizedIndex = -1;
        int headerIndex = -1;
        int assignedIndex = -1;
        int sortKey = 0;
    };

    std::vector<OcrRecord> ocrRecords;
    ocrRecords.reserve(static_cast<size_t>(ocrFiles.size()));
    int fallbackSeed = 0;
    for (const QFileInfo &fileInfo : ocrFiles) {
        try {
            OcrRecord record;
            record.fileInfo = fileInfo;
            record.data = BeamDataParser::parseOCRTable(fileInfo.absoluteFilePath());
            record.originalCollimator = record.data.collimatorSize;
            record.rawIndex = resolveIndexFromFileName(fileInfo);
            record.sortKey = computeOcrSortKey(fileInfo, record.rawIndex, fallbackSeed++);
            if (!qFuzzyIsNull(record.originalCollimator)) {
                record.headerIndex = findNearestNominalIndex(record.originalCollimator);
            }
            ocrRecords.push_back(std::move(record));
        } catch (const std::exception &ex) {
            const QString error = QStringLiteral("%1 の読み込みに失敗しました: %2")
                                      .arg(fileInfo.fileName(), ex.what());
            qWarning() << error;
            m_validationErrors << error;
        }
    }

    if (!ocrRecords.empty()) {
        for (OcrRecord &record : ocrRecords) {
            if (record.rawIndex >= 0) {
                record.normalizedIndex = record.rawIndex - m_ocrTableIndexOffset;
                if (record.normalizedIndex < 0
                    || record.normalizedIndex >= static_cast<int>(kNominalCollimators.size())) {
                    qWarning() << "OCR table index" << record.rawIndex << "(offset" << m_ocrTableIndexOffset
                               << ") is out of range for" << record.fileInfo.fileName();
                    record.normalizedIndex = -1;
                }
            }
        }

        std::sort(ocrRecords.begin(), ocrRecords.end(), [](const OcrRecord &lhs, const OcrRecord &rhs) {
            if (lhs.sortKey != rhs.sortKey) {
                return lhs.sortKey < rhs.sortKey;
            }
            return lhs.fileInfo.fileName() < rhs.fileInfo.fileName();
        });

        std::vector<bool> indexUsed(kNominalCollimators.size(), false);
        auto assignIndex = [&](OcrRecord &record, int index, const char *reason) {
            if (index < 0 || index >= static_cast<int>(kNominalCollimators.size())) {
                return;
            }
            if (indexUsed[index]) {
                return;
            }
            const double resolvedCollimator = nominalCollimatorForIndex(index);
            if (!qFuzzyCompare(1.0 + resolvedCollimator, 1.0 + record.originalCollimator)) {
                qDebug() << "OCR table" << record.fileInfo.fileName() << '(' << reason
                         << ") collimator adjusted" << record.originalCollimator << "->" << resolvedCollimator;
            }
            record.data.collimatorSize = resolvedCollimator;
            record.data.nominalIndex = index;
            record.assignedIndex = index;
            indexUsed[index] = true;
        };

        for (OcrRecord &record : ocrRecords) {
            if (record.normalizedIndex >= 0) {
                assignIndex(record, record.normalizedIndex, "filename index");
            }
        }

        for (OcrRecord &record : ocrRecords) {
            if (record.assignedIndex >= 0) {
                continue;
            }
            if (record.headerIndex >= 0) {
                assignIndex(record, record.headerIndex, "header nearest");
            }
        }

        for (OcrRecord &record : ocrRecords) {
            if (record.assignedIndex >= 0) {
                continue;
            }
            auto it = std::find(indexUsed.begin(), indexUsed.end(), false);
            if (it == indexUsed.end()) {
                qWarning() << "No remaining OCR slots available for" << record.fileInfo.fileName();
                continue;
            }
            const int fallbackIndex = static_cast<int>(std::distance(indexUsed.begin(), it));
            assignIndex(record, fallbackIndex, "fallback");
        }

        for (OcrRecord &record : ocrRecords) {
            if (record.assignedIndex < 0) {
                continue;
            }
            const double resolvedCollimator = record.data.collimatorSize;
            if (m_ocrData.find(resolvedCollimator) != m_ocrData.end()) {
                qWarning() << "Duplicate OCR table for collimator" << resolvedCollimator << "mm detected."
                           << "Existing:" << m_ocrData[resolvedCollimator].sourceFileName << "New:"
                           << record.fileInfo.fileName();
            }
            m_ocrIndexToCollimator[record.assignedIndex] = resolvedCollimator;
            record.data.sourceFileName = record.fileInfo.fileName();
            qDebug() << "OCR table" << record.fileInfo.fileName() << "assigned to index" << record.assignedIndex
                     << "->" << resolvedCollimator << "mm";
            m_ocrData[resolvedCollimator] = std::move(record.data);
        }
    }
    qDebug() << "=== Loaded OCR Data ===";
    for (const auto &pair : m_ocrData) {
        qDebug() << "Collimator:" << pair.first << "Depths:" << pair.second.depths.size()
                 << "Radii:" << pair.second.radii.size();
    }
    ocrLoaded = !m_ocrData.empty();
    if (!ocrLoaded) {
        const QString error = QStringLiteral("OCRtable*.dat が見つかりませんでした。");
        qWarning() << error;
        m_validationErrors << error;
    }

    m_isDataLoaded = dmLoaded && tmrLoaded && ocrLoaded;

    if (!m_isDataLoaded) {
        m_validationErrors << QStringLiteral("必要なビームデータが不足しています。");
    }

    return m_isDataLoaded;
}

bool BeamDataManager::isDataLoaded() const
{
    return m_isDataLoaded;
}

double BeamDataManager::getOutputFactor(double collimatorSize, double depth) const
{
    qDebug() << "★getOutputFactor called: collimator=" << collimatorSize
             << "depth=" << depth;

    if (!m_isDataLoaded || m_dmData.collimatorSizes.empty()
        || m_dmData.outputFactorMatrix.empty()) {
        qWarning() << "★DM data not loaded or empty";
        return 0.0;
    }

    // ★利用可能なコリメータサイズを表示★
    QStringList availableSizes;
    for (double size : m_dmData.collimatorSizes) {
        availableSizes << QString::number(size, 'f', 1);
    }
    qDebug() << "★Available DM collimator sizes:" << availableSizes;
    qDebug() << "★DM collimator range: [" 
             << m_dmData.collimatorSizes.front() << "," 
             << m_dmData.collimatorSizes.back() << "]";

    auto interpolateAtDepth = [&](int depthIndex) {
        if (depthIndex < 0
            || depthIndex >= static_cast<int>(m_dmData.outputFactorMatrix.size())) {
            return 0.0;
        }
        const std::vector<double> &row = m_dmData.outputFactorMatrix[depthIndex];
        return interpolateOutputFactorRow(m_dmData.collimatorSizes, row, collimatorSize);
    };

    double result = 0.0;
    if (!m_dmData.depths.empty()
        && m_dmData.depths.size() == m_dmData.outputFactorMatrix.size()) {
        const double clampedDepth = clampIndex(m_dmData.depths, depth);
        int depthIndex = findLowerIndex(m_dmData.depths, clampedDepth);
        if (depthIndex < 0) {
            depthIndex = 0;
        }
        int nextDepthIndex = std::min(depthIndex + 1,
                                      static_cast<int>(m_dmData.depths.size()) - 1);

        const double d0 = m_dmData.depths[static_cast<size_t>(depthIndex)];
        const double d1 = m_dmData.depths[static_cast<size_t>(nextDepthIndex)];
        const double v0 = interpolateAtDepth(depthIndex);
        const double v1 = interpolateAtDepth(nextDepthIndex);

        if (qFuzzyCompare(d0, d1)) {
            result = v0;
        } else {
            result = linearInterpolate(d0, d1, v0, v1, clampedDepth);
        }
    } else {
        result = interpolateAtDepth(0);
    }

    qDebug() << "★Output Factor interpolation result:" << result;

    return result;
}

const BeamDataParser::OCRData *BeamDataManager::findClosestOcrTable(double collimatorSize, double *matchedCollimator) const
{
    if (!m_isDataLoaded || m_ocrData.empty()) {
        if (matchedCollimator) {
            *matchedCollimator = 0.0;
        }
        return nullptr;
    }

    std::map<double, BeamDataParser::OCRData>::const_iterator selected = m_ocrData.end();
    if (!m_ocrIndexToCollimator.empty()) {
        const int nominalIndex = findNearestNominalIndex(collimatorSize);
        auto indexIt = m_ocrIndexToCollimator.find(nominalIndex);
        if (indexIt != m_ocrIndexToCollimator.end()) {
            auto exactIt = m_ocrData.find(indexIt->second);
            if (exactIt != m_ocrData.end()) {
                selected = exactIt;
            }
        }
    }

    if (selected == m_ocrData.end()) {
        selected = selectClosestOcrIterator(m_ocrData, collimatorSize);
    }
    if (selected == m_ocrData.end()) {
        if (matchedCollimator) {
            *matchedCollimator = 0.0;
        }
        return nullptr;
    }

    if (matchedCollimator) {
        *matchedCollimator = selected->first;
    }
    return &selected->second;
}

double BeamDataManager::getOCRRatio(double collimatorSize, double depth, double radius) const
{
    qDebug() << "OCR Request - Collimator:" << collimatorSize << "Depth:" << depth << "Radius:" << radius;

    if (!m_isDataLoaded) {
        qWarning() << "OCR Request aborted: ビームデータが読み込まれていません。";
        return 0.0;
    }
    if (m_ocrData.empty()) {
        qWarning() << "OCR Request aborted: 利用可能なOCRテーブルが存在しません。";
        return 0.0;
    }

    QStringList availableCollimators;
    for (const auto &pair : m_ocrData) {
        availableCollimators << QString::number(pair.first, 'f', 1);
    }
    qDebug() << "Available collimators:" << availableCollimators;

    double matchedCollimator = 0.0;
    const BeamDataParser::OCRData *ocr = findClosestOcrTable(collimatorSize, &matchedCollimator);
    if (!ocr) {
        qWarning() << "OCR Request failed: 適切なOCRテーブルが見つかりません。";
        return 0.0;
    }

    qDebug() << "Selected collimator:" << matchedCollimator;

    const double diff = std::abs(collimatorSize - matchedCollimator);
    constexpr double warningThreshold = 2.0;
    if (diff >= warningThreshold) {
        qWarning() << "要求コリメータ" << collimatorSize << "mm と使用テーブル" << matchedCollimator
                   << "mm の差が" << diff << "mm あります。";
    }

    const BeamDataParser::OCRData &ocrRef = *ocr;
    const int depthIndex = findLowerIndex(ocrRef.depths, depth);
    const int radiusIndex = findLowerIndex(ocrRef.radii, radius);
    if (depthIndex < 0) {
        qWarning() << "OCR Request failed: 深さ" << depth << "mm がテーブル範囲外です。";
        return 0.0;
    }
    if (radiusIndex < 0) {
        qWarning() << "OCR Request failed: 半径" << radius << "mm がテーブル範囲外です。";
        return 0.0;
    }

    const int nextDepthIndex = std::min(depthIndex + 1, static_cast<int>(ocrRef.depths.size()) - 1);
    const int nextRadiusIndex = std::min(radiusIndex + 1, static_cast<int>(ocrRef.radii.size()) - 1);

    const double d0 = ocrRef.depths[depthIndex];
    const double d1 = ocrRef.depths[nextDepthIndex];
    const double r0 = ocrRef.radii[radiusIndex];
    const double r1 = ocrRef.radii[nextRadiusIndex];

    const double q11 = ocrRef.ratios[depthIndex][radiusIndex];
    const double q21 = ocrRef.ratios[depthIndex][nextRadiusIndex];
    const double q12 = ocrRef.ratios[nextDepthIndex][radiusIndex];
    const double q22 = ocrRef.ratios[nextDepthIndex][nextRadiusIndex];

    const double clampedDepth = clampIndex(ocrRef.depths, depth);
    const double clampedRadius = clampIndex(ocrRef.radii, radius);

    return bilinearInterpolate(clampedRadius, clampedDepth, r0, r1, d0, d1, q11, q21, q12, q22);
}

double BeamDataManager::getTMRValue(double fieldSize, double depth) const
{
    if (!m_isDataLoaded || m_tmrData.fieldSizes.empty()) {
        return 0.0;
    }

    const int depthIndex = findLowerIndex(m_tmrData.depths, depth);
    if (depthIndex < 0) {
        return 0.0;
    }
    const int nextDepthIndex = std::min(depthIndex + 1, static_cast<int>(m_tmrData.depths.size()) - 1);

    const int fieldIndex = findLowerIndex(m_tmrData.fieldSizes, fieldSize);
    if (fieldIndex < 0) {
        return 0.0;
    }
    const int nextFieldIndex = std::min(fieldIndex + 1, static_cast<int>(m_tmrData.fieldSizes.size()) - 1);

    const double depth0 = m_tmrData.depths[depthIndex];
    const double depth1 = m_tmrData.depths[nextDepthIndex];
    const double field0 = m_tmrData.fieldSizes[fieldIndex];
    const double field1 = m_tmrData.fieldSizes[nextFieldIndex];

    const double q11 = m_tmrData.tmrValues[depthIndex][fieldIndex];
    const double q21 = m_tmrData.tmrValues[depthIndex][nextFieldIndex];
    const double q12 = m_tmrData.tmrValues[nextDepthIndex][fieldIndex];
    const double q22 = m_tmrData.tmrValues[nextDepthIndex][nextFieldIndex];

    const double clampedDepth = clampIndex(m_tmrData.depths, depth);
    const double clampedField = clampIndex(m_tmrData.fieldSizes, fieldSize);

    return bilinearInterpolate(clampedField, clampedDepth, field0, field1, depth0, depth1, q11, q21, q12, q22);
}

bool BeamDataManager::validateData()
{
    bool isValid = true;

    auto appendError = [&](const QString &message) {
        if (!m_validationErrors.contains(message)) {
            m_validationErrors << message;
        }
        qWarning() << message;
        isValid = false;
    };

    if (!m_isDataLoaded) {
        appendError(QStringLiteral("ビームデータが読み込まれていません。"));
        return false;
    }

    if (m_dmData.collimatorSizes.empty() || m_dmData.outputFactorMatrix.empty()) {
        appendError(QStringLiteral("DMTable: コリメータサイズまたは出力係数が空です。"));
    } else {
        const int columnCount = static_cast<int>(m_dmData.collimatorSizes.size());
        if (!m_dmData.outputFactors.empty()
            && static_cast<int>(m_dmData.outputFactors.size()) != columnCount) {
            appendError(QStringLiteral("DMTable: コリメータサイズと一次出力係数列の数が一致しません。"));
        }

        for (int rowIndex = 0; rowIndex < static_cast<int>(m_dmData.outputFactorMatrix.size());
             ++rowIndex) {
            const auto &row = m_dmData.outputFactorMatrix[rowIndex];
            if (static_cast<int>(row.size()) != columnCount) {
                appendError(QStringLiteral("DMTable: 深さインデックス %1 の列数がコリメータ数(%2)と一致しません。")
                                .arg(rowIndex)
                                .arg(columnCount));
                break;
            }
        }

        if (!m_dmData.depths.empty()
            && static_cast<int>(m_dmData.depths.size()) != static_cast<int>(m_dmData.outputFactorMatrix.size())) {
            appendError(QStringLiteral("DMTable: 深さの数と出力係数行の数が一致しません。"));
        }
    }

    if (m_tmrData.fieldSizes.empty() || m_tmrData.depths.empty()) {
        appendError(QStringLiteral("TMRtable: フィールドサイズまたは深さのデータが不足しています。"));
    } else if (m_tmrData.tmrValues.size() != m_tmrData.depths.size()) {
        appendError(QStringLiteral("TMRtable: 深さの数とTMR値の行数が一致しません。"));
    }

    for (const auto &pair : m_ocrData) {
        const BeamDataParser::OCRData &ocr = pair.second;
        const QString prefix = QStringLiteral("OCRtable(コリメータ %1 mm): ")
                                   .arg(QString::number(pair.first, 'f', 1));

        if (ocr.nominalIndex < 0
            || ocr.nominalIndex >= static_cast<int>(kNominalCollimators.size())) {
            appendError(prefix
                        + QStringLiteral("公称コリメータ番号が不正です (値: %1)。")
                              .arg(ocr.nominalIndex));
        }
        if (ocr.depths.empty()) {
            appendError(prefix + QStringLiteral("深さデータが空です。"));
        }
        if (ocr.radii.empty()) {
            appendError(prefix + QStringLiteral("半径データが空です。"));
        }
        if (!ocr.depths.empty() && ocr.ratios.size() != ocr.depths.size()) {
            appendError(prefix + QStringLiteral("深さの数と比率行数が一致しません。"));
        }
        if (!ocr.radii.empty()) {
            for (int depthIndex = 0; depthIndex < static_cast<int>(ocr.ratios.size()); ++depthIndex) {
                const auto &row = ocr.ratios[depthIndex];
                if (row.size() != ocr.radii.size()) {
                    appendError(prefix
                                + QStringLiteral("深さインデックス %1 の比率列数(%2)が半径数(%3)と一致しません。")
                                      .arg(depthIndex)
                                      .arg(row.size())
                                      .arg(ocr.radii.size()));
                }
            }
        }
    }

    return isValid;
}

QStringList BeamDataManager::getAvailableCollimatorSizes() const
{
    QStringList sizes;
    for (const auto &pair : m_ocrData) {
        sizes << QString::number(pair.first, 'f', 1);
    }
    if (sizes.isEmpty()) {
        for (double value : m_dmData.collimatorSizes) {
            sizes << QString::number(value, 'f', 1);
        }
    }
    return sizes;
}

QStringList BeamDataManager::getValidationErrors() const
{
    return m_validationErrors;
}

namespace {

QString formatDouble(double value, int precision)
{
    const QLocale locale = QLocale::c();
    return locale.toString(value, 'f', precision);
}

} // namespace

bool BeamDataManager::exportDmDataToCsv(const QString &filePath) const
{
    if (!m_isDataLoaded || m_dmData.collimatorSizes.empty()
        || m_dmData.outputFactorMatrix.empty()) {
        return false;
    }

    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream stream(&file);
    stream.setLocale(QLocale::c());

    stream << QStringLiteral("Depth(mm)");
    for (double collimator : m_dmData.collimatorSizes) {
        stream << QLatin1Char(',') << formatDouble(collimator, 1);
    }
    stream << QLatin1Char('\n');

    const int rowCount = static_cast<int>(m_dmData.outputFactorMatrix.size());
    const int depthCount = static_cast<int>(m_dmData.depths.size());
    for (int row = 0; row < rowCount; ++row) {
        double depthValue = m_dmData.depth;
        if (row < depthCount) {
            depthValue = m_dmData.depths[row];
        } else if (row > 0 && depthCount == 0) {
            depthValue = m_dmData.depth + row;
        }

        stream << formatDouble(depthValue, 3);

        const std::vector<double> &values = m_dmData.outputFactorMatrix[row];
        const int columnCount = std::min(static_cast<int>(values.size()),
                                         static_cast<int>(m_dmData.collimatorSizes.size()));
        for (int column = 0; column < columnCount; ++column) {
            stream << QLatin1Char(',') << formatDouble(values[column], 6);
        }

        // Fill remaining columns with blanks if matrix is short.
        for (int column = columnCount;
             column < static_cast<int>(m_dmData.collimatorSizes.size());
             ++column) {
            stream << QLatin1Char(',');
        }

        stream << QLatin1Char('\n');
    }

    return file.commit();
}

bool BeamDataManager::exportTmrDataToCsv(const QString &filePath) const
{
    if (!m_isDataLoaded || m_tmrData.fieldSizes.empty()
        || m_tmrData.tmrValues.empty()) {
        return false;
    }

    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream stream(&file);
    stream.setLocale(QLocale::c());

    stream << QStringLiteral("Depth(mm)");
    for (double fieldSize : m_tmrData.fieldSizes) {
        stream << QLatin1Char(',') << formatDouble(fieldSize, 1);
    }
    stream << QLatin1Char('\n');

    const int rowCount = static_cast<int>(m_tmrData.tmrValues.size());
    const int depthCount = static_cast<int>(m_tmrData.depths.size());
    for (int row = 0; row < rowCount; ++row) {
        double depthValue = (row < depthCount && !m_tmrData.depths.empty())
            ? m_tmrData.depths[row]
            : static_cast<double>(row);
        stream << formatDouble(depthValue, 3);

        const std::vector<double> &values = m_tmrData.tmrValues[row];
        const int columnCount = std::min(static_cast<int>(values.size()),
                                         static_cast<int>(m_tmrData.fieldSizes.size()));
        for (int column = 0; column < columnCount; ++column) {
            stream << QLatin1Char(',') << formatDouble(values[column], 6);
        }
        for (int column = columnCount;
             column < static_cast<int>(m_tmrData.fieldSizes.size());
             ++column) {
            stream << QLatin1Char(',');
        }
        stream << QLatin1Char('\n');
    }

    return file.commit();
}

bool BeamDataManager::exportOcrDataToCsv(const QString &filePath,
                                         double minCollimator,
                                         double maxCollimator) const
{
    if (!m_isDataLoaded || m_ocrData.empty()) {
        return false;
    }

    const double lowerBound = std::min(minCollimator, maxCollimator);
    const double upperBound = std::max(minCollimator, maxCollimator);

    bool hasDataInRange = false;
    for (const auto &pair : m_ocrData) {
        const double collimator = pair.first;
        if (collimator + 1e-6 < lowerBound || collimator - 1e-6 > upperBound) {
            continue;
        }
        const auto &ocr = pair.second;
        if (!ocr.depths.empty() && !ocr.radii.empty() && !ocr.ratios.empty()) {
            hasDataInRange = true;
            break;
        }
    }

    if (!hasDataInRange) {
        return false;
    }

    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream stream(&file);
    stream.setLocale(QLocale::c());
    stream << QStringLiteral("Collimator(mm),Depth(mm),Radius(mm),OCR Ratio")
           << QLatin1Char('\n');

    for (const auto &pair : m_ocrData) {
        const double collimator = pair.first;
        if (collimator + 1e-6 < lowerBound || collimator - 1e-6 > upperBound) {
            continue;
        }
        const auto &ocr = pair.second;
        const int depthCount = static_cast<int>(ocr.depths.size());
        const int radiusCount = static_cast<int>(ocr.radii.size());
        if (depthCount == 0 || radiusCount == 0
            || static_cast<int>(ocr.ratios.size()) < depthCount) {
            continue;
        }

        for (int depthIndex = 0; depthIndex < depthCount; ++depthIndex) {
            const std::vector<double> &row = ocr.ratios[depthIndex];
            const int valueCount = std::min(static_cast<int>(row.size()), radiusCount);
            for (int radiusIndex = 0; radiusIndex < valueCount; ++radiusIndex) {
                stream << formatDouble(collimator, 1) << QLatin1Char(',')
                       << formatDouble(ocr.depths[depthIndex], 3) << QLatin1Char(',')
                       << formatDouble(ocr.radii[radiusIndex], 3) << QLatin1Char(',')
                       << formatDouble(row[radiusIndex], 6) << QLatin1Char('\n');
            }
        }
    }

    return file.commit();
}

bool BeamDataManager::exportAllDataToCsv(const QString &directoryPath,
                                         QStringList *exportedFiles) const
{
    if (!m_isDataLoaded) {
        return false;
    }

    QDir dir(directoryPath);
    if (dir.path().isEmpty()) {
        return false;
    }
    if (!dir.exists()) {
        if (!dir.mkpath(QStringLiteral("."))) {
            return false;
        }
    }

    const QString timestamp =
        QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));

    QStringList generated;
    bool success = true;

    const auto recordResult = [&](const QString &filePath, bool ok) {
        if (ok) {
            generated.append(filePath);
        } else {
            success = false;
        }
    };

    const QString dmPath =
        dir.filePath(QStringLiteral("DMTable_%1.csv").arg(timestamp));
    recordResult(dmPath, exportDmDataToCsv(dmPath));

    const QString ocrPath =
        dir.filePath(QStringLiteral("OCRTables_%1.csv").arg(timestamp));
    recordResult(ocrPath, exportOcrDataToCsv(ocrPath));

    const QString tmrPath =
        dir.filePath(QStringLiteral("TMRTable_%1.csv").arg(timestamp));
    recordResult(tmrPath, exportTmrDataToCsv(tmrPath));

    if (exportedFiles) {
        *exportedFiles = generated;
    }

    return success;
}

} // namespace CyberKnife
