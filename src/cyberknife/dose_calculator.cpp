#include "cyberknife/dose_calculator.h"

#include "cyberknife/beam_data_locator.h"
#include "cyberknife/beam_data_manager.h"
#include "cyberknife/geometry_calculator.h"
#include "dicom/dicom_volume.h"
#include "dicom/rtdose_volume.h"

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QtGlobal>
#include <QtMath>
#include <QFile>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QSettings>
#include <QSet>
#include <QString>
#include <QTextStream>
#include <QtConcurrent/QtConcurrentMap>

#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>

using CyberKnife::BeamDataParser;
using CyberKnife::CyberKnifeDoseLog;

namespace {

constexpr int kDepthHounsfieldThreshold = -800;
constexpr double kOffAxisLimitMultiplier = 2.0;
constexpr double kBeamSamplingStepSizeMm = 1.0;
//constexpr int kBeamSamplingMaxSteps = 2000;
constexpr int kBeamSamplingMaxSteps = 1300;
constexpr int kBeamSamplingExitTolerance = 100;
constexpr double kReferenceSadMm = 800.0;
constexpr double kMaxTmrDepthMm = 449.99999;
constexpr double kMaxOcrDepthMm = 249.99999;
constexpr double kMaxOcrRadiusMm = 59.99999;
constexpr double kRayTracingDoseScale = 0.01;

struct BeamDepthProfile {
    double entryDistance = std::numeric_limits<double>::quiet_NaN();
    double stepSize = kBeamSamplingStepSizeMm;
    QVector<double> cumulativeDepths;

    bool isValid() const
    {
        return std::isfinite(entryDistance) && !cumulativeDepths.isEmpty() && stepSize > 0.0;
    }

    double interpolate(double sad) const
    {
        if (!isValid()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (!std::isfinite(sad) || sad <= entryDistance) {
            return 0.0;
        }

        const double relative = sad - entryDistance;
        if (relative <= 0.0) {
            return 0.0;
        }

        const double index = relative / stepSize;
        if (index < 1.0) {
            return cumulativeDepths[0] * index;
        }

        const int sampleCount = cumulativeDepths.size();
        if (sampleCount == 1) {
            return cumulativeDepths[0];
        }

        const int lower = static_cast<int>(std::floor(index));
        if (lower >= sampleCount - 1) {
            const double lastValue = cumulativeDepths[sampleCount - 1];
            const double prevValue = cumulativeDepths[sampleCount - 2];
            const double slope = (lastValue - prevValue);
            const double extra = index - (sampleCount - 1);
            return lastValue + slope * extra;
        }

        const int upper = qBound(0, lower + 1, sampleCount - 1);
        const double lowerValue = cumulativeDepths[lower];
        const double upperValue = cumulativeDepths[upper];
        const double fraction = index - static_cast<double>(lower);
        return lowerValue + (upperValue - lowerValue) * fraction;
    }
};

struct DensityTableEntry {
    double rawValue = 0.0;
    double density = 0.0;
};

struct DensityTableState {
    QVector<DensityTableEntry> entries;
    double offset = 1000.0;
    QString source;
};

struct RefineResult {
    bool success = false;
    int calculatedCount = 0;
    int interpolatedCount = 0;
    int totalRegionVoxels = 0;

    double interpolationRatio() const {
        if (totalRegionVoxels == 0) return 0.0;
        return static_cast<double>(interpolatedCount) / static_cast<double>(totalRegionVoxels);
    }
};

QVector<DensityTableEntry> defaultDensityTable()
{
    return {
        {223.0, 0.0},
        {224.0, 0.20000000298023224},
        {502.0, 0.49000000953674316},
        {966.0, 0.94900000095367432},
        {990.0, 0.97600001096725464},
        {1019.0, 1.0},
        {1077.0, 1.0429999828338623},
        {1078.0, 1.0520000457763672},
        {1268.0, 1.1169999837875366},
        {1930.0, 1.4559999704360962},
        {3593.0, 2.3069999217987061},
    };
}

void normalizeDensityEntries(QVector<DensityTableEntry> &entries)
{
    std::sort(entries.begin(), entries.end(), [](const DensityTableEntry &a, const DensityTableEntry &b) {
        return a.rawValue < b.rawValue;
    });

    auto uniqueEnd = std::unique(entries.begin(), entries.end(), [](const DensityTableEntry &a, const DensityTableEntry &b) {
        return qFuzzyCompare(a.rawValue, b.rawValue);
    });
    entries.erase(uniqueEnd, entries.end());
}

QVector<DensityTableEntry> loadDensityTableFromFile(const QString &path)
{
    QVector<DensityTableEntry> entries;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return entries;
    }

    QTextStream stream(&file);
    while (!stream.atEnd()) {
        const QString line = stream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) {
            continue;
        }

        QString cleaned = line;
        cleaned.replace(QLatin1Char(','), QLatin1Char(' '));
        cleaned.replace(QLatin1Char(';'), QLatin1Char(' '));
        cleaned.replace(QLatin1Char('\t'), QLatin1Char(' '));

        const QStringList parts = cleaned.split(QLatin1Char(' '), Qt::SkipEmptyParts);
        if (parts.size() < 2) {
            continue;
        }

        bool okRaw = false;
        bool okDensity = false;
        const double rawValue = parts[0].toDouble(&okRaw);
        const double density = parts[1].toDouble(&okDensity);
        if (okRaw && okDensity) {
            entries.append({rawValue, density});
        }
    }

    return entries;
}

DensityTableState loadInitialDensityTableState()
{
    DensityTableState localState;

    QSettings settings("ShioRIS3", "ShioRIS3");
    localState.offset = settings.value(QStringLiteral("cyberknife/densityTableOffset"), 1000.0).toDouble();
    const QString customFile = settings.value(QStringLiteral("cyberknife/densityTableFile")).toString().trimmed();

    QSet<QString> seen;
    QVector<QString> candidates;

    auto addCandidate = [&](const QString &absolutePath) {
        if (absolutePath.isEmpty()) {
            return;
        }
        QFileInfo info(absolutePath);
        const QString resolved = info.absoluteFilePath();
        if (seen.contains(resolved)) {
            return;
        }
        if (!info.exists()) {
            return;
        }
        seen.insert(resolved);
        candidates.append(resolved);
    };

    auto addRelativeCandidate = [&](const QDir &baseDir, const QString &relativePath) {
        if (relativePath.isEmpty()) {
            return;
        }
        addCandidate(baseDir.absoluteFilePath(relativePath));
    };

    if (!customFile.isEmpty()) {
        QFileInfo info(customFile);
        if (info.isAbsolute()) {
            addCandidate(info.absoluteFilePath());
        } else {
            addCandidate(QDir::current().absoluteFilePath(customFile));
            if (QCoreApplication::instance()) {
                addRelativeCandidate(QDir(QCoreApplication::applicationDirPath()), customFile);
            }
        }
    }

    if (QCoreApplication::instance()) {
        QDir appDir(QCoreApplication::applicationDirPath());
        addRelativeCandidate(appDir, QStringLiteral("cyberknife_density_table.csv"));
        addRelativeCandidate(appDir, QStringLiteral("resources/cyberknife_density_table.csv"));
        addRelativeCandidate(appDir, QStringLiteral("../resources/cyberknife_density_table.csv"));
    }

    addRelativeCandidate(QDir::current(), QStringLiteral("cyberknife_density_table.csv"));
    addRelativeCandidate(QDir::current(), QStringLiteral("resources/cyberknife_density_table.csv"));

    for (const QString &candidate : candidates) {
        QVector<DensityTableEntry> loaded = loadDensityTableFromFile(candidate);
        if (!loaded.isEmpty()) {
            normalizeDensityEntries(loaded);
            localState.entries = loaded;
            localState.source = candidate;
            break;
        }
    }

    if (localState.entries.isEmpty() && !customFile.isEmpty()) {
        qCWarning(CyberKnifeDoseLog)
            << "Failed to load CT electron density table from" << customFile
            << ". Falling back to defaults.";
    }

    if (localState.entries.isEmpty()) {
        QVector<DensityTableEntry> defaults = defaultDensityTable();
        normalizeDensityEntries(defaults);
        localState.entries = defaults;
        localState.source = QStringLiteral("built-in defaults");
    }

    if (!localState.entries.isEmpty()) {
        qCInfo(CyberKnifeDoseLog)
            << "Loaded CT electron density table from" << localState.source
            << "(" << localState.entries.size() << "points, offset" << localState.offset << ")";
    } else {
        qCWarning(CyberKnifeDoseLog)
            << "Failed to prepare CT electron density table. Falling back to linear conversion.";
    }

    return localState;
}

std::once_flag &densityTableInitFlag()
{
    static std::once_flag flag;
    return flag;
}

std::mutex &densityTableMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::shared_ptr<const DensityTableState> &densityTableBaseState()
{
    static std::shared_ptr<const DensityTableState> baseState;
    return baseState;
}

std::shared_ptr<const DensityTableState> &densityTableCurrentState()
{
    static std::shared_ptr<const DensityTableState> currentState;
    return currentState;
}

std::atomic<quint64> &densityTableVersionCounter()
{
    static std::atomic<quint64> version{0};
    return version;
}

void ensureDensityTableInitialized()
{
    std::call_once(densityTableInitFlag(), []() {
        DensityTableState initialState = loadInitialDensityTableState();
        auto initialPtr = std::make_shared<DensityTableState>(std::move(initialState));
        {
            std::lock_guard<std::mutex> lock(densityTableMutex());
            densityTableBaseState() = initialPtr;
            densityTableCurrentState() = initialPtr;
            densityTableVersionCounter().store(1, std::memory_order_release);
        }
    });
}

std::shared_ptr<const DensityTableState> densityTableSnapshot()
{
    ensureDensityTableInitialized();
    std::shared_ptr<const DensityTableState> current = std::atomic_load(&densityTableCurrentState());
    if (current) {
        return current;
    }

    std::lock_guard<std::mutex> lock(densityTableMutex());
    current = densityTableCurrentState();
    if (!current) {
        current = densityTableBaseState();
        std::atomic_store(&densityTableCurrentState(), current);
    }
    return current;
}

quint64 densityTableCurrentVersion()
{
    ensureDensityTableInitialized();
    return densityTableVersionCounter().load(std::memory_order_acquire);
}

bool setDensityTableOverride(const DensityTableState &state)
{
    if (state.entries.isEmpty()) {
        return false;
    }

    ensureDensityTableInitialized();
    std::shared_ptr<const DensityTableState> newState = std::make_shared<DensityTableState>(state);
    {
        std::lock_guard<std::mutex> lock(densityTableMutex());
        std::atomic_store(&densityTableCurrentState(), newState);
    }

    densityTableVersionCounter().fetch_add(1, std::memory_order_acq_rel);
    qCInfo(CyberKnifeDoseLog)
        << "Applied CT electron density table from" << newState->source
        << "(" << newState->entries.size() << "points, offset" << newState->offset << ")";
    return true;
}

void restoreDensityTableToBase()
{
    ensureDensityTableInitialized();
    std::shared_ptr<const DensityTableState> base = densityTableBaseState();
    if (!base) {
        return;
    }

    std::shared_ptr<const DensityTableState> current = std::atomic_load(&densityTableCurrentState());
    if (current == base) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(densityTableMutex());
        std::atomic_store(&densityTableCurrentState(), base);
    }

    densityTableVersionCounter().fetch_add(1, std::memory_order_acq_rel);
    qCInfo(CyberKnifeDoseLog)
        << "Restored CT electron density table from" << base->source
        << "(" << base->entries.size() << "points, offset" << base->offset << ")";
}

class HounsfieldConverter {
public:
    static constexpr int kMinHU = -1200;
    static constexpr int kMaxHU = 4000;

    static double convertToPhysicalDensity(int hounsfield)
    {
        static thread_local quint64 cachedVersion = std::numeric_limits<quint64>::max();
        static thread_local std::shared_ptr<const DensityTableState> cachedState;
        static thread_local const DensityTableState *cachedPtr = nullptr;

        const quint64 currentVersion = densityTableCurrentVersion();
        if (!cachedState || cachedVersion != currentVersion) {
            cachedState = densityTableSnapshot();
            cachedPtr = cachedState.get();
            cachedVersion = currentVersion;
        }

        const DensityTableState *state = cachedPtr;
        if (state && !state->entries.isEmpty()) {
            const double rawValue = static_cast<double>(hounsfield) + state->offset;
            if (state->entries.size() == 1) {
                return qMax(0.0, state->entries.first().density);
            }

            if (rawValue <= state->entries.first().rawValue) {
                return qMax(0.0, state->entries.first().density);
            }
            if (rawValue >= state->entries.last().rawValue) {
                return qMax(0.0, state->entries.last().density);
            }

            auto upperIt = std::lower_bound(state->entries.begin(), state->entries.end(), rawValue,
                                            [](const DensityTableEntry &entry, double value) {
                                                return entry.rawValue < value;
                                            });
            if (upperIt == state->entries.begin()) {
                return qMax(0.0, upperIt->density);
            }
            const auto lowerIt = upperIt - 1;
            const double ratio = (rawValue - lowerIt->rawValue) / (upperIt->rawValue - lowerIt->rawValue);
            const double density = lowerIt->density + (upperIt->density - lowerIt->density) * ratio;
            return qMax(0.0, density);
        }

        hounsfield = qBound(kMinHU, hounsfield, kMaxHU);
        const double density = 1.0 + static_cast<double>(hounsfield) / 1000.0;
        return qMax(0.0, density);
    }
};

int lowerIndex(const QVector<float> &values, double target)
{
    if (values.isEmpty()) {
        return -1;
    }
    if (values.size() == 1) {
        return 0;
    }
    auto it = std::lower_bound(values.begin(), values.end(), static_cast<float>(target));
    if (it == values.begin()) {
        return 0;
    }
    if (it == values.end()) {
        return values.size() - 2;
    }
    const int upperIndex = static_cast<int>(std::distance(values.begin(), it));
    return qMax(0, upperIndex - 1);
}

double sample1D(const QVector<float> &xs, const QVector<float> &ys, double x)
{
    if (xs.isEmpty() || ys.isEmpty()) {
        return 0.0;
    }
    if (xs.size() == 1) {
        return ys.first();
    }
    if (x <= xs.first()) {
        return ys.first();
    }
    if (x >= xs.last()) {
        return ys.last();
    }

    auto it = std::lower_bound(xs.begin(), xs.end(), static_cast<float>(x));
    int upperIndex = static_cast<int>(std::distance(xs.begin(), it));
    upperIndex = qBound(1, upperIndex, xs.size() - 1);
    const int lower = upperIndex - 1;

    const float x0 = xs[lower];
    const float x1 = xs[upperIndex];
    const float y0 = ys[lower];
    const float y1 = ys[upperIndex];

    if (qFuzzyCompare(x0, x1)) {
        return y0;
    }

    const double ratio = (x - x0) / (x1 - x0);
    return y0 + (y1 - y0) * ratio;
}

template <typename Container>
double clampToContainerRange(const Container &values, double value)
{
    if (values.empty()) {
        return value;
    }
    if (value <= values.front()) {
        return values.front();
    }
    if (value >= values.back()) {
        return values.back();
    }
    return value;
}

double sample2D(const QVector<float> &depths,
                const QVector<float> &radii,
                const QVector<float> &table,
                int radiusCount,
                double depth,
                double radius)
{
    if (depths.isEmpty() || radii.isEmpty() || table.isEmpty() || radiusCount <= 0) {
        return 1.0;
    }

    const int depthLower = lowerIndex(depths, depth);
    const int radiusLower = lowerIndex(radii, radius);
    const int depthUpper = qBound(depthLower, depthLower + 1, depths.size() - 1);
    const int radiusUpper = qBound(radiusLower, radiusLower + 1, radii.size() - 1);

    const float d0 = depths[depthLower];
    const float d1 = depths[depthUpper];
    const float r0 = radii[radiusLower];
    const float r1 = radii[radiusUpper];

    auto idx = [&](int d, int r) { return d * radiusCount + r; };

    const float q11 = table[idx(depthLower, radiusLower)];
    const float q21 = table[idx(depthLower, radiusUpper)];
    const float q12 = table[idx(depthUpper, radiusLower)];
    const float q22 = table[idx(depthUpper, radiusUpper)];

    if (qFuzzyCompare(d0, d1) && qFuzzyCompare(r0, r1)) {
        return q11;
    }

    const double clampedDepth = clampToContainerRange(depths, depth);
    const double clampedRadius = clampToContainerRange(radii, radius);

    double t = 0.0;
    double u = 0.0;
    if (!qFuzzyCompare(r0, r1)) {
        u = (clampedRadius - r0) / (r1 - r0);
    }
    if (!qFuzzyCompare(d0, d1)) {
        t = (clampedDepth - d0) / (d1 - d0);
    }

    const double c00 = q11 + (q21 - q11) * u;
    const double c10 = q12 + (q22 - q12) * u;
    return c00 + (c10 - c00) * t;
}

template <typename Container>
int lowerIndexForContainer(const Container &values, double target)
{
    if (values.empty()) {
        return -1;
    }
    if (values.size() == 1) {
        return 0;
    }

    auto it = std::lower_bound(values.begin(), values.end(), target);
    if (it == values.begin()) {
        return 0;
    }
    if (it == values.end()) {
        return static_cast<int>(values.size()) - 2;
    }

    const int upperIndex = static_cast<int>(std::distance(values.begin(), it));
    return std::max(0, upperIndex - 1);
}

double sampleOcrFromManagerTable(const BeamDataParser::OCRData &ocr, double depth, double radius)
{
    if (ocr.depths.empty() || ocr.radii.empty() || ocr.ratios.empty()) {
        return 1.0;
    }

    const int depthLower = lowerIndexForContainer(ocr.depths, depth);
    const int radiusLower = lowerIndexForContainer(ocr.radii, radius);
    if (depthLower < 0 || radiusLower < 0) {
        return 1.0;
    }

    const int depthUpper = std::min(depthLower + 1, static_cast<int>(ocr.depths.size()) - 1);
    const int radiusUpper = std::min(radiusLower + 1, static_cast<int>(ocr.radii.size()) - 1);

    const double d0 = ocr.depths[depthLower];
    const double d1 = ocr.depths[depthUpper];
    const double r0 = ocr.radii[radiusLower];
    const double r1 = ocr.radii[radiusUpper];

    auto ratioAt = [&](int depthIndex, int radiusIndex) -> double {
        if (ocr.ratios.empty()) {
            return 1.0;
        }
        depthIndex = std::max(0, std::min(depthIndex, static_cast<int>(ocr.ratios.size()) - 1));
        const auto &row = ocr.ratios[depthIndex];
        if (row.empty()) {
            return 1.0;
        }
        radiusIndex = std::max(0, std::min(radiusIndex, static_cast<int>(row.size()) - 1));
        return row[static_cast<size_t>(radiusIndex)];
    };

    const double q11 = ratioAt(depthLower, radiusLower);
    const double q21 = ratioAt(depthLower, radiusUpper);
    const double q12 = ratioAt(depthUpper, radiusLower);
    const double q22 = ratioAt(depthUpper, radiusUpper);

    if (qFuzzyCompare(d0, d1) && qFuzzyCompare(r0, r1)) {
        return q11;
    }

    const double clampedDepth = clampToContainerRange(ocr.depths, depth);
    const double clampedRadius = clampToContainerRange(ocr.radii, radius);

    double u = 0.0;
    double t = 0.0;
    if (!qFuzzyCompare(r0, r1)) {
        u = (clampedRadius - r0) / (r1 - r0);
    }
    if (!qFuzzyCompare(d0, d1)) {
        t = (clampedDepth - d0) / (d1 - d0);
    }

    const double c00 = q11 + (q21 - q11) * u;
    const double c10 = q12 + (q22 - q12) * u;
    return c00 + (c10 - c00) * t;
}

double computeEffectiveFieldSize(double collimatorSize, double sad)
{
    if (!std::isfinite(collimatorSize) || collimatorSize <= 0.0) {
        return 0.0;
    }
    if (!std::isfinite(sad) || sad <= 0.0) {
        return collimatorSize;
    }
    return collimatorSize * sad / kReferenceSadMm;
}

struct ManagerBeamDataSample {
    double outputFactor = 0.0;
    double ocrRatio = 1.0;
    double tmr = 0.0;
    bool outputValid = false;
    bool ocrValid = false;
    bool tmrValid = false;
};

ManagerBeamDataSample sampleManagerBeamData(const CyberKnife::BeamDataManager &manager,
                                           const BeamDataParser::OCRData *ocr,
                                           double collimatorSize,
                                           double effectiveFieldSize,
                                           double tmrDepth,
                                           double ocrDepth,
                                           double ocrRadius)
{
    ManagerBeamDataSample sample;

    sample.outputFactor = manager.getOutputFactor(collimatorSize, ocrDepth);
    sample.outputValid = std::isfinite(sample.outputFactor) && sample.outputFactor > 0.0;

    if (ocr) {
        sample.ocrRatio = sampleOcrFromManagerTable(*ocr, ocrDepth, ocrRadius);
        sample.ocrValid = std::isfinite(sample.ocrRatio) && sample.ocrRatio > 0.0;
    } else {
        sample.ocrRatio = manager.getOCRRatio(collimatorSize, ocrDepth, ocrRadius);
        sample.ocrValid = std::isfinite(sample.ocrRatio) && sample.ocrRatio > 0.0;
    }

    const double fieldForManager = (std::isfinite(effectiveFieldSize) && effectiveFieldSize > 0.0)
                                       ? effectiveFieldSize
                                       : collimatorSize;
    sample.tmr = manager.getTMRValue(fieldForManager, tmrDepth);
    sample.tmrValid = std::isfinite(sample.tmr) && sample.tmr > 0.0;

    qCDebug(CyberKnifeDoseLog) << "Manager dose factors" << "collimator" << collimatorSize
                               << "effectiveField" << fieldForManager << "tmrDepth" << tmrDepth
                               << "ocrDepth" << ocrDepth << "ocrRadius" << ocrRadius
                               << "OF" << sample.outputFactor << "OCR" << sample.ocrRatio
                               << "TMR" << sample.tmr;

    return sample;
}

double sampleHounsfieldAtPatientPoint(const DicomVolume &volume, const QVector3D &patientPoint)
{
    const cv::Mat &data = volume.data();
    if (data.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int width = volume.width();
    const int height = volume.height();
    const int depth = volume.depth();

    QVector3D voxel = volume.patientToVoxelContinuous(patientPoint);
    const double x = voxel.x();
    const double y = voxel.y();
    const double z = voxel.z();

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    const double dx = x - static_cast<double>(x0);
    const double dy = y - static_cast<double>(y0);
    const double dz = z - static_cast<double>(z0);

    auto sample = [&](int xi, int yi, int zi) -> double {
        if (xi < 0 || xi >= width || yi < 0 || yi >= height || zi < 0 || zi >= depth) {
            return -1000.0;
        }
        const short *slice = data.ptr<short>(zi);
        const int index2D = yi * width + xi;
        return static_cast<double>(slice[index2D]);
    };

    const double v000 = sample(x0, y0, z0);
    const double v100 = sample(x1, y0, z0);
    const double v010 = sample(x0, y1, z0);
    const double v110 = sample(x1, y1, z0);
    const double v001 = sample(x0, y0, z1);
    const double v101 = sample(x1, y0, z1);
    const double v011 = sample(x0, y1, z1);
    const double v111 = sample(x1, y1, z1);

    const double c00 = v000 + (v100 - v000) * dx;
    const double c10 = v010 + (v110 - v010) * dx;
    const double c01 = v001 + (v101 - v001) * dx;
    const double c11 = v011 + (v111 - v011) * dx;

    const double c0 = c00 + (c10 - c00) * dy;
    const double c1 = c01 + (c11 - c01) * dy;

    return c0 + (c1 - c0) * dz;
}

BeamDepthProfile precomputeBeamDepths(const DicomVolume &volume,
                                      const CyberKnife::GeometryCalculator::BeamGeometry &beam)
{
    BeamDepthProfile profile;
    const cv::Mat &data = volume.data();
    if (data.empty()) {
        return profile;
    }

    QVector3D direction = (beam.targetPosition - beam.sourcePosition);
    if (direction.lengthSquared() <= 0.0f) {
        return profile;
    }
    direction.normalize();

    const QVector3D stepVector = direction * profile.stepSize;
    QVector3D position = beam.sourcePosition;
    double cumulativeDepth = 0.0;
    bool inside = false;
    int exitCounter = 0;
    double entryDistance = std::numeric_limits<double>::quiet_NaN();

    for (int step = 0; step < kBeamSamplingMaxSteps; ++step) {
        const double huValue = sampleHounsfieldAtPatientPoint(volume, position);
        const int hu = std::isfinite(huValue) ? static_cast<int>(std::round(huValue)) : -1000;

        if (hu > kDepthHounsfieldThreshold) {
            if (!inside) {
                inside = true;
                entryDistance = static_cast<double>(step) * profile.stepSize;
            }
            cumulativeDepth += HounsfieldConverter::convertToPhysicalDensity(hu) * profile.stepSize;
            exitCounter = 0;
        } else if (inside) {
            ++exitCounter;
            if (exitCounter > kBeamSamplingExitTolerance) {
                break;
            }
        } else {
            // Move to next position before continuing
            position += stepVector;
            continue;
        }

        if (inside) {
            profile.cumulativeDepths.append(cumulativeDepth);
        }

        position += stepVector;
    }

    if (inside && std::isfinite(entryDistance)) {
        profile.entryDistance = entryDistance;
    }

    return profile;
}

std::function<double(int, int, int)> makeVoxelCalculator(const DicomVolume &volume,
                                                         CyberKnife::CyberKnifeDoseCalculator *calculator,
                                                         const CyberKnife::GeometryCalculator::BeamGeometry &beam,
                                                         std::shared_ptr<BeamDepthProfile> depthProfile,
                                                         double beamWeight,
                                                         std::atomic<bool> *cancelRequested,
                                                         std::atomic<int> *progressCounter,
                                                         int totalVoxels,
                                                         std::function<void(int)> progressCallback)
{
    const cv::Mat data = volume.data();
    const int width = volume.width();
    const int height = volume.height();
    const int depth = volume.depth();
    const int planeSize = width * height;
    const DicomVolume *volumePtr = &volume;
    CyberKnife::CyberKnifeDoseCalculator *calculatorPtr = calculator;
    const CyberKnife::GeometryCalculator::BeamGeometry beamCopy = beam;
    std::function<void(int)> progressCallbackCopy = std::move(progressCallback);

    std::shared_ptr<std::atomic<int>> progressPercent;
    if (progressCallbackCopy) {
        progressPercent = std::make_shared<std::atomic<int>>(-1);
    }

    CyberKnife::CyberKnifeDoseCalculator::FastLookupContext fastContext;
    if (calculatorPtr) {
        fastContext = calculatorPtr->buildFastLookupContext(beamCopy.collimatorSize);
    }

    return [data,
            width,
            height,
            planeSize,
            depth,
            volumePtr,
            calculatorPtr,
            beamCopy,
            depthProfile,
            beamWeight,
            cancelRequested,
            progressCounter,
            totalVoxels,
            progressCallbackCopy,
            progressPercent,
            fastContext](int x, int y, int z) -> double {
        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            return 0.0;
        }

        const short *slice = data.ptr<short>(z);
        const int index2D = y * width + x;
        const int hu = static_cast<int>(slice[index2D]);
        if (hu < kDepthHounsfieldThreshold) {
            return 0.0;
        }

        const double density = HounsfieldConverter::convertToPhysicalDensity(hu);

        QVector3D patientPoint = volumePtr->voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
        const double depthAlongBeam =
            CyberKnife::GeometryCalculator::calculateDepth(beamCopy, patientPoint);
        const double offAxis =
            CyberKnife::GeometryCalculator::calculateOffAxisDistance(beamCopy, patientPoint);
        const double ssd = CyberKnife::GeometryCalculator::calculateSSD(beamCopy, patientPoint);
        const QVector3D beamCoords =
            CyberKnife::GeometryCalculator::patientToBeamCoordinate(patientPoint, beamCopy);
        const double radiusSad = std::hypot(beamCoords.x(), beamCoords.y());
        double radius800 = radiusSad;
        if (ssd > 0.0 && std::isfinite(ssd)) {
            radius800 = radiusSad * (kReferenceSadMm / ssd);
        }
        double referenceOffAxisLimit = fastContext.hasOffAxisLimit
                                           ? fastContext.offAxisLimit
                                           : beamCopy.collimatorSize * kOffAxisLimitMultiplier;
        if (!std::isfinite(referenceOffAxisLimit) || referenceOffAxisLimit <= 0.0) {
            referenceOffAxisLimit = beamCopy.collimatorSize * kOffAxisLimitMultiplier;
        }
        const double offAxisLimit = referenceOffAxisLimit;
        if (radius800 > offAxisLimit) {
            return 0.0;
        }

        double depthValue = std::numeric_limits<double>::quiet_NaN();
        if (depthProfile && depthProfile->isValid()) {
            depthValue = depthProfile->interpolate(depthAlongBeam);
        }
        if (!std::isfinite(depthValue)) {
            depthValue = depthAlongBeam;
        }

        CyberKnife::CyberKnifeDoseCalculator::CalculationPoint cp{patientPoint, density};
        double baseDose = calculatorPtr->calculatePointDoseWithContext(cp,
                                                                       beamCopy,
                                                                       depthValue,
                                                                       offAxis,
                                                                       100.0,
                                                                       fastContext);
        double weightedDose = baseDose * beamWeight;

        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            return 0.0;
        }

        if (calculatorPtr &&
            calculatorPtr->doseModel() ==
                CyberKnife::CyberKnifeDoseCalculator::DoseModel::PrimaryPlusScatter) {
            double scatterFactor = 1.0;
            const int radius = 1;
            double densitySum = 0.0;
            int densityCount = 0;
            bool cancelled = false;
            for (int dz = -radius; dz <= radius; ++dz) {
                if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
                    cancelled = true;
                    break;
                }
                const int zi = z + dz;
                if (zi < 0 || zi >= depth) {
                    continue;
                }
                const short *neighborSlice = data.ptr<short>(zi);
                for (int dy = -radius; dy <= radius; ++dy) {
                    if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
                        cancelled = true;
                        break;
                    }
                    const int yi = y + dy;
                    if (yi < 0 || yi >= height) {
                        continue;
                    }
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
                            cancelled = true;
                            break;
                        }
                        const int xi = x + dx;
                        if (xi < 0 || xi >= width) {
                            continue;
                        }
                        const int neighborIndex = yi * width + xi;
                        const int neighborHu = static_cast<int>(neighborSlice[neighborIndex]);
                        if (neighborHu < kDepthHounsfieldThreshold) {
                            continue;
                        }
                        densitySum += HounsfieldConverter::convertToPhysicalDensity(neighborHu);
                        ++densityCount;
                    }
                    if (cancelled) {
                        break;
                    }
                }
                if (cancelled) {
                    break;
                }
            }
            if (cancelled) {
                return 0.0;
            }
            if (densityCount > 0) {
                const double averageDensity = densitySum / static_cast<double>(densityCount);
                const double deviation = averageDensity - 1.0;
                const double maxAdjustment = 0.10;
                const double adjustment = std::clamp(deviation * 0.25, -maxAdjustment, maxAdjustment);
                scatterFactor = 1.0 + adjustment;
            }
            weightedDose *= scatterFactor;
        }

        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            return 0.0;
        }

        if (progressCallbackCopy && progressCounter &&
            (!cancelRequested || !cancelRequested->load(std::memory_order_relaxed))) {
            const int done = progressCounter->fetch_add(1, std::memory_order_relaxed) + 1;
            if (totalVoxels > 0) {
                const int percent = static_cast<int>((static_cast<long long>(done) * 100) / totalVoxels);
                if (percent >= 0 && percent <= 100) {
                    int expected = progressPercent->load(std::memory_order_relaxed);
                    while (percent > expected &&
                           !progressPercent->compare_exchange_weak(expected, percent, std::memory_order_relaxed)) {
                    }
                    if (percent > expected) {
                        progressCallbackCopy(percent);
                    }
                }
            }
        }

        return weightedDose;
    };
}

QVector<int> generateSteppedIndices(int size, int step)
{
    QVector<int> indices;
    if (size <= 0) {
        return indices;
    }

    step = qMax(1, step);
    indices.append(0);
    for (int value = step; value < size - 1; value += step) {
        indices.append(value);
    }
    if (indices.back() != size - 1) {
        indices.append(size - 1);
    }
    return indices;
}

int computeSteppedIndexCount(int size, int step)
{
    if (size <= 0) {
        return 0;
    }

    step = qMax(1, step);

    int count = 1; // Always include index 0 when size > 0
    for (int value = step; value < size - 1; value += step) {
        ++count;
    }

    if (size > 1) {
        ++count; // Ensure the last voxel (size - 1) is included
    }

    return count;
}

/**
 * @brief Generates indices that are new for the current step size, excluding those from coarser steps
 *
 * For hierarchical refinement, only points that weren't calculated in previous passes should be computed.
 * Example: size=9, currentStep=2, previousStep=4
 *   - currentStep indices:  [0, 2, 4, 6, 8]
 *   - previousStep indices: [0, 4, 8]
 *   - new indices:          [2, 6]  (only points not in previousStep)
 *
 * @param size The dimension size
 * @param currentStep Current step size
 * @param previousStep Previous (coarser) step size (0 means no previous step)
 * @return Indices that should be newly calculated at this step
 */
QVector<int> generateNewSteppedIndices(int size, int currentStep, int previousStep)
{
    QVector<int> currentIndices = generateSteppedIndices(size, currentStep);

    // If no previous step, all current indices are new
    if (previousStep <= 0) {
        return currentIndices;
    }

    QSet<int> previousSet;
    QVector<int> previousIndices = generateSteppedIndices(size, previousStep);
    for (int idx : previousIndices) {
        previousSet.insert(idx);
    }

    // Filter out indices that were already calculated in previous step
    QVector<int> newIndices;
    for (int idx : currentIndices) {
        if (!previousSet.contains(idx)) {
            newIndices.append(idx);
        }
    }

    return newIndices;
}

QVector<int> buildPrevIndexMap(int size, const QVector<int> &indices)
{
    QVector<int> map(size, indices.isEmpty() ? 0 : indices.first());
    if (indices.isEmpty()) {
        return map;
    }

    int current = 0;
    for (int pos = 0; pos < size; ++pos) {
        while (current + 1 < indices.size() && pos >= indices[current + 1]) {
            ++current;
        }
        map[pos] = indices[current];
    }
    return map;
}

QVector<int> buildNextIndexMap(int size, const QVector<int> &indices)
{
    QVector<int> map(size, indices.isEmpty() ? 0 : indices.last());
    if (indices.isEmpty()) {
        return map;
    }

    int current = indices.size() - 1;
    for (int pos = size - 1; pos >= 0; --pos) {
        while (current - 1 >= 0 && pos <= indices[current - 1]) {
            --current;
        }
        map[pos] = indices[current];
    }
    return map;
}

int interpolateDoseVolume(RTDoseVolume &doseVolume,
                          const QVector<quint8> &mask,
                          const QVector<int> &xIndices,
                          const QVector<int> &yIndices,
                          const QVector<int> &zIndices,
                          std::atomic<bool> *cancelRequested = nullptr)
{
    if (mask.isEmpty()) {
        return 0;
    }

    const int width = doseVolume.width();
    const int height = doseVolume.height();
    const int depth = doseVolume.depth();

    if (width <= 0 || height <= 0 || depth <= 0) {
        return 0;
    }

    cv::Mat &volumeMat = doseVolume.data();
    if (volumeMat.empty()) {
        return 0;
    }

    // Build index maps once for all threads to share
    QVector<int> prevX = buildPrevIndexMap(width, xIndices);
    QVector<int> nextX = buildNextIndexMap(width, xIndices);
    QVector<int> prevY = buildPrevIndexMap(height, yIndices);
    QVector<int> nextY = buildNextIndexMap(height, yIndices);
    QVector<int> prevZ = buildPrevIndexMap(depth, zIndices);
    QVector<int> nextZ = buildNextIndexMap(depth, zIndices);

    const int planeSize = width * height;
    std::atomic<int> interpolatedCount{0};

    // Parallelize over Z slices for better performance
    QVector<int> allZIndices;
    allZIndices.reserve(depth);
    for (int z = 0; z < depth; ++z) {
        allZIndices.append(z);
    }

    QtConcurrent::blockingMap(allZIndices, [&](int z) {
        // Check cancellation at the beginning of each slice (not in inner loops)
        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            return;
        }

        float *slice = volumeMat.ptr<float>(z);
        const quint8 *maskSlice = mask.constData() + z * planeSize;
        const int z0 = prevZ[z];
        const int z1 = nextZ[z];
        const float zRatio = (z1 == z0) ? 0.0f : static_cast<float>(z - z0) / static_cast<float>(z1 - z0);

        const float *sliceZ0 = volumeMat.ptr<float>(z0);
        const float *sliceZ1 = volumeMat.ptr<float>(z1);

        int localCount = 0;

        for (int y = 0; y < height; ++y) {
            float *row = slice + y * width;
            const quint8 *maskRow = maskSlice + y * width;
            if (!maskRow) {
                continue;
            }

            const int y0 = prevY[y];
            const int y1 = nextY[y];
            const float yRatio = (y1 == y0) ? 0.0f : static_cast<float>(y - y0) / static_cast<float>(y1 - y0);

            const float *rowZ0Y0 = sliceZ0 + y0 * width;
            const float *rowZ0Y1 = sliceZ0 + y1 * width;
            const float *rowZ1Y0 = sliceZ1 + y0 * width;
            const float *rowZ1Y1 = sliceZ1 + y1 * width;

            for (int x = 0; x < width; ++x) {
                if (maskRow[x]) {
                    continue;
                }

                const int x0 = prevX[x];
                const int x1 = nextX[x];
                const float xRatio = (x1 == x0) ? 0.0f : static_cast<float>(x - x0) / static_cast<float>(x1 - x0);

                const float v000 = rowZ0Y0[x0];
                const float v100 = rowZ0Y0[x1];
                const float v010 = rowZ0Y1[x0];
                const float v110 = rowZ0Y1[x1];
                const float v001 = rowZ1Y0[x0];
                const float v101 = rowZ1Y0[x1];
                const float v011 = rowZ1Y1[x0];
                const float v111 = rowZ1Y1[x1];

                const float c00 = v000 + (v100 - v000) * xRatio;
                const float c10 = v010 + (v110 - v010) * xRatio;
                const float c01 = v001 + (v101 - v001) * xRatio;
                const float c11 = v011 + (v111 - v011) * xRatio;

                const float c0 = c00 + (c10 - c00) * yRatio;
                const float c1 = c01 + (c11 - c01) * yRatio;

                row[x] = c0 + (c1 - c0) * zRatio;
                ++localCount;
            }
        }

        // Accumulate local count to global count once per slice
        if (localCount > 0) {
            interpolatedCount.fetch_add(localCount, std::memory_order_relaxed);
        }
    });

    return interpolatedCount.load(std::memory_order_relaxed);
}

/**
 * @brief Identifies voxels in the dose volume that exceed a threshold value
 * @param doseVolume The dose volume to analyze
 * @param threshold The dose threshold (e.g., 0.5 * maxDose)
 * @return A mask where 1 indicates voxels >= threshold, 0 otherwise
 */
QVector<quint8> identifyHighDoseRegion(const RTDoseVolume &doseVolume, float threshold)
{
    const cv::Mat &volume = doseVolume.data();
    if (volume.empty()) {
        return QVector<quint8>();
    }

    const int width = volume.size[2];
    const int height = volume.size[1];
    const int depth = volume.size[0];
    const qint64 totalVoxels = static_cast<qint64>(width) * height * depth;

    if (totalVoxels > std::numeric_limits<int>::max()) {
        qWarning() << "Volume too large for high-dose region identification";
        return QVector<quint8>();
    }

    QVector<quint8> mask(static_cast<int>(totalVoxels), quint8(0));

    for (int z = 0; z < depth; ++z) {
        const float *slice = volume.ptr<float>(z);
        for (int y = 0; y < height; ++y) {
            const float *row = slice + y * width;
            for (int x = 0; x < width; ++x) {
                const float dose = row[x];
                if (dose >= threshold) {
                    const int index = z * height * width + y * width + x;
                    mask[index] = 1;
                }
            }
        }
    }

    return mask;
}

/**
 * @brief Refines specific regions of the dose volume with a finer step size
 * @param volume CT volume for density lookup
 * @param doseVolume Dose volume to refine (modified in-place)
 * @param regionMask Mask indicating which voxels to refine (1 = refine, 0 = skip)
 * @param stepXY Step size for XY plane
 * @param stepZ Step size for Z axis
 * @param previousStepXY Previous (coarser) step size for XY (0 means no previous step)
 * @param previousStepZ Previous (coarser) step size for Z (0 means no previous step)
 * @param calculator Voxel dose calculation function
 * @param cancelRequested Optional cancellation flag (checked periodically)
 * @return RefineResult containing success status, calculated count, interpolated count, and total region voxels
 */
RefineResult refineRegionWithStep(const DicomVolume &volume,
                                  RTDoseVolume &doseVolume,
                                  const QVector<quint8> &regionMask,
                                  int stepXY,
                                  int stepZ,
                                  int previousStepXY,
                                  int previousStepZ,
                                  const std::function<double(int, int, int)> &calculator,
                                  std::atomic<bool> *cancelRequested)
{
    RefineResult result;
    result.success = true;

    const int width = volume.width();
    const int height = volume.height();
    const int depth = volume.depth();

    if (regionMask.isEmpty() || !calculator) {
        return result;
    }

    // Count total voxels in the region
    for (int i = 0; i < regionMask.size(); ++i) {
        if (regionMask[i] > 0) {
            ++result.totalRegionVoxels;
        }
    }

    // Generate indices for points that are NEW at this step size
    // (excluding points already calculated in the previous coarser step)
    QVector<int> xIndicesNew = generateNewSteppedIndices(width, stepXY, previousStepXY);
    QVector<int> yIndicesNew = generateNewSteppedIndices(height, stepXY, previousStepXY);
    QVector<int> zIndicesNew = generateNewSteppedIndices(depth, stepZ, previousStepZ);

    // Also generate ALL indices at the current step size (for interpolation later)
    QVector<int> xIndicesAll = generateSteppedIndices(width, stepXY);
    QVector<int> yIndicesAll = generateSteppedIndices(height, stepXY);
    QVector<int> zIndicesAll = generateSteppedIndices(depth, stepZ);

    // Create a mask to track which voxels we compute in this pass
    QVector<quint8> computedMask(regionMask.size(), quint8(0));

    // Compute dose only for voxels in the high-dose region (only NEW points)
    QVector<int> relevantSlices;
    for (int z : zIndicesNew) {
        bool hasRelevantVoxels = false;
        for (int y : yIndicesNew) {
            for (int x : xIndicesNew) {
                const int index = z * height * width + y * width + x;
                if (regionMask[index] > 0) {
                    hasRelevantVoxels = true;
                    break;
                }
            }
            if (hasRelevantVoxels) break;
        }
        if (hasRelevantVoxels) {
            relevantSlices.append(z);
        }
    }

    // Parallel calculation on relevant slices
    std::atomic<int> calculatedCounter{0};
    QtConcurrent::blockingMap(relevantSlices, [&](int z) {
        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            return;
        }
        float *slice = doseVolume.data().ptr<float>(z);
        for (int y : yIndicesNew) {
            if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
                return;
            }
            float *row = slice + y * width;
            for (int x : xIndicesNew) {
                if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
                    return;
                }
                const int index = z * height * width + y * width + x;
                if (regionMask[index] > 0) {
                    const double value = calculator(x, y, z);
                    row[x] = static_cast<float>(value);
                    computedMask[index] = 1;
                    calculatedCounter.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    result.calculatedCount = calculatedCounter.load(std::memory_order_relaxed);

    // If step size > 1, interpolate within the refined region
    if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
        result.success = false;
        return result;
    }

    if (stepXY > 1 || stepZ > 1) {
        // Mark points for interpolation within the high-dose region
        // Points already calculated (either in this pass or previous passes) are marked as 1 (skip interpolation)
        // Points to be interpolated are marked as 0
        QVector<quint8> interpolationMask(regionMask.size(), quint8(1));

        // Within the high-dose region, mark calculated grid points as "skip"
        // and intermediate points as "interpolate"
        QSet<int> calculatedXSet, calculatedYSet, calculatedZSet;
        for (int x : xIndicesAll) calculatedXSet.insert(x);
        for (int y : yIndicesAll) calculatedYSet.insert(y);
        for (int z : zIndicesAll) calculatedZSet.insert(z);

        const int total = regionMask.size();
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int index = z * height * width + y * width + x;
                    if (regionMask[index] > 0) {
                        // If this point is on the calculation grid, mark it as calculated (skip interpolation)
                        if (calculatedXSet.contains(x) && calculatedYSet.contains(y) && calculatedZSet.contains(z)) {
                            interpolationMask[index] = 1;
                        } else {
                            // Otherwise, mark it for interpolation
                            interpolationMask[index] = 0;
                        }
                    }
                }
            }
        }

        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            result.success = false;
            return result;
        }

        // Use ALL indices at current step size (including points from previous passes)
        result.interpolatedCount = interpolateDoseVolume(doseVolume,
                                                         interpolationMask,
                                                         xIndicesAll,
                                                         yIndicesAll,
                                                         zIndicesAll,
                                                         cancelRequested);

        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            result.success = false;
            return result;
        }
    }

    result.success = !(cancelRequested && cancelRequested->load(std::memory_order_relaxed));
    return result;
}

} // namespace

namespace CyberKnife {

class FastDoseLookup {
public:
    struct ConeData {
        double collimatorSize = 0.0;
        double dmFactor = 1.0;
        double dmReferenceDepth = 15.0;
        double offAxisLimit = 0.0;
        QVector<float> tmrDepths;
        QVector<float> tmrFieldSizes;
        QVector<float> tmrValues;
        int tmrFieldCount = 0;
        QVector<float> ocrDepths;
        QVector<float> ocrRadii;
        QVector<float> ocrRatios;
        int ocrDepthCount = 0;
        int ocrRadiusCount = 0;
        double matchedOcrCollimator = 0.0;
        QString ocrSourceFileName;
    };

    bool build(const BeamDataManager &manager)
    {
        m_cones.clear();
        m_manager = nullptr;
        if (!manager.isDataLoaded()) {
            return false;
        }
        m_manager = &manager;

        const auto &dmData = manager.dmData();
        const auto &tmrData = manager.tmrData();
        if (dmData.collimatorSizes.empty() || tmrData.depths.empty()) {
            return false;
        }

        m_cones.reserve(static_cast<int>(dmData.collimatorSizes.size()));
        for (int i = 0; i < static_cast<int>(dmData.collimatorSizes.size()); ++i) {
            ConeData cone;
            cone.collimatorSize = dmData.collimatorSizes[i];
            cone.dmFactor = (i < static_cast<int>(dmData.outputFactors.size()))
                                ? dmData.outputFactors[i]
                                : (dmData.outputFactors.empty() ? 1.0 : dmData.outputFactors.back());
            cone.dmReferenceDepth = dmData.depth;
            cone.offAxisLimit = cone.collimatorSize * kOffAxisLimitMultiplier;

            const int tmrDepthCount = static_cast<int>(tmrData.depths.size());
            const int tmrFieldCount = static_cast<int>(tmrData.fieldSizes.size());
            cone.tmrDepths.reserve(tmrDepthCount);
            cone.tmrFieldSizes.reserve(tmrFieldCount);
            cone.tmrFieldCount = tmrFieldCount;

            for (double depth : tmrData.depths) {
                cone.tmrDepths.append(static_cast<float>(depth));
            }
            for (double field : tmrData.fieldSizes) {
                cone.tmrFieldSizes.append(static_cast<float>(field));
            }

            if (tmrDepthCount > 0 && tmrFieldCount > 0) {
                cone.tmrValues.resize(tmrDepthCount * tmrFieldCount);
                for (int d = 0; d < tmrDepthCount; ++d) {
                    const auto &row = tmrData.tmrValues[static_cast<size_t>(d)];
                    for (int f = 0; f < tmrFieldCount; ++f) {
                        double value = 0.0;
                        if (f < static_cast<int>(row.size())) {
                            value = row[static_cast<size_t>(f)];
                        }
                        cone.tmrValues[d * tmrFieldCount + f] = static_cast<float>(value);
                    }
                }
            }

            double matchedCollimator = 0.0;
            const BeamDataParser::OCRData *ocr =
                manager.findClosestOcrTable(cone.collimatorSize, &matchedCollimator);
            if (ocr) {
                cone.ocrDepthCount = static_cast<int>(ocr->depths.size());
                cone.ocrRadiusCount = static_cast<int>(ocr->radii.size());
                cone.ocrDepths.reserve(cone.ocrDepthCount);
                cone.ocrRadii.reserve(cone.ocrRadiusCount);
                cone.matchedOcrCollimator = matchedCollimator;
                cone.ocrSourceFileName = ocr->sourceFileName;
                for (double depth : ocr->depths) {
                    cone.ocrDepths.append(static_cast<float>(depth));
                }
                for (double radius : ocr->radii) {
                    cone.ocrRadii.append(static_cast<float>(radius));
                }
                if (cone.ocrDepthCount > 0 && cone.ocrRadiusCount > 0) {
                    cone.ocrRatios.resize(cone.ocrDepthCount * cone.ocrRadiusCount);
                    for (int d = 0; d < cone.ocrDepthCount; ++d) {
                        for (int r = 0; r < cone.ocrRadiusCount; ++r) {
                            const double ratio = (d < static_cast<int>(ocr->ratios.size()) &&
                                                  r < static_cast<int>(ocr->ratios[d].size()))
                                                     ? ocr->ratios[d][r]
                                                     : 1.0;
                            cone.ocrRatios[d * cone.ocrRadiusCount + r] = static_cast<float>(ratio);
                        }
                    }
                }
            }

            m_cones.push_back(std::move(cone));
        }

        std::sort(m_cones.begin(), m_cones.end(), [](const ConeData &lhs, const ConeData &rhs) {
            return lhs.collimatorSize < rhs.collimatorSize;
        });

        return !m_cones.isEmpty();
    }

    const ConeData *matchCone(double collimatorSize) const
    {
        if (m_cones.isEmpty()) {
            return nullptr;
        }

        auto it = std::lower_bound(m_cones.begin(), m_cones.end(), collimatorSize,
                                   [](const ConeData &cone, double size) { return cone.collimatorSize < size; });
        if (it == m_cones.end()) {
            return &m_cones.back();
        }
        if (it == m_cones.begin()) {
            return &(*it);
        }

        const ConeData &upper = *it;
        const ConeData &lower = *(it - 1);
        if (qFabs(upper.collimatorSize - collimatorSize) <= qFabs(lower.collimatorSize - collimatorSize)) {
            return &upper;
        }
        return &lower;
    }

    bool isReady() const { return !m_cones.isEmpty(); }

    double dmFactor(const ConeData &cone) const { return cone.dmFactor; }

    double tmr(const ConeData &cone, double depth, double effectiveFieldSize) const
    {
        double correctedField = effectiveFieldSize;
        if (!std::isfinite(correctedField) || correctedField <= 0.0) {
            correctedField = cone.collimatorSize;
        }

        if (m_manager) {
            const double value = m_manager->getTMRValue(correctedField, depth);
            if (std::isfinite(value) && value > 0.0) {
                qCDebug(CyberKnifeDoseLog) << "Fast lookup TMR" << "collimator" << cone.collimatorSize
                                           << "effectiveField" << correctedField << "depth" << depth
                                           << "result" << value;
                return value;
            }
        }

        if (cone.tmrFieldCount > 0 && !cone.tmrDepths.isEmpty() && !cone.tmrFieldSizes.isEmpty()
            && !cone.tmrValues.isEmpty()) {
            const double fallback =
                sample2D(cone.tmrDepths, cone.tmrFieldSizes, cone.tmrValues, cone.tmrFieldCount, depth, correctedField);
            qCDebug(CyberKnifeDoseLog) << "Fast lookup TMR fallback" << "collimator" << cone.collimatorSize
                                       << "effectiveField" << correctedField << "depth" << depth << "result"
                                       << fallback;
            return fallback;
        }

        qCDebug(CyberKnifeDoseLog) << "Fast lookup TMR fallback (default)" << "collimator" << cone.collimatorSize
                                   << "effectiveField" << correctedField << "depth" << depth;
        return 1.0;
    }

    double ocr(const ConeData &cone, double depth, double radius) const
    {
        if (cone.ocrDepthCount <= 0 || cone.ocrRadiusCount <= 0 || cone.ocrRatios.isEmpty()) {
            return 1.0;
        }
        return sample2D(cone.ocrDepths, cone.ocrRadii, cone.ocrRatios, cone.ocrRadiusCount, depth, radius);
    }

    double offAxisLimit(const ConeData &cone) const { return cone.offAxisLimit; }

    bool lookupOcrInfo(double collimatorSize, double *matchedCollimator, QString *sourceFile) const
    {
        const ConeData *cone = matchCone(collimatorSize);
        if (!cone || cone->ocrDepthCount <= 0 || cone->ocrRadiusCount <= 0 || cone->ocrRatios.isEmpty()) {
            return false;
        }
        if (matchedCollimator) {
            *matchedCollimator = cone->matchedOcrCollimator > 0.0 ? cone->matchedOcrCollimator : cone->collimatorSize;
        }
        if (sourceFile) {
            *sourceFile = cone->ocrSourceFileName;
        }
        return true;
    }

private:
    QVector<ConeData> m_cones;
    const BeamDataManager *m_manager = nullptr;
};

Q_LOGGING_CATEGORY(CyberKnifeDoseLog, "cyberknife.dose")

bool CyberKnifeDoseCalculator::prepareDoseVolumeStorage(const DicomVolume &volume,
                                                        RTDoseVolume &doseVolume,
                                                        QStringList &errors)
{
    const int width = volume.width();
    const int height = volume.height();
    const int depth = volume.depth();

    if (width <= 0 || height <= 0 || depth <= 0) {
        errors << QStringLiteral("CTボリュームの寸法が不正です。");
        return false;
    }

    QVector3D origin = volume.voxelToPatient(0, 0, 0);
    QVector3D rowVector = volume.voxelToPatient(1, 0, 0) - origin;
    QVector3D colVector = volume.voxelToPatient(0, 1, 0) - origin;
    QVector3D sliceVector = volume.voxelToPatient(0, 0, 1) - origin;

    if (!rowVector.isNull()) {
        rowVector.normalize();
    }
    if (!colVector.isNull()) {
        colVector.normalize();
    }
    if (!sliceVector.isNull()) {
        sliceVector.normalize();
    }

    try {
        int sizes[3] = {depth, height, width};
        doseVolume.m_volume.create(3, sizes, CV_32F);
        doseVolume.m_volume.setTo(0.0f);
    } catch (const cv::Exception &ex) {
        qCWarning(CyberKnifeDoseLog) << "Failed to allocate dose volume:" << ex.what();
        errors << QStringLiteral("線量ボリュームの確保に失敗しました。");
        return false;
    } catch (const std::bad_alloc &) {
        qCWarning(CyberKnifeDoseLog) << "Failed to allocate dose volume due to insufficient memory.";
        errors << QStringLiteral("線量ボリュームの確保に必要なメモリが不足しています。");
        return false;
    }

    doseVolume.m_width = width;
    doseVolume.m_height = height;
    doseVolume.m_depth = depth;
    doseVolume.m_spacingX = volume.spacingX();
    doseVolume.m_spacingY = volume.spacingY();
    doseVolume.m_spacingZ = volume.spacingZ();
    doseVolume.m_originX = origin.x();
    doseVolume.m_originY = origin.y();
    doseVolume.m_originZ = origin.z();
    doseVolume.m_frameUID = volume.frameOfReferenceUID();
    doseVolume.m_rowDir[0] = rowVector.x();
    doseVolume.m_rowDir[1] = rowVector.y();
    doseVolume.m_rowDir[2] = rowVector.z();
    doseVolume.m_colDir[0] = colVector.x();
    doseVolume.m_colDir[1] = colVector.y();
    doseVolume.m_colDir[2] = colVector.z();
    doseVolume.m_sliceDir[0] = sliceVector.x();
    doseVolume.m_sliceDir[1] = sliceVector.y();
    doseVolume.m_sliceDir[2] = sliceVector.z();

    doseVolume.m_zOffsets.resize(static_cast<size_t>(depth));
    for (int z = 0; z < depth; ++z) {
        QVector3D p = volume.voxelToPatient(0, 0, z);
        QVector3D diff = p - origin;
        doseVolume.m_zOffsets[static_cast<size_t>(z)] = QVector3D::dotProduct(sliceVector, diff);
    }

    // デバッグ: 線量作成時の値を出力
    qDebug() << "=== DOSE CREATION DEBUG ===";
    qDebug() << QString("  CT spacing: %1 x %2 x %3")
        .arg(volume.spacingX()).arg(volume.spacingY()).arg(volume.spacingZ());
    qDebug() << QString("  CT ImagePositionPatient: (%1, %2, %3)")
        .arg(volume.originX()).arg(volume.originY()).arg(volume.originZ());
    qDebug() << QString("  CT voxelToPatient(0,0,0): (%1, %2, %3)")
        .arg(origin.x()).arg(origin.y()).arg(origin.z());
    double ctOriginDiff = QVector3D(volume.originX(), volume.originY(), volume.originZ()).distanceToPoint(origin);
    qDebug() << QString("  ** DIFFERENCE (CT origin - voxel(0,0,0)): %1 mm **").arg(ctOriginDiff);
    qDebug() << QString("  Dose spacing set to: %1 x %2 x %3")
        .arg(doseVolume.m_spacingX).arg(doseVolume.m_spacingY).arg(doseVolume.m_spacingZ);
    qDebug() << QString("  Dose origin set to: (%1, %2, %3)")
        .arg(doseVolume.m_originX).arg(doseVolume.m_originY).arg(doseVolume.m_originZ);
    if (!doseVolume.m_zOffsets.empty()) {
        qDebug() << QString("  Dose m_zOffsets[0]: %1").arg(doseVolume.m_zOffsets[0]);
        qDebug() << QString("  Dose m_zOffsets size: %1").arg(doseVolume.m_zOffsets.size());
        if (depth > 1) {
            qDebug() << QString("  Dose m_zOffsets[1]: %1").arg(doseVolume.m_zOffsets[1]);
        }
        if (depth > 2) {
            qDebug() << QString("  Dose m_zOffsets[2]: %1").arg(doseVolume.m_zOffsets[2]);
        }
    }

    doseVolume.m_maxDose = 0.0;
    doseVolume.m_patientShift = QVector3D(0.0f, 0.0f, 0.0f);
    doseVolume.m_ctToDose.setToIdentity();
    doseVolume.m_hasIOP = true;
    doseVolume.m_hasIPP = true;

    return true;
}

bool CyberKnifeDoseCalculator::applyToVolumeParallel(const DicomVolume &volume,
                                                     RTDoseVolume &doseVolume,
                                                     const std::function<double(int, int, int)> &calculator,
                                                     bool accumulate,
                                                     const QVector<int> &xIndices,
                                                     const QVector<int> &yIndices,
                                                     const QVector<int> &zIndices,
                                                     QVector<quint8> *computedMask,
                                                     std::atomic<bool> *cancelRequested)
{
    const int depth = volume.depth();
    const int height = volume.height();
    const int width = volume.width();

    if (depth <= 0 || height <= 0 || width <= 0) {
        return true;
    }

    QVector<int> sliceIndices = zIndices;
    if (sliceIndices.isEmpty()) {
        sliceIndices.resize(depth);
        std::iota(sliceIndices.begin(), sliceIndices.end(), 0);
    }

    QVector<int> defaultY;
    QVector<int> defaultX;
    const QVector<int> *yPtr = &yIndices;
    if (yIndices.isEmpty()) {
        defaultY.resize(height);
        std::iota(defaultY.begin(), defaultY.end(), 0);
        yPtr = &defaultY;
    }
    const QVector<int> *xPtr = &xIndices;
    if (xIndices.isEmpty()) {
        defaultX.resize(width);
        std::iota(defaultX.begin(), defaultX.end(), 0);
        xPtr = &defaultX;
    }
    const QVector<int> &yLoop = *yPtr;
    const QVector<int> &xLoop = *xPtr;

    quint8 *maskData = nullptr;
    int planeSize = width * height;
    if (computedMask && !computedMask->isEmpty()) {
        maskData = computedMask->data();
    }

    QtConcurrent::blockingMap(sliceIndices, [&](int z) {
        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
            return;
        }
        float *slice = doseVolume.m_volume.ptr<float>(z);
        quint8 *maskSlice = maskData ? maskData + z * planeSize : nullptr;
        for (int y : yLoop) {
            if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
                return;
            }
            float *row = slice + y * width;
            quint8 *maskRow = maskSlice ? maskSlice + y * width : nullptr;
            for (int x : xLoop) {
                if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
                    return;
                }
                // Skip if already computed (check mask before calculation)
                if (maskRow && maskRow[x]) {
                    continue;
                }
                const double value = calculator(x, y, z);
                if (accumulate) {
                    row[x] = static_cast<float>(row[x] + value);
                } else {
                    row[x] = static_cast<float>(value);
                }
                if (maskRow) {
                    maskRow[x] = 1;
                }
            }
        }
    });

    if (cancelRequested && cancelRequested->load(std::memory_order_relaxed)) {
        return false;
    }

    return true;
}

void FastDoseLookupDeleter::operator()(FastDoseLookup *ptr) const
{
    delete ptr;
}

CyberKnifeDoseCalculator::~CyberKnifeDoseCalculator() = default;

CyberKnifeDoseCalculator::DensityTableInfo CyberKnifeDoseCalculator::densityTableInfo() const
{
    DensityTableInfo info;
    const std::shared_ptr<const DensityTableState> state = densityTableSnapshot();
    if (!state) {
        return info;
    }
    info.offset = state->offset;
    info.source = state->source;
    info.entries.reserve(state->entries.size());
    for (const DensityTableEntry &entry : state->entries) {
        info.entries.append({entry.rawValue, entry.density});
    }
    return info;
}

bool CyberKnifeDoseCalculator::applyDensityTableOverride(const DensityTableInfo &info)
{
    DensityTableState state;
    state.offset = info.offset;
    state.source = info.source;
    state.entries.reserve(info.entries.size());

    for (const auto &entry : info.entries) {
        DensityTableEntry converted;
        converted.rawValue = entry.first;
        converted.density = entry.second;
        if (!std::isfinite(converted.rawValue) || !std::isfinite(converted.density)) {
            continue;
        }
        state.entries.append(converted);
    }

    normalizeDensityEntries(state.entries);
    if (state.entries.isEmpty()) {
        qCWarning(CyberKnifeDoseLog)
            << "Ignoring request to apply empty CT electron density table override from"
            << info.source;
        return false;
    }

    return setDensityTableOverride(state);
}

void CyberKnifeDoseCalculator::clearDensityTableOverride()
{
    restoreDensityTableToBase();
}

void CyberKnifeDoseCalculator::setDoseModel(DoseModel model)
{
    m_doseModel.store(model, std::memory_order_relaxed);
}

CyberKnifeDoseCalculator::DoseModel CyberKnifeDoseCalculator::doseModel() const
{
    return m_doseModel.load(std::memory_order_relaxed);
}

bool CyberKnifeDoseCalculator::initialize(const QString &beamDataPath)
{
    m_lastErrors.clear();
    const QString resolvedPath = BeamDataLocator::resolveBeamDataDirectory(beamDataPath);
    if (resolvedPath.isEmpty()) {
        qCWarning(CyberKnifeDoseLog)
            << "Failed to resolve CyberKnife beam data directory from"
            << (beamDataPath.isEmpty() ? QStringLiteral("auto-discovery") : beamDataPath);
        m_lastErrors << QStringLiteral("ビームデータディレクトリを特定できませんでした。");
        return false;
    }

    m_beamDataManager = std::make_unique<BeamDataManager>();
    if (!m_beamDataManager->loadBeamData(resolvedPath)) {
        const QStringList errors = m_beamDataManager->getValidationErrors();
        qCWarning(CyberKnifeDoseLog) << "Failed to load CyberKnife beam data from" << resolvedPath << errors;
        m_beamDataManager.reset();
        m_lastErrors = errors;
        if (m_lastErrors.isEmpty()) {
            m_lastErrors << QStringLiteral("ビームデータの読み込みに失敗しました。");
        }
        return false;
    }

    if (!m_beamDataManager->validateData()) {
        qCWarning(CyberKnifeDoseLog) << "Beam data validation failed" << m_beamDataManager->getValidationErrors();
        m_lastErrors = m_beamDataManager->getValidationErrors();
        m_beamDataManager.reset();
        return false;
    }

    m_fastLookup = FastDoseLookupPtr(new FastDoseLookup());
    if (!m_fastLookup->build(*m_beamDataManager)) {
        qCWarning(CyberKnifeDoseLog) << "Fast dose lookup cache build failed.";
        m_fastLookup.reset();
    }

    QSettings settings("ShioRIS3", "ShioRIS3");
    settings.setValue("cyberknife/beamDataPath", resolvedPath);

    qCInfo(CyberKnifeDoseLog) << "CyberKnife beam data initialized from" << resolvedPath;
    m_lastErrors.clear();

#ifdef ENABLE_GPU_DOSE_CALCULATION
    // Auto-initialize GPU backend with multi-GPU support
    if (initializeGPU(true)) {  // Enable multi-GPU mode
        int gpuCount = getGPUCount();
        if (gpuCount > 1) {
            qCInfo(CyberKnifeDoseLog) << "GPU backend auto-initialized with" << gpuCount << "GPUs:" << getGPUDeviceInfo();
        } else {
            qCInfo(CyberKnifeDoseLog) << "GPU backend auto-initialized:" << getGPUDeviceInfo();
        }
        m_gpuEnabled = true;  // Enable GPU by default
    } else {
        qCInfo(CyberKnifeDoseLog) << "GPU backend not available, will use CPU for dose calculation";
    }
#endif

    return true;
}

void CyberKnifeDoseCalculator::setResolutionOptions(const ResolutionOptions &options)
{
    m_resolutionOptions = options;
    m_resolutionOptions.stepXY = qMax(1, options.stepXY);
    m_resolutionOptions.stepZ = qMax(1, options.stepZ);

    m_resolutionOptions.dynamicThresholdStep2 =
        std::clamp(options.dynamicThresholdStep2, 0.0, 1.0);
    m_resolutionOptions.dynamicThresholdStep1 =
        std::clamp(options.dynamicThresholdStep1, 0.0, 1.0);

    if (m_resolutionOptions.dynamicThresholdStep1 < m_resolutionOptions.dynamicThresholdStep2) {
        m_resolutionOptions.dynamicThresholdStep1 = m_resolutionOptions.dynamicThresholdStep2;
    }
}

CyberKnifeDoseCalculator::FastLookupContext
CyberKnifeDoseCalculator::buildFastLookupContext(double collimatorSize) const
{
    FastLookupContext context;

    double managerMatched = 0.0;
    QString managerSource;
    const BeamDataParser::OCRData *managerOcr = nullptr;
    if (m_beamDataManager) {
        managerOcr = m_beamDataManager->findClosestOcrTable(collimatorSize, &managerMatched);
        if (managerOcr) {
            managerSource = managerOcr->sourceFileName;
        } else {
            managerSource.clear();
            qCWarning(CyberKnifeDoseLog)
                << "Beam data manager could not locate an OCR table for collimator"
                << collimatorSize << "mm.";
        }
        context.managerOcr = managerOcr;
        context.managerMatchedCollimator = managerMatched;
        context.managerSourceFile = managerSource;
    }

    if (!m_fastLookup || !m_fastLookup->isReady()) {
        return context;
    }

    const FastDoseLookup::ConeData *coneData = m_fastLookup->matchCone(collimatorSize);
    if (!coneData) {
        return context;
    }

    context.coneData = static_cast<const void *>(coneData);
    context.offAxisLimit = m_fastLookup->offAxisLimit(*coneData);
    context.hasOffAxisLimit = std::isfinite(context.offAxisLimit) && context.offAxisLimit > 0.0;
    context.matchedCollimator = coneData->matchedOcrCollimator > 0.0 ? coneData->matchedOcrCollimator
                                                                    : coneData->collimatorSize;
    context.sourceFile = coneData->ocrSourceFileName;

    if (!m_beamDataManager) {
        context.useFastLookup = true;
        return context;
    }

    if (!managerOcr) {
        context.useFastLookup = true;
        return context;
    }

    const bool sizeMatches = qFuzzyCompare(1.0 + context.matchedCollimator, 1.0 + managerMatched);
    const bool sourceMatches = context.sourceFile.isEmpty() || managerSource.isEmpty()
                               || context.sourceFile == managerSource;
    if (sizeMatches && sourceMatches) {
        context.useFastLookup = true;
    } else {
        qCWarning(CyberKnifeDoseLog)
            << "Fast OCR lookup mismatch for collimator" << collimatorSize << "mm:"
            << "cache matched" << context.matchedCollimator << "mm (" << context.sourceFile << ")"
            << "manager matched" << managerMatched << "mm (" << managerSource << ")";
    }

    return context;
}

double CyberKnifeDoseCalculator::calculatePointDoseWithContext(const CalculationPoint &point,
                                                              const GeometryCalculator::BeamGeometry &beam,
                                                              double depth,
                                                              double offAxis,
                                                              double referencedose,
                                                              const FastLookupContext &context) const
{
    if (!isReady()) {
        qCWarning(CyberKnifeDoseLog) << "Dose calculator is not ready.";
        return 0.0;
    }

    if (!std::isfinite(depth)) {
        return 0.0;
    }

    if (depth < 0.0) {
        qCDebug(CyberKnifeDoseLog) << "Negative depth encountered" << depth;
        return 0.0;
    }

    const double ssd = GeometryCalculator::calculateSSD(beam, point.position);
    const double axisLength = (beam.targetPosition - beam.sourcePosition).length();
    const double sad =
        (axisLength > 0.0 && std::isfinite(axisLength)) ? axisLength : (beam.SAD > 0.0 ? beam.SAD : ssd);

    // VBA互換: ビーム源から計算点までの距離（各点で可変）
    // VBAの式: f_siz = col * SAD / 800 のSADに相当
    const double sourceToPointDistance = (point.position - beam.sourcePosition).length();

    const QVector3D beamCoords = GeometryCalculator::patientToBeamCoordinate(point.position, beam);
    const double radiusSad = std::hypot(beamCoords.x(), beamCoords.y());
    double radius800 = radiusSad;
    if (ssd > 0.0 && std::isfinite(ssd)) {
        radius800 = radiusSad * (kReferenceSadMm / ssd);
    }

    double referenceOffAxisLimit = context.hasOffAxisLimit ? context.offAxisLimit
                                                           : beam.collimatorSize * kOffAxisLimitMultiplier;
    if (!std::isfinite(referenceOffAxisLimit) || referenceOffAxisLimit <= 0.0) {
        referenceOffAxisLimit = beam.collimatorSize * kOffAxisLimitMultiplier;
    }
    const double offAxisLimit = referenceOffAxisLimit;

    qCDebug(CyberKnifeDoseLog) << "Off-axis limit evaluation" << "SAD" << sad << "SSD" << ssd << "radius800"
                               << radius800 << "offAxisLimit" << offAxisLimit << "offAxis" << offAxis;

    if (radius800 > offAxisLimit) {
        return 0.0;
    }

    double outputFactor = 0.0;
    double ocrRatio = 1.0;
    double tmr = 0.0;
    // VBA互換: 実効照射野サイズの計算に sourceToPointDistance を使用
    if (!resolveDoseFactors(beam, depth, radius800, sourceToPointDistance, context, outputFactor, ocrRatio, tmr)) {
        return 0.0;
    }

    const DoseModel model = doseModel();
    if (model == DoseModel::RayTracing) {
        // VBA互換の計算式: Dose = MU × SF × TMR × OCR × (800/SAD)²
        // ここではMU値を除いた単位MUあたりの線量を計算
        // VBAのSAD = sourceToPointDistance（ビーム源から計算点までの距離）
        if (!(sourceToPointDistance > 0.0) || !std::isfinite(sourceToPointDistance)) {
            return 0.0;
        }
        const double distanceTerm = (kReferenceSadMm * kReferenceSadMm) / (sourceToPointDistance * sourceToPointDistance);
        // VBAとの違い: kRayTracingDoseScale = 0.01 はデータの単位差を吸収
        const double dose = outputFactor * tmr * ocrRatio * distanceTerm * kRayTracingDoseScale;
        qCDebug(CyberKnifeDoseLog)
            << "RayTracing dose" << "depth" << depth << "offAxis" << offAxis << "OF" << outputFactor << "OCR"
            << ocrRatio << "TMR" << tmr << "sourceToPointDist" << sourceToPointDistance
            << "distanceTerm" << distanceTerm << "scale" << kRayTracingDoseScale
            << "result" << dose;
        return dose;
    }

    // PrimaryOnly/PrimaryPlusScatterモード
    // VBA互換: 距離補正項を (800/sourceToPointDistance)² に修正
    const double densityScaling = qMax(0.0, point.density);
    const double distanceFactor = (sourceToPointDistance > 0.0)
                                      ? (kReferenceSadMm * kReferenceSadMm) / (sourceToPointDistance * sourceToPointDistance)
                                      : 1.0;
    // RayTracingと同じスケール係数を使用して一貫性を保つ
    const double dose = outputFactor * tmr * ocrRatio * densityScaling * distanceFactor * kRayTracingDoseScale;

    qCDebug(CyberKnifeDoseLog) << "Dose calculation details" << "depth" << depth << "offAxis" << offAxis
                               << "OF" << outputFactor << "OCR" << ocrRatio << "TMR" << tmr
                               << "density" << densityScaling << "sourceToPointDist" << sourceToPointDistance
                               << "distanceFactor" << distanceFactor << "scale" << kRayTracingDoseScale
                               << "result" << dose;

    return dose;
}

bool CyberKnifeDoseCalculator::resolveDoseFactors(const GeometryCalculator::BeamGeometry &beam,
                                                  double depth,
                                                  double radius800,
                                                  double sad,
                                                  const FastLookupContext &context,
                                                  double &outputFactor,
                                                  double &ocrRatio,
                                                  double &tmr) const
{
    const BeamDataParser::OCRData *managerOcr = context.managerOcr;
    ManagerBeamDataSample managerSample;
    const bool hasManager = (m_beamDataManager != nullptr);

    outputFactor = 0.0;
    ocrRatio = 1.0;
    tmr = 0.0;

    const double clippedTmrDepth = qBound(0.0, depth, kMaxTmrDepthMm);
    const double clippedOcrDepth = qBound(0.0, depth, kMaxOcrDepthMm);
    const double clippedRadius = qBound(0.0, radius800, kMaxOcrRadiusMm);

    if (!qFuzzyCompare(depth, clippedTmrDepth) || !qFuzzyCompare(depth, clippedOcrDepth)
        || !qFuzzyCompare(radius800, clippedRadius)) {
        qCDebug(CyberKnifeDoseLog) << "Dose factor inputs clipped"
                                   << "originalDepth" << depth << "tmrDepth" << clippedTmrDepth
                                   << "ocrDepth" << clippedOcrDepth << "radius800" << radius800
                                   << "ocrRadius" << clippedRadius;
    }

    const double effectiveFieldSize = computeEffectiveFieldSize(beam.collimatorSize, sad);
    qCDebug(CyberKnifeDoseLog) << "Effective field size"
                               << "collimator" << beam.collimatorSize << "SAD" << sad
                               << "corrected" << effectiveFieldSize;

    if (hasManager) {
        managerSample = sampleManagerBeamData(*m_beamDataManager,
                                              managerOcr,
                                              beam.collimatorSize,
                                              effectiveFieldSize,
                                              clippedTmrDepth,
                                              clippedOcrDepth,
                                              clippedRadius);
        if (managerSample.outputValid) {
            outputFactor = managerSample.outputFactor;
        }
        if (managerSample.ocrValid) {
            ocrRatio = managerSample.ocrRatio;
        }
        if (managerSample.tmrValid) {
            tmr = managerSample.tmr;
        }
    }

    const auto *coneData = static_cast<const FastDoseLookup::ConeData *>(context.coneData);
    if (context.useFastLookup) {
        if (coneData && m_fastLookup && m_fastLookup->isReady()) {
            const double fastOutput = m_fastLookup->dmFactor(*coneData);
            if (!managerSample.outputValid && std::isfinite(fastOutput) && fastOutput > 0.0) {
                outputFactor = fastOutput;
            }

            const double fastOcrRatio = m_fastLookup->ocr(*coneData, clippedOcrDepth, clippedRadius);
            if (!managerSample.ocrValid && std::isfinite(fastOcrRatio) && fastOcrRatio > 0.0) {
                ocrRatio = fastOcrRatio;
            }

            const double fastTmr = m_fastLookup->tmr(*coneData, clippedTmrDepth, effectiveFieldSize);
            if (!managerSample.tmrValid && std::isfinite(fastTmr) && fastTmr > 0.0) {
                tmr = fastTmr;
            }

            if ((!std::isfinite(outputFactor) || outputFactor <= 0.0)) {
                if (managerSample.outputValid) {
                    outputFactor = managerSample.outputFactor;
                } else if (std::isfinite(fastOutput) && fastOutput > 0.0) {
                    outputFactor = fastOutput;
                }
            }

            if ((!std::isfinite(tmr) || tmr <= 0.0)) {
                if (managerSample.tmrValid) {
                    tmr = managerSample.tmr;
                } else if (std::isfinite(fastTmr) && fastTmr > 0.0) {
                    tmr = fastTmr;
                }
            }
        } else if (!hasManager && coneData && m_fastLookup && m_fastLookup->isReady()) {
            const double fastOutput = m_fastLookup->dmFactor(*coneData);
            if (std::isfinite(fastOutput) && fastOutput > 0.0) {
                outputFactor = fastOutput;
            }
            const double fastOcrRatio = m_fastLookup->ocr(*coneData, clippedOcrDepth, clippedRadius);
            if (std::isfinite(fastOcrRatio) && fastOcrRatio > 0.0) {
                ocrRatio = fastOcrRatio;
            }
            const double fastTmr = m_fastLookup->tmr(*coneData, clippedTmrDepth, effectiveFieldSize);
            if (std::isfinite(fastTmr) && fastTmr > 0.0) {
                tmr = fastTmr;
            }
        }
    }

    if ((!std::isfinite(outputFactor) || outputFactor <= 0.0) || (!std::isfinite(tmr) || tmr <= 0.0)) {
        qCWarning(CyberKnifeDoseLog) << "Invalid beam data for dose calculation"
                                     << "OF:" << outputFactor << "TMR:" << tmr
                                     << "depth:" << depth << "collimator:" << beam.collimatorSize;
        return false;
    }

    if (!std::isfinite(ocrRatio) || ocrRatio <= 0.0) {
        ocrRatio = 1.0;
    }

    return true;
}

double CyberKnifeDoseCalculator::calculatePointDose(const CalculationPoint &point,
                                                    const GeometryCalculator::BeamGeometry &beam,
                                                    double referencedose)
{
    const double depth = GeometryCalculator::calculateDepth(beam, point.position);
    const double offAxis = GeometryCalculator::calculateOffAxisDistance(beam, point.position);
    return calculatePointDoseWithGeometry(point, beam, depth, offAxis, referencedose);
}

double CyberKnifeDoseCalculator::calculatePointDoseWithGeometry(const CalculationPoint &point,
                                                                const GeometryCalculator::BeamGeometry &beam,
                                                                double depth,
                                                                double offAxis,
                                                                double referencedose)
{
    const FastLookupContext context = buildFastLookupContext(beam.collimatorSize);
    return calculatePointDoseWithContext(point, beam, depth, offAxis, referencedose, context);
}

bool CyberKnifeDoseCalculator::computeRayTracingPointMetrics(const CalculationPoint &point,
                                                             const GeometryCalculator::BeamGeometry &beam,
                                                             const DicomVolume *ctVolume,
                                                             RayTracingPointMetrics *outMetrics) const
{
    if (!outMetrics) {
        return false;
    }

    RayTracingPointMetrics metrics;

    metrics.ssd = GeometryCalculator::calculateSSD(beam, point.position);
    const double axisDepth = GeometryCalculator::calculateDepth(beam, point.position);
    metrics.depth = axisDepth;
    if (!std::isfinite(axisDepth)) {
        *outMetrics = metrics;
        return false;
    }

    metrics.offAxis = GeometryCalculator::calculateOffAxisDistance(beam, point.position);
    const QVector3D beamCoords = GeometryCalculator::patientToBeamCoordinate(point.position, beam);
    metrics.offAxisX = beamCoords.x();
    metrics.offAxisY = beamCoords.y();
    metrics.radiusSad = std::sqrt(metrics.offAxisX * metrics.offAxisX + metrics.offAxisY * metrics.offAxisY);
    metrics.radius800 = metrics.radiusSad;
    if (std::isfinite(metrics.ssd) && metrics.ssd > 0.0) {
        metrics.radius800 = metrics.radiusSad * (kReferenceSadMm / metrics.ssd);
    }

    metrics.effectiveDepth = metrics.depth;
    if (ctVolume) {
        const cv::Mat &data = ctVolume->data();
        if (!data.empty()) {
            BeamDepthProfile profile = precomputeBeamDepths(*ctVolume, beam);
            if (profile.isValid()) {
                const double physicalDepth = axisDepth - profile.entryDistance;
                if (std::isfinite(physicalDepth)) {
                    metrics.depth = std::max(0.0, physicalDepth);
                }

                const double interpolated = profile.interpolate(axisDepth);
                if (std::isfinite(interpolated)) {
                    metrics.effectiveDepth = interpolated;
                }
            }
        }
    }

    metrics.tmr = std::numeric_limits<double>::quiet_NaN();
    metrics.unitDose = std::numeric_limits<double>::quiet_NaN();
    if (isReady()) {
        const double axisLength = (beam.targetPosition - beam.sourcePosition).length();
        const double sad = (axisLength > 0.0 && std::isfinite(axisLength))
                               ? axisLength
                               : (beam.SAD > 0.0 ? beam.SAD : metrics.ssd);
        if (sad > 0.0 && std::isfinite(sad)) {
            const FastLookupContext context = buildFastLookupContext(beam.collimatorSize);
            double outputFactor = 0.0;
            double ocrRatio = 1.0;
            double tmr = 0.0;
            if (resolveDoseFactors(beam,
                                   metrics.effectiveDepth,
                                   metrics.radius800,
                                   sad,
                                   context,
                                   outputFactor,
                                   ocrRatio,
                                   tmr)) {
                metrics.tmr = tmr;
                const double distanceTerm = (kReferenceSadMm * kReferenceSadMm) / (sad * sad);
                metrics.unitDose = outputFactor * tmr * ocrRatio * distanceTerm * kRayTracingDoseScale;
                qCDebug(CyberKnifeDoseLog) << "RayTracing metrics" << "SAD" << sad << "distanceTerm"
                                           << distanceTerm << "scale" << kRayTracingDoseScale << "OF"
                                           << outputFactor << "OCR" << ocrRatio << "TMR" << tmr;
            }
        }
    }

    *outMetrics = metrics;
    return true;
}

bool CyberKnifeDoseCalculator::calculateVolumeDose(const DicomVolume &volume,
                                                   const GeometryCalculator::BeamGeometry &beam,
                                                   RTDoseVolume &doseVolume)
{
    return calculateVolumeDoseWithProgress(volume, beam, doseVolume, {});
}

bool CyberKnifeDoseCalculator::calculateVolumeDoseWithProgress(const DicomVolume &volume,
                                                               const GeometryCalculator::BeamGeometry &beam,
                                                               RTDoseVolume &doseVolume,
                                                               std::function<void(int)> progressCallback)
{
    m_lastErrors.clear();

    if (!isReady()) {
        qCWarning(CyberKnifeDoseLog) << "Dose calculator is not ready.";
        m_lastErrors << QStringLiteral("線量計算エンジンが初期化されていません。");
        return false;
    }

    const cv::Mat &data = volume.data();
    if (data.empty()) {
        qCWarning(CyberKnifeDoseLog) << "Input CT volume data is empty.";
        m_lastErrors << QStringLiteral("CTボリュームデータが空です。");
        return false;
    }

    QStringList errors;
    if (!prepareDoseVolumeStorage(volume, doseVolume, errors)) {
        m_lastErrors = errors;
        return false;
    }

    const int width = volume.width();
    const int height = volume.height();
    const int depth = volume.depth();

    // Create a mutable copy of beam and initialize basis vectors to avoid redundant calculations
    GeometryCalculator::BeamGeometry beamCopy = beam;
    beamCopy.initializeBasis();

#ifdef ENABLE_GPU_DOSE_CALCULATION
    // Try GPU acceleration if enabled
    if (isGPUEnabled()) {
        qCInfo(CyberKnifeDoseLog) << "GPU dose calculation enabled, using" << m_gpuBackend->getBackendName();

        // Upload CT volume to GPU
        if (!m_gpuBackend->uploadCTVolume(volume.data())) {
            qCWarning(CyberKnifeDoseLog) << "Failed to upload CT volume to GPU:" << m_gpuBackend->getLastError();
            qCWarning(CyberKnifeDoseLog) << "Falling back to CPU calculation";
            m_gpuEnabled = false; // Disable GPU for this session
        } else {
            // Upload beam data to GPU
            GPUBeamData gpuBeamData = convertBeamDataToGPU(beamCopy.collimatorSize);
            if (!m_gpuBackend->uploadBeamData(gpuBeamData)) {
                qCWarning(CyberKnifeDoseLog) << "Failed to upload beam data to GPU:" << m_gpuBackend->getLastError();
                qCWarning(CyberKnifeDoseLog) << "Falling back to CPU calculation";
                m_gpuEnabled = false;
            } else {
                qCInfo(CyberKnifeDoseLog) << "GPU data upload complete, proceeding with GPU calculation";
            }
        }
    }
#endif

    // Dynamic Grid Size Mode: adaptive multi-pass refinement
    if (m_resolutionOptions.useDynamicGridMode) {
        qCInfo(CyberKnifeDoseLog) << "Using Dynamic Grid Size Mode with thresholds"
                                  << m_resolutionOptions.dynamicThresholdStep2 << "and"
                                  << m_resolutionOptions.dynamicThresholdStep1;

        auto depthProfile = std::make_shared<BeamDepthProfile>(precomputeBeamDepths(volume, beamCopy));

        // Pass 1: Coarse calculation with Step 4
        qCInfo(CyberKnifeDoseLog) << "Dynamic mode Pass 1: StepSize 4 (coarse calculation)";
        {
            int stepXY = 4;
            int stepZ = 4;

#ifdef ENABLE_GPU_DOSE_CALCULATION
            // Try GPU calculation first if enabled
            if (isGPUEnabled()) {
                qCInfo(CyberKnifeDoseLog) << "Pass 1: Using GPU acceleration";
                GPUComputeParams params = createGPUComputeParams(volume, beamCopy, stepXY, stepXY, stepZ, 1.0);
                if (depthProfile && depthProfile->isValid()) {
                    params.depthEntryDistance = depthProfile->entryDistance;
                    params.depthStepSize = depthProfile->stepSize;
                    params.depthCumulative.reserve(depthProfile->cumulativeDepths.size());
                    for (double value : depthProfile->cumulativeDepths) {
                        params.depthCumulative.push_back(static_cast<float>(value));
                    }
                }

                if (m_gpuBackend->calculateDose(params, doseVolume.m_volume)) {
                    qCInfo(CyberKnifeDoseLog) << "Pass 1: GPU calculation successful (1/64 voxels)";

                    // Use GPU interpolation to fill up to Step 2 grid (8/64 total)
                    QVector<int> xIndicesStep4 = generateSteppedIndices(width, stepXY);
                    QVector<int> yIndicesStep4 = generateSteppedIndices(height, stepXY);
                    QVector<int> zIndicesStep4 = generateSteppedIndices(depth, stepZ);

                    QVector<quint8> computedMask;
                    const qint64 maskSize64 = static_cast<qint64>(width) * height * depth;
                    if (maskSize64 <= std::numeric_limits<int>::max()) {
                        computedMask.resize(static_cast<int>(maskSize64));
                        std::fill(computedMask.begin(), computedMask.end(), quint8(0));

                        // Mark Step 4 grid voxels as already computed (1/64)
                        QtConcurrent::blockingMap(zIndicesStep4, [&](int z) {
                            const int zOffset = z * height * width;
                            for (int y : yIndicesStep4) {
                                const int yOffset = zOffset + y * width;
                                for (int x : xIndicesStep4) {
                                    computedMask[yOffset + x] = 1;
                                }
                            }
                        });

                        // GPU interpolation to Step 2 grid (fill up to 8/64)
                        qCInfo(CyberKnifeDoseLog) << "Pass 1: GPU interpolation to Step 2 grid";
                        std::vector<uint8_t> maskStdVec(computedMask.begin(), computedMask.end());

                        // Interpolate to Step 2 grid (2 voxel spacing), not Step 4
                        if (m_gpuBackend->interpolateVolume(doseVolume.m_volume, maskStdVec, 2)) {
                            qCInfo(CyberKnifeDoseLog) << "Pass 1: GPU interpolation successful";

                            // Mark Step 2 grid points as computed (8/64 total)
                            QVector<int> xIndicesStep2 = generateSteppedIndices(width, 2);
                            QVector<int> yIndicesStep2 = generateSteppedIndices(height, 2);
                            QVector<int> zIndicesStep2 = generateSteppedIndices(depth, 2);

                            QtConcurrent::blockingMap(zIndicesStep2, [&](int z) {
                                const int zOffset = z * height * width;
                                for (int y : yIndicesStep2) {
                                    const int yOffset = zOffset + y * width;
                                    for (int x : xIndicesStep2) {
                                        computedMask[yOffset + x] = 1;
                                    }
                                }
                            });

                            qCInfo(CyberKnifeDoseLog) << "Pass 1 complete: 8/64 voxels computed/interpolated on GPU";
                        } else {
                            qCWarning(CyberKnifeDoseLog) << "Pass 1: GPU interpolation failed:" << m_gpuBackend->getLastError();
                            qCWarning(CyberKnifeDoseLog) << "Falling back to CPU interpolation";
                            interpolateDoseVolume(doseVolume,
                                                  computedMask,
                                                  xIndicesStep4,
                                                  yIndicesStep4,
                                                  zIndicesStep4,
                                                  m_resolutionOptions.cancelRequested);
                        }
                    }

                    // Calculate statistics after Pass 1
                    double minDose = 0.0;
                    double maxDose = 0.0;
                    cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
                    qCInfo(CyberKnifeDoseLog) << "Pass 1 complete. Max dose:" << maxDose;

                    // Update progress
                    if (progressCallback) {
                        progressCallback(50);
                    }

                    // Pass 2: Recalculate Step 2 grid (8/64) voxels above threshold, then interpolate to full resolution
                    const float thresholdStep2 = static_cast<float>(maxDose * m_resolutionOptions.dynamicThresholdStep2);
                    qCInfo(CyberKnifeDoseLog) << "Pass 2: Recalculate Step 2 grid voxels >= " << thresholdStep2
                                              << " (" << (m_resolutionOptions.dynamicThresholdStep2 * 100) << "% of max)";

                    // GPU recalculation with threshold
                    GPUComputeParams paramsStep2 = createGPUComputeParams(volume, beamCopy, 2, 2, 2, 1.0);
                    if (depthProfile && depthProfile->isValid()) {
                        paramsStep2.depthEntryDistance = depthProfile->entryDistance;
                        paramsStep2.depthStepSize = depthProfile->stepSize;
                        paramsStep2.depthCumulative.reserve(depthProfile->cumulativeDepths.size());
                        for (double value : depthProfile->cumulativeDepths) {
                            paramsStep2.depthCumulative.push_back(static_cast<float>(value));
                        }
                    }

                    if (m_gpuBackend->recalculateDoseWithThreshold(paramsStep2, doseVolume.m_volume, thresholdStep2)) {
                        qCInfo(CyberKnifeDoseLog) << "Pass 2: GPU threshold recalculation successful";
                    } else {
                        qCWarning(CyberKnifeDoseLog) << "Pass 2: GPU threshold recalculation failed:" << m_gpuBackend->getLastError();
                        // Continue even if recalculation failed
                    }

                    // GPU interpolation to full resolution (64/64)
                    qCInfo(CyberKnifeDoseLog) << "Pass 2: GPU interpolation to full resolution (64/64)";

                    // Use the existing mask from Pass 1 (which already has Step 2 grid marked)
                    // All Step 2 grid points (8/64) should be marked as computed
                    std::vector<uint8_t> maskStdVec(computedMask.begin(), computedMask.end());

                    // Interpolate from Step 2 grid to full resolution (step=1 means fill all voxels)
                    if (m_gpuBackend->interpolateVolume(doseVolume.m_volume, maskStdVec, 1)) {
                        qCInfo(CyberKnifeDoseLog) << "Pass 2: GPU interpolation to full resolution successful";
                    } else {
                        qCWarning(CyberKnifeDoseLog) << "Pass 2: GPU interpolation failed:" << m_gpuBackend->getLastError();
                        qCWarning(CyberKnifeDoseLog) << "Falling back to CPU interpolation";

                        QVector<int> xIndicesStep2 = generateSteppedIndices(width, 2);
                        QVector<int> yIndicesStep2 = generateSteppedIndices(height, 2);
                        QVector<int> zIndicesStep2 = generateSteppedIndices(depth, 2);

                        interpolateDoseVolume(doseVolume,
                                              computedMask,
                                              xIndicesStep2,
                                              yIndicesStep2,
                                              zIndicesStep2,
                                              m_resolutionOptions.cancelRequested);
                    }

                    // Calculate final statistics
                    cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
                    doseVolume.m_maxDose = maxDose;
                    qCInfo(CyberKnifeDoseLog) << "Pass 2 complete. Final max dose:" << maxDose;

                    // Update progress
                    if (progressCallback) {
                        progressCallback(75);
                    }

                    // Pass 3: Recalculate non-Step-2-grid voxels above threshold
                    const float thresholdStep1 = static_cast<float>(maxDose * m_resolutionOptions.dynamicThresholdStep1);
                    qCInfo(CyberKnifeDoseLog) << "Pass 3: Recalculate non-Step-2-grid voxels >= " << thresholdStep1
                                              << " (" << (m_resolutionOptions.dynamicThresholdStep1 * 100) << "% of max)";

                    // GPU recalculation with threshold and skipStep=2 (skip Step 2 grid)
                    GPUComputeParams paramsStep1 = createGPUComputeParams(volume, beamCopy, 1, 1, 1, 1.0);
                    if (depthProfile && depthProfile->isValid()) {
                        paramsStep1.depthEntryDistance = depthProfile->entryDistance;
                        paramsStep1.depthStepSize = depthProfile->stepSize;
                        paramsStep1.depthCumulative.reserve(depthProfile->cumulativeDepths.size());
                        for (double value : depthProfile->cumulativeDepths) {
                            paramsStep1.depthCumulative.push_back(static_cast<float>(value));
                        }
                    }

                    if (m_gpuBackend->recalculateDoseWithThreshold(paramsStep1, doseVolume.m_volume, thresholdStep1, 2)) {
                        qCInfo(CyberKnifeDoseLog) << "Pass 3: GPU threshold recalculation successful";
                    } else {
                        qCWarning(CyberKnifeDoseLog) << "Pass 3: GPU threshold recalculation failed:" << m_gpuBackend->getLastError();
                        // Continue even if recalculation failed
                    }

                    // Calculate final statistics
                    cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
                    doseVolume.m_maxDose = maxDose;
                    qCInfo(CyberKnifeDoseLog) << "Pass 3 complete. Final max dose:" << maxDose;

                    // Update progress
                    if (progressCallback) {
                        progressCallback(100);
                    }

                    return true;
                } else {
                    qCWarning(CyberKnifeDoseLog) << "Pass 1: GPU calculation failed:" << m_gpuBackend->getLastError();
                    qCWarning(CyberKnifeDoseLog) << "Falling back to CPU for Pass 1";
                    m_gpuEnabled = false; // Disable for remaining passes
                }
            }

            // CPU fallback or if GPU disabled
            if (!isGPUEnabled())
#endif
            {
                QVector<int> xIndices = generateSteppedIndices(width, stepXY);
                QVector<int> yIndices = generateSteppedIndices(height, stepXY);
                QVector<int> zIndices = generateSteppedIndices(depth, stepZ);

                QVector<quint8> computedMask;
                const qint64 maskSize64 = static_cast<qint64>(width) * height * depth;
                if (maskSize64 <= std::numeric_limits<int>::max()) {
                    computedMask.resize(static_cast<int>(maskSize64));
                    std::fill(computedMask.begin(), computedMask.end(), quint8(0));
                }

                const qint64 coarseCount64 = static_cast<qint64>(xIndices.size()) * yIndices.size() * zIndices.size();
                const int totalSamples = coarseCount64 > std::numeric_limits<int>::max()
                                             ? std::numeric_limits<int>::max()
                                             : static_cast<int>(coarseCount64);

                std::atomic<int> progressCounter{0};
                std::function<void(int)> callbackCopy = progressCallback;

                auto calculator = makeVoxelCalculator(volume,
                                                      this,
                                                      beamCopy,
                                                      depthProfile,
                                                      1.0,
                                                      m_resolutionOptions.cancelRequested,
                                                      callbackCopy ? &progressCounter : nullptr,
                                                      totalSamples * 3,  // Multiply by 3 for 3 passes
                                                      callbackCopy);

                if (!applyToVolumeParallel(volume,
                                           doseVolume,
                                           calculator,
                                           false,
                                           xIndices,
                                           yIndices,
                                           zIndices,
                                           computedMask.isEmpty() ? nullptr : &computedMask,
                                           m_resolutionOptions.cancelRequested)) {
                    return false;
                }

                if (m_resolutionOptions.cancelRequested &&
                    m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                    return false;
                }

                if (!computedMask.isEmpty()) {
                    interpolateDoseVolume(doseVolume,
                                          computedMask,
                                          xIndices,
                                          yIndices,
                                          zIndices,
                                          m_resolutionOptions.cancelRequested);
                }
            }
        }

        // Check for cancellation after Pass 1
        if (m_resolutionOptions.cancelRequested && m_resolutionOptions.cancelRequested->load()) {
            qCInfo(CyberKnifeDoseLog) << "Calculation cancelled after Pass 1";
            return false;
        }

        // Find maximum dose after Pass 1
        double minDose = 0.0;
        double maxDose = 0.0;
        cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
        qCInfo(CyberKnifeDoseLog) << "Pass 1 complete. Max dose:" << maxDose;

        if (maxDose <= 0.0) {
            qCWarning(CyberKnifeDoseLog) << "Max dose is zero after Pass 1, skipping refinement";
            doseVolume.m_maxDose = maxDose;
            if (progressCallback &&
                (!m_resolutionOptions.cancelRequested ||
                 !m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed))) {
                progressCallback(100);
            }
            return true;
        }

        // Pass 2: Refine with Step 2, then recalculate interpolated high-dose voxels
        const float thresholdStep2 = static_cast<float>(maxDose * m_resolutionOptions.dynamicThresholdStep2);
        qCInfo(CyberKnifeDoseLog) << "Dynamic mode Pass 2: StepSize 2, recalculate interpolated voxels >= "
                                  << thresholdStep2 << " (" << (m_resolutionOptions.dynamicThresholdStep2 * 100)
                                  << "% of max)";
        {
            int stepXY = 2;
            int stepZ = 2;

            // Generate all indices at step 2
            QVector<int> xIndicesAll = generateSteppedIndices(width, stepXY);
            QVector<int> yIndicesAll = generateSteppedIndices(height, stepXY);
            QVector<int> zIndicesAll = generateSteppedIndices(depth, stepZ);

            // Create mask: mark Pass 1 grid points (step=4) as already calculated
            // Use parallel mask creation for better performance
            QVector<quint8> computedMask;
            const qint64 maskSize64 = static_cast<qint64>(width) * height * depth;
            if (maskSize64 <= std::numeric_limits<int>::max()) {
                computedMask.resize(static_cast<int>(maskSize64));
                std::fill(computedMask.begin(), computedMask.end(), quint8(0));

                QVector<int> zIndicesPass1 = generateSteppedIndices(depth, 4);

                // Parallelize mask creation over Z slices
                QtConcurrent::blockingMap(zIndicesPass1, [&](int z) {
                    if (m_resolutionOptions.cancelRequested &&
                        m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                        return;
                    }
                    const int zOffset = z * height * width;
                    for (int y = 0; y < height; y += 4) {
                        for (int x = 0; x < width; x += 4) {
                            computedMask[zOffset + y * width + x] = 1;
                        }
                    }
                });
            }

            // Calculate all grid points at step 2 (Pass 1 points are skipped via mask)
            auto calculator = makeVoxelCalculator(volume,
                                                  this,
                                                  beamCopy,
                                                  depthProfile,
                                                  1.0,
                                                  m_resolutionOptions.cancelRequested,
                                                  nullptr,
                                                  0,
                                                  {});

            if (!applyToVolumeParallel(volume,
                                       doseVolume,
                                       calculator,
                                       false,  // Do NOT accumulate - replace interpolated values
                                       xIndicesAll,
                                       yIndicesAll,
                                       zIndicesAll,
                                       computedMask.isEmpty() ? nullptr : &computedMask,
                                       m_resolutionOptions.cancelRequested)) {
                return false;
            }

            if (m_resolutionOptions.cancelRequested &&
                m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                return false;
            }

            // Interpolate all non-grid points
            int interpolatedCount = 0;
            if (!computedMask.isEmpty()) {
                interpolatedCount = interpolateDoseVolume(doseVolume,
                                                         computedMask,
                                                         xIndicesAll,
                                                         yIndicesAll,
                                                         zIndicesAll,
                                                         m_resolutionOptions.cancelRequested);
            }

            if (m_resolutionOptions.cancelRequested &&
                m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                return false;
            }

            // Recalculate interpolated voxels with dose >= threshold
            // Use mathematical check (modulo) instead of QSet for much better performance
            std::atomic<int> recalcCount{0};
            QVector<int> allZIndices;
            allZIndices.reserve(depth);
            for (int z = 0; z < depth; ++z) {
                allZIndices.append(z);
            }

            QtConcurrent::blockingMap(allZIndices, [&](int z) {
                if (m_resolutionOptions.cancelRequested &&
                    m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                    return;
                }

                float *slice = doseVolume.data().ptr<float>(z);
                const bool zOnGrid = (z % stepZ == 0);

                for (int y = 0; y < height; ++y) {
                    if (m_resolutionOptions.cancelRequested &&
                        m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                        return;
                    }

                    float *row = slice + y * width;
                    const bool yOnGrid = (y % stepXY == 0);

                    for (int x = 0; x < width; ++x) {
                        // Check if this voxel is NOT on the grid (i.e., was interpolated)
                        // Use modulo check instead of QSet::contains() - much faster
                        const bool isOnGrid = (x % stepXY == 0) && yOnGrid && zOnGrid;

                        if (!isOnGrid && row[x] >= thresholdStep2) {
                            // Recalculate this interpolated high-dose voxel
                            const double value = calculator(x, y, z);
                            row[x] = static_cast<float>(value);
                            recalcCount.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            });

            qCInfo(CyberKnifeDoseLog) << "Pass 2 complete. Interpolated:" << interpolatedCount
                                      << "Recalculated interpolated high-dose voxels:" << recalcCount.load();
        }

        // Check for cancellation after Pass 2
        if (m_resolutionOptions.cancelRequested && m_resolutionOptions.cancelRequested->load()) {
            qCInfo(CyberKnifeDoseLog) << "Calculation cancelled after Pass 2";
            return false;
        }

        // Pass 3: Interpolate all remaining voxels, then recalculate interpolated high-dose voxels
        const float thresholdStep1 = static_cast<float>(maxDose * m_resolutionOptions.dynamicThresholdStep1);
        qCInfo(CyberKnifeDoseLog) << "Dynamic mode Pass 3: Interpolate all voxels, recalculate interpolated voxels >= "
                                  << thresholdStep1 << " (" << (m_resolutionOptions.dynamicThresholdStep1 * 100)
                                  << "% of max)";
        {
            // Create mask: mark Pass 1 and Pass 2 grid points as already calculated
            // Use parallel mask creation for better performance
            QVector<quint8> computedMask;
            const qint64 maskSize64 = static_cast<qint64>(width) * height * depth;
            if (maskSize64 <= std::numeric_limits<int>::max()) {
                computedMask.resize(static_cast<int>(maskSize64));
                std::fill(computedMask.begin(), computedMask.end(), quint8(0));

                // Mark Pass 2 grid points (step=2) as calculated
                // These include Pass 1 grid points (step=4) as well
                QVector<int> zIndicesPass2 = generateSteppedIndices(depth, 2);

                // Parallelize mask creation over Z slices
                QtConcurrent::blockingMap(zIndicesPass2, [&](int z) {
                    if (m_resolutionOptions.cancelRequested &&
                        m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                        return;
                    }
                    const int zOffset = z * height * width;
                    for (int y = 0; y < height; y += 2) {
                        for (int x = 0; x < width; x += 2) {
                            computedMask[zOffset + y * width + x] = 1;
                        }
                    }
                });
            }

            if (m_resolutionOptions.cancelRequested &&
                m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                return false;
            }

            // Interpolate all non-grid points (step 2 grid is used for interpolation)
            int interpolatedCount = 0;
            if (!computedMask.isEmpty()) {
                QVector<int> xIndicesStep2 = generateSteppedIndices(width, 2);
                QVector<int> yIndicesStep2 = generateSteppedIndices(height, 2);
                QVector<int> zIndicesStep2 = generateSteppedIndices(depth, 2);

                interpolatedCount = interpolateDoseVolume(doseVolume,
                                                         computedMask,
                                                         xIndicesStep2,
                                                         yIndicesStep2,
                                                         zIndicesStep2,
                                                         m_resolutionOptions.cancelRequested);
            }

            if (m_resolutionOptions.cancelRequested &&
                m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                return false;
            }

            qCInfo(CyberKnifeDoseLog) << "Pass 3 interpolation complete. Interpolated:" << interpolatedCount << "voxels";

            // Create calculator for recalculation
            auto calculator = makeVoxelCalculator(volume,
                                                  this,
                                                  beamCopy,
                                                  depthProfile,
                                                  1.0,
                                                  m_resolutionOptions.cancelRequested,
                                                  nullptr,
                                                  0,
                                                  {});

            // Recalculate interpolated voxels with dose >= threshold
            // Only recalculate voxels that were interpolated (not on step 2 grid)
            // Use mathematical check (modulo) instead of QSet for much better performance
            const int stepXY = 2;
            const int stepZ = 2;
            std::atomic<int> recalcCount{0};
            QVector<int> allZIndices;
            allZIndices.reserve(depth);
            for (int z = 0; z < depth; ++z) {
                allZIndices.append(z);
            }

            QtConcurrent::blockingMap(allZIndices, [&](int z) {
                if (m_resolutionOptions.cancelRequested &&
                    m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                    return;
                }

                float *slice = doseVolume.data().ptr<float>(z);
                const bool zOnGrid = (z % stepZ == 0);

                for (int y = 0; y < height; ++y) {
                    if (m_resolutionOptions.cancelRequested &&
                        m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                        return;
                    }

                    float *row = slice + y * width;
                    const bool yOnGrid = (y % stepXY == 0);

                    for (int x = 0; x < width; ++x) {
                        // Check if this voxel is NOT on the step 2 grid (i.e., was interpolated)
                        // Use modulo check instead of QSet::contains() - much faster
                        const bool isOnGrid = (x % stepXY == 0) && yOnGrid && zOnGrid;

                        if (!isOnGrid && row[x] >= thresholdStep1) {
                            // Recalculate this interpolated high-dose voxel
                            const double value = calculator(x, y, z);
                            row[x] = static_cast<float>(value);
                            recalcCount.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            });

            qCInfo(CyberKnifeDoseLog) << "Pass 3 complete. Recalculated interpolated high-dose voxels:" << recalcCount.load();
        }

        // Final max dose calculation
        if (m_resolutionOptions.cancelRequested &&
            m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
            return false;
        }

        cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
        doseVolume.m_maxDose = maxDose;
        qCInfo(CyberKnifeDoseLog) << "Dynamic Grid Mode complete. Final max dose:" << maxDose;

        if (progressCallback &&
            (!m_resolutionOptions.cancelRequested ||
             !m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed))) {
            progressCallback(100);
        }

        return true;
    }

    // Precompute depth profile for this beam (used by both GPU and CPU paths)
    auto depthProfile = std::make_shared<BeamDepthProfile>(precomputeBeamDepths(volume, beamCopy));

    // Standard mode: use configured step size
    int effectiveStepXY = qMax(1, m_resolutionOptions.stepXY);
    int effectiveStepZ = qMax(1, m_resolutionOptions.stepZ);

#ifdef ENABLE_GPU_DOSE_CALCULATION
    // Try GPU calculation first if enabled
    if (isGPUEnabled()) {
        qCInfo(CyberKnifeDoseLog) << "Standard mode: Using GPU acceleration (step:"
                                  << effectiveStepXY << "," << effectiveStepZ << ")";
        GPUComputeParams params = createGPUComputeParams(volume, beamCopy,
                                                         effectiveStepXY, effectiveStepXY, effectiveStepZ, 100.0);
        if (depthProfile && depthProfile->isValid()) {
            params.depthEntryDistance = depthProfile->entryDistance;
            params.depthStepSize = depthProfile->stepSize;
            params.depthCumulative.reserve(depthProfile->cumulativeDepths.size());
            for (double value : depthProfile->cumulativeDepths) {
                params.depthCumulative.push_back(static_cast<float>(value));
            }
        }

        if (m_gpuBackend->calculateDose(params, doseVolume.m_volume)) {
            qCInfo(CyberKnifeDoseLog) << "Standard mode: GPU calculation successful";

            // Update progress
            if (progressCallback) {
                progressCallback(90); // 90% after GPU calculation
            }

            // Interpolation (CPU fallback - GPU interpolation not yet implemented)
            bool needsInterpolation = (effectiveStepXY > 1 || effectiveStepZ > 1);
            if (needsInterpolation) {
                qCInfo(CyberKnifeDoseLog) << "Performing CPU-based interpolation (GPU interpolation not yet implemented)";

                QVector<int> xIndices = generateSteppedIndices(width, effectiveStepXY);
                QVector<int> yIndices = generateSteppedIndices(height, effectiveStepXY);
                QVector<int> zIndices = generateSteppedIndices(depth, effectiveStepZ);

                QVector<quint8> computedMask;
                const qint64 maskSize64 = static_cast<qint64>(width) * height * depth;
                if (maskSize64 <= std::numeric_limits<int>::max()) {
                    computedMask.resize(static_cast<int>(maskSize64));
                    std::fill(computedMask.begin(), computedMask.end(), quint8(0));

                    // Mark computed grid points
                    for (int z : zIndices) {
                        for (int y : yIndices) {
                            for (int x : xIndices) {
                                computedMask[z * height * width + y * width + x] = 1;
                            }
                        }
                    }

                    interpolateDoseVolume(doseVolume,
                                          computedMask,
                                          xIndices,
                                          yIndices,
                                          zIndices,
                                          m_resolutionOptions.cancelRequested);
                }
            }

            // Compute final statistics
            double minDose = 0.0;
            double maxDose = 0.0;
            cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
            doseVolume.m_maxDose = maxDose;

            if (progressCallback) {
                progressCallback(100);
            }

            return true;
        } else {
            qCWarning(CyberKnifeDoseLog) << "Standard mode: GPU calculation failed:" << m_gpuBackend->getLastError();
            qCWarning(CyberKnifeDoseLog) << "Falling back to CPU calculation";
            m_gpuEnabled = false;
        }
    }
#endif

    // CPU fallback or if GPU disabled
    QVector<int> xIndices = generateSteppedIndices(width, effectiveStepXY);
    QVector<int> yIndices = generateSteppedIndices(height, effectiveStepXY);
    QVector<int> zIndices = generateSteppedIndices(depth, effectiveStepZ);

    bool needsInterpolation = (effectiveStepXY > 1 || effectiveStepZ > 1);
    QVector<quint8> computedMask;
    QVector<quint8> *maskPtr = nullptr;
    if (needsInterpolation) {
        const qint64 maskSize64 = static_cast<qint64>(width) * height * depth;
        if (maskSize64 > std::numeric_limits<int>::max()) {
            qCWarning(CyberKnifeDoseLog)
                << "Dose volume too large for coarse mask allocation; falling back to full resolution.";
            effectiveStepXY = 1;
            effectiveStepZ = 1;
            xIndices = generateSteppedIndices(width, effectiveStepXY);
            yIndices = generateSteppedIndices(height, effectiveStepXY);
            zIndices = generateSteppedIndices(depth, effectiveStepZ);
            needsInterpolation = false;
        } else {
            computedMask.resize(static_cast<int>(maskSize64));
            std::fill(computedMask.begin(), computedMask.end(), quint8(0));
            maskPtr = &computedMask;
        }
    }

    const qint64 coarseCount64 = static_cast<qint64>(xIndices.size()) * yIndices.size() * zIndices.size();
    const int totalSamples = coarseCount64 > std::numeric_limits<int>::max()
                                 ? std::numeric_limits<int>::max()
                                 : static_cast<int>(coarseCount64);

    std::atomic<int> progressCounter{0};
    std::function<void(int)> callbackCopy = progressCallback;

    auto calculator = makeVoxelCalculator(volume,
                                          this,
                                          beamCopy,
                                          depthProfile,
                                          1.0,
                                          m_resolutionOptions.cancelRequested,
                                          callbackCopy ? &progressCounter : nullptr,
                                          totalSamples,
                                          callbackCopy);

    if (!applyToVolumeParallel(volume,
                               doseVolume,
                               calculator,
                               false,
                               xIndices,
                               yIndices,
                               zIndices,
                               maskPtr,
                               m_resolutionOptions.cancelRequested)) {
        return false;
    }

    if (m_resolutionOptions.cancelRequested &&
        m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
        return false;
    }

    if (needsInterpolation && maskPtr) {
        interpolateDoseVolume(doseVolume,
                              *maskPtr,
                              xIndices,
                              yIndices,
                              zIndices,
                              m_resolutionOptions.cancelRequested);
    }

    if (m_resolutionOptions.cancelRequested &&
        m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
        return false;
    }

    double minDose = 0.0;
    double maxDose = 0.0;
    cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
    doseVolume.m_maxDose = maxDose;

    if (callbackCopy &&
        (!m_resolutionOptions.cancelRequested ||
         !m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed))) {
        callbackCopy(100);
    }

    return true;
}

bool CyberKnifeDoseCalculator::calculateMultiBeamVolumeDose(const DicomVolume &volume,
                                                            const QVector<GeometryCalculator::BeamGeometry> &beams,
                                                            const QVector<double> &beamWeights,
                                                            RTDoseVolume &doseVolume)
{
    return calculateMultiBeamVolumeDoseWithProgress(volume,
                                                    beams,
                                                    beamWeights,
                                                    {},
                                                    doseVolume);
}

bool CyberKnifeDoseCalculator::calculateMultiBeamVolumeDoseWithProgress(
    const DicomVolume &volume,
    const QVector<GeometryCalculator::BeamGeometry> &beams,
    const QVector<double> &beamWeights,
    std::function<void(int, int)> progressCallback,
    RTDoseVolume &doseVolume)
{
    m_lastErrors.clear();

    if (!isReady()) {
        qCWarning(CyberKnifeDoseLog) << "Dose calculator is not ready.";
        m_lastErrors << QStringLiteral("線量計算エンジンが初期化されていません。");
        return false;
    }

    if (beams.isEmpty()) {
        qCWarning(CyberKnifeDoseLog) << "No beams provided for multi-beam calculation.";
        m_lastErrors << QStringLiteral("ビームが指定されていません。");
        return false;
    }

    if (!beamWeights.isEmpty() && beamWeights.size() != beams.size()) {
        qCWarning(CyberKnifeDoseLog) << "Beam weight count does not match beam count.";
        m_lastErrors << QStringLiteral("ビーム数と重み数が一致しません。");
        return false;
    }

    QVector<double> weights = beamWeights;
    if (weights.isEmpty()) {
        weights.resize(beams.size());
        std::fill(weights.begin(), weights.end(), 1.0);
    }

    QStringList errors;
    if (!prepareDoseVolumeStorage(volume, doseVolume, errors)) {
        m_lastErrors = errors;
        return false;
    }

    doseVolume.m_volume.setTo(0.0f);

    for (int beamIndex = 0; beamIndex < beams.size(); ++beamIndex) {
        const auto &beam = beams[beamIndex];
        double matchedSize = 0.0;
        QString ocrSource;
        const bool fromCache = m_fastLookup && m_fastLookup->isReady() &&
                               m_fastLookup->lookupOcrInfo(beam.collimatorSize, nullptr, nullptr);
        if (resolveOcrTableInfo(beam.collimatorSize, &matchedSize, &ocrSource)) {
            qCInfo(CyberKnifeDoseLog) << "Beam" << (beamIndex + 1) << "collimator" << beam.collimatorSize
                                     << "uses OCR table"
                                     << (ocrSource.isEmpty() ? QStringLiteral("<unknown>") : ocrSource)
                                     << "matched to" << matchedSize << "mm"
                                     << (fromCache ? "[fast cache]" : "[manager fallback]");
        } else {
            qCWarning(CyberKnifeDoseLog) << "Failed to resolve OCR table info for beam"
                                         << (beamIndex + 1) << "collimator" << beam.collimatorSize;
        }
    }

    const int width = volume.width();
    const int height = volume.height();
    const int depth = volume.depth();

    // If Dynamic Grid Size Mode is enabled, use it for all beams
    if (m_resolutionOptions.useDynamicGridMode) {
        qCInfo(CyberKnifeDoseLog) << "Multi-beam calculation using Dynamic Grid Size Mode";

        const int beamCount = beams.size();
        const double epsilon = std::numeric_limits<double>::epsilon();
        bool anyContribution = false;

        // Calculate each beam and accumulate
        for (int beamIndex = 0; beamIndex < beamCount; ++beamIndex) {
            if (m_resolutionOptions.cancelRequested &&
                m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                qCInfo(CyberKnifeDoseLog) << "Calculation cancelled before processing beam" << (beamIndex + 1);
                return false;
            }
            const auto &beam = beams[beamIndex];
            const double weight = std::abs(weights[beamIndex]) > epsilon ? weights[beamIndex] : 0.0;

            if (weight <= epsilon) {
                qCInfo(CyberKnifeDoseLog) << "Skipping beam" << (beamIndex + 1) << "due to zero weight";
                continue;
            }

            qCInfo(CyberKnifeDoseLog) << "Calculating beam" << (beamIndex + 1) << "/" << beamCount
                                      << "with weight" << weight;

            // Create temporary dose volume for this beam
            RTDoseVolume tempDose;

            // Progress callback for this beam
            auto beamProgressCallback = [&](int percent) {
                if (progressCallback) {
                    double overall = (static_cast<double>(beamIndex) + percent / 100.0) / beamCount;
                    int overallPercent = static_cast<int>(overall * 100.0);
                    progressCallback(beamIndex, std::clamp(overallPercent, 0, 100));
                }
            };

            // Calculate this beam with Dynamic Grid Mode
            if (!calculateVolumeDoseWithProgress(volume, beam, tempDose, beamProgressCallback)) {
                m_lastErrors << QStringLiteral("ビーム %1 の計算に失敗しました。").arg(beamIndex + 1);
                return false;
            }

            if (m_resolutionOptions.cancelRequested &&
                m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
                qCInfo(CyberKnifeDoseLog) << "Calculation cancelled while accumulating beam" << (beamIndex + 1);
                return false;
            }

            // Accumulate weighted dose into result
            const cv::Mat &tempMat = tempDose.data();
            if (beamIndex == 0) {
                // First beam: just copy
                tempMat.copyTo(doseVolume.m_volume);
                if (weight != 1.0) {
                    doseVolume.m_volume *= weight;
                }
                anyContribution = true;
            } else {
                // Subsequent beams: accumulate
                if (weight != 1.0) {
                    cv::addWeighted(doseVolume.m_volume, 1.0, tempMat, weight, 0.0, doseVolume.m_volume);
                } else {
                    doseVolume.m_volume += tempMat;
                }
                anyContribution = true;
            }
        }

        if (m_resolutionOptions.cancelRequested &&
            m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
            return false;
        }

        // Calculate final max dose
        double minDose = 0.0;
        double maxDose = 0.0;
        cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
        doseVolume.m_maxDose = maxDose;

        if (progressCallback &&
            (!m_resolutionOptions.cancelRequested ||
             !m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed))) {
            progressCallback(beamCount - 1, 100);
        }

        return anyContribution;
    }

    // Standard mode: use configured step size and GPU acceleration (if enabled)
    qCInfo(CyberKnifeDoseLog) << "Multi-beam calculation using standard mode with configured step size";

    const int beamCount = beams.size();
    const double epsilon = std::numeric_limits<double>::epsilon();
    bool anyContribution = false;

    // Calculate each beam and accumulate
    for (int beamIndex = 0; beamIndex < beamCount; ++beamIndex) {
        if (m_resolutionOptions.cancelRequested &&
            m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
            qCInfo(CyberKnifeDoseLog) << "Calculation cancelled before processing beam" << (beamIndex + 1);
            return false;
        }
        const auto &beam = beams[beamIndex];
        const double weight = std::abs(weights[beamIndex]) > epsilon ? weights[beamIndex] : 0.0;

        if (weight <= epsilon) {
            qCInfo(CyberKnifeDoseLog) << "Skipping beam" << (beamIndex + 1) << "due to zero weight";
            if (progressCallback) {
                progressCallback(beamIndex, 100);
            }
            continue;
        }

        qCInfo(CyberKnifeDoseLog) << "Calculating beam" << (beamIndex + 1) << "/" << beamCount
                                  << "with weight" << weight;

        // Create temporary dose volume for this beam
        RTDoseVolume tempDose;

        // Progress callback for this beam
        auto beamProgressCallback = [&](int percent) {
            if (progressCallback) {
                double overall = (static_cast<double>(beamIndex) + percent / 100.0) / beamCount;
                int overallPercent = static_cast<int>(overall * 100.0);
                progressCallback(beamIndex, std::clamp(overallPercent, 0, 100));
            }
        };

        // Calculate this beam with standard mode (GPU will be used if enabled)
        if (!calculateVolumeDoseWithProgress(volume, beam, tempDose, beamProgressCallback)) {
            m_lastErrors << QStringLiteral("ビーム %1 の計算に失敗しました。").arg(beamIndex + 1);
            return false;
        }

        if (m_resolutionOptions.cancelRequested &&
            m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
            qCInfo(CyberKnifeDoseLog) << "Calculation cancelled while accumulating beam" << (beamIndex + 1);
            return false;
        }

        // Accumulate weighted dose into result
        const cv::Mat &tempMat = tempDose.data();
        if (beamIndex == 0) {
            // First beam: just copy
            tempMat.copyTo(doseVolume.m_volume);
            if (weight != 1.0) {
                doseVolume.m_volume *= weight;
            }
            anyContribution = true;
        } else {
            // Subsequent beams: accumulate
            if (weight != 1.0) {
                cv::addWeighted(doseVolume.m_volume, 1.0, tempMat, weight, 0.0, doseVolume.m_volume);
            } else {
                doseVolume.m_volume += tempMat;
            }
            anyContribution = true;
        }
    }

    if (!anyContribution) {
        m_lastErrors << QStringLiteral("有効なビームがありませんでした。");
        return false;
    }

    if (m_resolutionOptions.cancelRequested &&
        m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed)) {
        return false;
    }

    // Calculate final max dose
    double minDose = 0.0;
    double maxDose = 0.0;
    cv::minMaxIdx(doseVolume.m_volume, &minDose, &maxDose);
    doseVolume.m_maxDose = maxDose;

    if (progressCallback &&
        (!m_resolutionOptions.cancelRequested ||
         !m_resolutionOptions.cancelRequested->load(std::memory_order_relaxed))) {
        progressCallback(beamCount - 1, 100);
    }

    return true;
}

bool CyberKnifeDoseCalculator::resolveOcrTableInfo(double collimatorSize,
                                                   double *matchedCollimator,
                                                   QString *sourceFile) const
{
    if (!isReady()) {
        if (matchedCollimator) {
            *matchedCollimator = 0.0;
        }
        if (sourceFile) {
            sourceFile->clear();
        }
        return false;
    }

    const BeamDataParser::OCRData *managerOcr = nullptr;
    double managerMatched = 0.0;
    QString managerSource;

    if (m_fastLookup && m_fastLookup->isReady()) {
        double fastMatched = 0.0;
        QString fastSource;
        if (m_fastLookup->lookupOcrInfo(collimatorSize, &fastMatched, &fastSource)) {
            if (!m_beamDataManager) {
                if (matchedCollimator) {
                    *matchedCollimator = fastMatched;
                }
                if (sourceFile) {
                    *sourceFile = fastSource;
                }
                return true;
            }

            managerOcr = m_beamDataManager->findClosestOcrTable(collimatorSize, &managerMatched);
            if (!managerOcr) {
                if (matchedCollimator) {
                    *matchedCollimator = fastMatched;
                }
                if (sourceFile) {
                    *sourceFile = fastSource;
                }
                return true;
            }

            managerSource = managerOcr->sourceFileName;
            const bool sizeMatches = qFuzzyCompare(1.0 + fastMatched, 1.0 + managerMatched);
            const bool sourceMatches = fastSource.isEmpty() || managerSource.isEmpty()
                                      || fastSource == managerSource;
            if (sizeMatches && sourceMatches) {
                if (matchedCollimator) {
                    *matchedCollimator = fastMatched;
                }
                if (sourceFile) {
                    *sourceFile = fastSource;
                }
                return true;
            }

            qCWarning(CyberKnifeDoseLog)
                << "Fast OCR lookup mismatch for collimator" << collimatorSize << "mm:"
                << "cache matched" << fastMatched << "mm (" << fastSource << ")"
                << "manager matched" << managerMatched << "mm (" << managerSource << ")";
        }
    }

    if (!managerOcr) {
        managerOcr = m_beamDataManager->findClosestOcrTable(collimatorSize, &managerMatched);
        if (managerOcr) {
            managerSource = managerOcr->sourceFileName;
        }
    }

    double matched = managerMatched;
    const BeamDataParser::OCRData *ocr = managerOcr;
    if (!ocr) {
        if (matchedCollimator) {
            *matchedCollimator = 0.0;
        }
        if (sourceFile) {
            sourceFile->clear();
        }
        return false;
    }

    if (matchedCollimator) {
        *matchedCollimator = matched;
    }
    if (sourceFile) {
        *sourceFile = ocr->sourceFileName;
    }
    return true;
}

#ifdef ENABLE_GPU_DOSE_CALCULATION

bool CyberKnifeDoseCalculator::initializeGPU(bool useMultiGPU, int maxGPUs)
{
    if (m_gpuBackend && m_gpuBackend->isReady()) {
        return true; // Already initialized
    }

    qDebug() << "======================================";
    qDebug() << "CyberKnifeDoseCalculator: Initializing GPU backend...";
    if (useMultiGPU) {
        qDebug() << "Multi-GPU mode:" << (maxGPUs > 0 ? QString("enabled (max %1 GPUs)").arg(maxGPUs) : "enabled (all GPUs)");
    }
    qDebug() << "======================================";

    // Create GPU backend (auto-detect best available)
    m_gpuBackend = GPUDoseBackendFactory::createBackend(GPUBackendType::None, useMultiGPU, maxGPUs);

    if (!m_gpuBackend) {
        qWarning() << "CyberKnifeDoseCalculator: No GPU backend available";
        m_gpuEnabled = false;
        return false;
    }

    if (!m_gpuBackend->isReady()) {
        qWarning() << "CyberKnifeDoseCalculator: GPU backend initialization failed:" << m_gpuBackend->getLastError();
        m_gpuBackend.reset();
        m_gpuEnabled = false;
        return false;
    }

    qDebug() << "======================================";
    qDebug() << "CyberKnifeDoseCalculator: GPU backend initialized successfully";
    qDebug() << "GPU computing:" << m_gpuBackend->getBackendName();
    qDebug() << "GPU device:" << m_gpuBackend->getDeviceInfo();
    qDebug() << "======================================";

    m_gpuEnabled = true;
    return true;
}

bool CyberKnifeDoseCalculator::isGPUEnabled() const
{
    return m_gpuEnabled && m_gpuBackend && m_gpuBackend->isReady();
}

void CyberKnifeDoseCalculator::setGPUEnabled(bool enabled)
{
    if (enabled && !m_gpuBackend) {
        // Try to initialize GPU if not already done (with multi-GPU support)
        initializeGPU(true);
    } else if (!enabled) {
        m_gpuEnabled = false;
    } else {
        m_gpuEnabled = enabled;
    }
}

QString CyberKnifeDoseCalculator::getGPUDeviceInfo() const
{
    if (m_gpuBackend && m_gpuBackend->isReady()) {
        return QString("%1: %2")
            .arg(m_gpuBackend->getBackendName())
            .arg(m_gpuBackend->getDeviceInfo());
    }
    return "GPU not available";
}

int CyberKnifeDoseCalculator::getGPUCount() const
{
    if (!m_gpuBackend || !m_gpuBackend->isReady()) {
        return 0;
    }

    // Check if this is a multi-GPU backend
    if (m_gpuBackend->getBackendName().contains("Multi-GPU")) {
        // Extract GPU count from backend name (e.g., "Multi-GPU CUDA (2 GPUs)")
        QString name = m_gpuBackend->getBackendName();
        int start = name.indexOf('(');
        int end = name.indexOf(" GPU");
        if (start != -1 && end != -1) {
            QString countStr = name.mid(start + 1, end - start - 1);
            return countStr.toInt();
        }
    }

    // Single GPU backend
    return 1;
}

GPUBeamData CyberKnifeDoseCalculator::convertBeamDataToGPU(double collimatorSize) const
{
    GPUBeamData gpuData;

    if (!m_beamDataManager) {
        qWarning() << "CyberKnifeDoseCalculator::convertBeamDataToGPU: No beam data manager";
        return gpuData;
    }

    // Get DM (Output Factor) data
    const auto& dmData = m_beamDataManager->dmData();
    gpuData.ofDepthCount = static_cast<int>(dmData.depths.size());
    gpuData.ofCollimatorCount = static_cast<int>(dmData.collimatorSizes.size());

    // Flatten 2D matrix to 1D array (row-major: depth x collimator)
    gpuData.ofTable.reserve(gpuData.ofDepthCount * gpuData.ofCollimatorCount);
    for (const auto& row : dmData.outputFactorMatrix) {
        for (double val : row) {
            gpuData.ofTable.push_back(static_cast<float>(val));
        }
    }

    gpuData.ofDepths.reserve(dmData.depths.size());
    for (double d : dmData.depths) {
        gpuData.ofDepths.push_back(static_cast<float>(d));
    }

    gpuData.ofCollimators.reserve(dmData.collimatorSizes.size());
    for (double c : dmData.collimatorSizes) {
        gpuData.ofCollimators.push_back(static_cast<float>(c));
    }

    // Get TMR data
    const auto& tmrData = m_beamDataManager->tmrData();
    gpuData.tmrDepthCount = static_cast<int>(tmrData.depths.size());
    gpuData.tmrFieldSizeCount = static_cast<int>(tmrData.fieldSizes.size());

    // Flatten 2D matrix (row-major: depth x field size)
    gpuData.tmrTable.reserve(gpuData.tmrDepthCount * gpuData.tmrFieldSizeCount);
    for (const auto& row : tmrData.tmrValues) {
        for (double val : row) {
            gpuData.tmrTable.push_back(static_cast<float>(val));
        }
    }

    gpuData.tmrDepths.reserve(tmrData.depths.size());
    for (double d : tmrData.depths) {
        gpuData.tmrDepths.push_back(static_cast<float>(d));
    }

    gpuData.tmrFieldSizes.reserve(tmrData.fieldSizes.size());
    for (double fs : tmrData.fieldSizes) {
        gpuData.tmrFieldSizes.push_back(static_cast<float>(fs));
    }

    // Get OCR data for the given collimator
    double matchedCollimator = 0.0;
    const auto* ocrData = m_beamDataManager->findClosestOcrTable(collimatorSize, &matchedCollimator);

    if (ocrData) {
        gpuData.matchedCollimator = static_cast<float>(matchedCollimator);
        gpuData.ocrDepthCount = static_cast<int>(ocrData->depths.size());
        gpuData.ocrRadiusCount = static_cast<int>(ocrData->radii.size());
        gpuData.ocrCollimatorCount = 1; // Single collimator table

        // Flatten 2D matrix (row-major: depth x radius)
        gpuData.ocrTable.reserve(gpuData.ocrDepthCount * gpuData.ocrRadiusCount);
        for (const auto& row : ocrData->ratios) {
            for (double val : row) {
                gpuData.ocrTable.push_back(static_cast<float>(val));
            }
        }

        gpuData.ocrDepths.reserve(ocrData->depths.size());
        for (double d : ocrData->depths) {
            gpuData.ocrDepths.push_back(static_cast<float>(d));
        }

        gpuData.ocrRadii.reserve(ocrData->radii.size());
        for (double r : ocrData->radii) {
            gpuData.ocrRadii.push_back(static_cast<float>(r));
        }

        gpuData.ocrCollimators.push_back(static_cast<float>(matchedCollimator));

        qDebug() << "GPU Beam Data: OCR table for collimator" << collimatorSize
                 << "matched to" << matchedCollimator << "mm";
    } else {
        qWarning() << "GPU Beam Data: No OCR table found for collimator" << collimatorSize;
    }

    qDebug() << "GPU Beam Data conversion complete:"
             << "OF:" << gpuData.ofDepthCount << "x" << gpuData.ofCollimatorCount
             << "TMR:" << gpuData.tmrDepthCount << "x" << gpuData.tmrFieldSizeCount
             << "OCR:" << gpuData.ocrDepthCount << "x" << gpuData.ocrRadiusCount;

    return gpuData;
}

GPUComputeParams CyberKnifeDoseCalculator::createGPUComputeParams(
    const DicomVolume &volume,
    const GeometryCalculator::BeamGeometry &beam,
    int stepX, int stepY, int stepZ,
    double referenceDose) const
{
    GPUComputeParams params;

    // Volume dimensions
    params.width = volume.width();
    params.height = volume.height();
    params.depth = volume.depth();

    // Voxel spacing
    params.spacingX = volume.spacingX();
    params.spacingY = volume.spacingY();
    params.spacingZ = volume.spacingZ();

    // Volume origin
    params.originX = volume.originX();
    params.originY = volume.originY();
    params.originZ = volume.originZ();

    // Image orientation (direction cosines)
    const double* rowDir = volume.rowDirection();
    const double* colDir = volume.colDirection();
    const double* sliceDir = volume.sliceDirection();

    for (int i = 0; i < 3; i++) {
        params.orientationX[i] = rowDir[i];
        params.orientationY[i] = colDir[i];
        params.orientationZ[i] = sliceDir[i];
    }

    // Step size
    params.stepX = stepX;
    params.stepY = stepY;
    params.stepZ = stepZ;

    params.gridCountX = computeSteppedIndexCount(params.width, stepX);
    params.gridCountY = computeSteppedIndexCount(params.height, stepY);
    params.gridCountZ = computeSteppedIndexCount(params.depth, stepZ);

    // Reference dose
    params.referenceDose = referenceDose;

    // Beam geometry (make a copy and ensure basis is initialized)
    params.beam = beam;
    params.beam.initializeBasis();

    return params;
}

#endif // ENABLE_GPU_DOSE_CALCULATION

} // namespace CyberKnife

