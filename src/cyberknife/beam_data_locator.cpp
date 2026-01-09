#include "cyberknife/beam_data_locator.h"

#include <QCoreApplication>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QProcessEnvironment>
#include <QSettings>
#include <utility>

namespace CyberKnife {

Q_LOGGING_CATEGORY(CyberKnifeBeamLocatorLog, "cyberknife.beamdata.locator")

static QStringList candidatePathsFromEnvironment()
{
    QStringList paths;
    const QByteArray envPath = qgetenv("CYBERKNIFE_BEAM_DATA_PATH");
    if (!envPath.isEmpty()) {
        paths << QString::fromLocal8Bit(envPath);
    }
    return paths;
}

static QStringList candidatePathsFromSettings()
{
    QStringList paths;
    QSettings settings("ShioRIS3", "ShioRIS3");
    const QString explicitPath = settings.value("cyberknife/beamDataPath").toString();
    if (!explicitPath.isEmpty()) {
        paths << explicitPath;
    }
    const QString legacyPath = settings.value("paths/beamDataRoot").toString();
    if (!legacyPath.isEmpty()) {
        paths << legacyPath;
    }
    return paths;
}

static QStringList candidatePathsNearApplication()
{
    QStringList paths;
    if (const auto *app = QCoreApplication::instance()) {
        const QString appDir = app->applicationDirPath();
        const QDir dir(appDir);
        paths << dir.filePath("beam_data");
        paths << dir.filePath("../share/shioris3/beam_data");
    }
    return paths;
}

QString BeamDataLocator::resolveBeamDataDirectory(const QString &preferredPath)
{
    QStringList candidates;
    candidates.reserve(8);

    appendIfValid(candidates, preferredPath);

    for (const QString &path : candidatePathsFromEnvironment()) {
        appendIfValid(candidates, path);
    }

    for (const QString &path : candidatePathsFromSettings()) {
        appendIfValid(candidates, path);
    }

    for (const QString &path : candidatePathsNearApplication()) {
        appendIfValid(candidates, path);
    }

    for (const QString &candidate : std::as_const(candidates)) {
        if (candidate.isEmpty()) {
            continue;
        }

        const QDir dir(candidate);
        if (!dir.exists()) {
            continue;
        }

        if (!hasRequiredFiles(dir)) {
            qCDebug(CyberKnifeBeamLocatorLog) << "Candidate" << candidate
                                              << "is missing required beam data files.";
            continue;
        }

        qCInfo(CyberKnifeBeamLocatorLog) << "Beam data directory resolved:" << dir.absolutePath();
        return dir.absolutePath();
    }

    qCWarning(CyberKnifeBeamLocatorLog) << "Failed to resolve CyberKnife beam data directory.";
    return QString();
}

bool BeamDataLocator::hasRequiredFiles(const QDir &dir)
{
    const QString dmTable = dir.filePath("DMTable.dat");
    const QString tmrTable = dir.filePath("TMRtable.dat");
    const QStringList ocrTables = dir.entryList({"OCRtable*.dat"}, QDir::Files | QDir::Readable);

    if (!QFileInfo::exists(dmTable)) {
        return false;
    }
    if (!QFileInfo::exists(tmrTable)) {
        return false;
    }
    if (ocrTables.isEmpty()) {
        return false;
    }
    return true;
}

void BeamDataLocator::appendIfValid(QStringList &list, const QString &path)
{
    if (path.isEmpty()) {
        return;
    }

    const QString cleaned = QDir::cleanPath(path);
    if (!list.contains(cleaned)) {
        list.append(cleaned);
    }
}

} // namespace CyberKnife
