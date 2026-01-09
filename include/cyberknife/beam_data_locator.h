#pragma once

#include <QDir>
#include <QLoggingCategory>
#include <QString>
#include <QStringList>

namespace CyberKnife {

Q_DECLARE_LOGGING_CATEGORY(CyberKnifeBeamLocatorLog)

class BeamDataLocator {
public:
    static QString resolveBeamDataDirectory(const QString &preferredPath = QString());

private:
    static bool hasRequiredFiles(const QDir &dir);
    static void appendIfValid(QStringList &list, const QString &path);
};

} // namespace CyberKnife
