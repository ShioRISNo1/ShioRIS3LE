#include "database/database_manager.h"
#include "database/smart_scanner.h"
#include "database/database_sync_manager.h"

#include <QDir>
#include <QFileInfo>

DatabaseSyncManager::DatabaseSyncManager(DatabaseManager& db, SmartScanner& scanner, QObject* parent)
    : QObject(parent), m_db(db), m_scanner(scanner) {
    connect(&m_watcher, &QFileSystemWatcher::fileChanged, this, &DatabaseSyncManager::onFileChanged);
    connect(&m_watcher, &QFileSystemWatcher::directoryChanged, this, &DatabaseSyncManager::onDirChanged);
    connect(&m_debounceTimer, &QTimer::timeout, this, &DatabaseSyncManager::flushQueue);
    m_debounceTimer.setInterval(500); // debounce burst of events
    m_debounceTimer.setSingleShot(true);
}

void DatabaseSyncManager::start() {
    stop();
    const QString patientsDir = QString::fromStdString(m_db.dataRoot()) + "/Patients";
    addRecursive(patientsDir);
}

void DatabaseSyncManager::stop() {
    if (!m_watcher.files().isEmpty()) m_watcher.removePaths(m_watcher.files());
    if (!m_watcher.directories().isEmpty()) m_watcher.removePaths(m_watcher.directories());
}

void DatabaseSyncManager::onFileChanged(const QString& path) {
    m_pending.insert(path);
    m_debounceTimer.start();
}

void DatabaseSyncManager::onDirChanged(const QString& path) {
    // Re-add subdirectories in case of new folders
    addRecursive(path);
    m_pending.insert(path);
    m_debounceTimer.start();
}

void DatabaseSyncManager::flushQueue() {
    for (const auto& p : m_pending) {
        m_scanner.scanPath(p.toStdString());
        emit syncEvent(p);
    }
    m_pending.clear();
}

void DatabaseSyncManager::addRecursive(const QString& dirPath) {
    QDir dir(dirPath);
    if (!dir.exists()) return;
    if (!m_watcher.directories().contains(dir.absolutePath()))
        m_watcher.addPath(dir.absolutePath());

    QFileInfoList entries = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo& fi : entries) {
        addRecursive(fi.absoluteFilePath());
    }
    // Also watch files in this directory
    QFileInfoList files = dir.entryInfoList(QDir::Files);
    for (const QFileInfo& fi : files) {
        if (!m_watcher.files().contains(fi.absoluteFilePath()))
            m_watcher.addPath(fi.absoluteFilePath());
    }
}
