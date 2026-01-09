#pragma once

#include <QObject>
#include <QString>
#include <QFileSystemWatcher>
#include <QTimer>
#include <unordered_set>

class DatabaseManager;
class SmartScanner;

// DatabaseSyncManager: Watches filesystem and synchronizes changes to DB.
class DatabaseSyncManager : public QObject {
    Q_OBJECT
public:
    DatabaseSyncManager(DatabaseManager& db, SmartScanner& scanner, QObject* parent = nullptr);

    // Start watching under dataRoot/Patients recursively.
    void start();

    // Stop watching
    void stop();

signals:
    void syncEvent(const QString& path);

private slots:
    void onFileChanged(const QString& path);
    void onDirChanged(const QString& path);
    void flushQueue();

private:
    void addRecursive(const QString& dirPath);

    DatabaseManager& m_db;
    SmartScanner& m_scanner;
    QFileSystemWatcher m_watcher;
    QTimer m_debounceTimer;
    std::unordered_set<QString> m_pending;
};

