#pragma once

#include <string>
#include <functional>
#include <mutex>

struct sqlite3;

// DatabaseManager: Thin RAII wrapper around SQLite3 with basic utilities.
class DatabaseManager {
public:
    // Initialize with root data directory (e.g., "ShioRIS3_Data").
    explicit DatabaseManager(const std::string& dataRoot);
    ~DatabaseManager();

    // Returns the default data root path located directly under the user's
    // home directory:
    // - Windows: %USERPROFILE%/ShioRIS3_Data
    // - Unix-like systems: $HOME/ShioRIS3_Data
    static std::string defaultDataRoot();

    DatabaseManager(const DatabaseManager&) = delete;
    DatabaseManager& operator=(const DatabaseManager&) = delete;

    // Opens or creates the database and ensures schema exists.
    bool open();

    // Returns true if a database connection is open.
    bool isOpen() const;

    // Executes a single SQL statement without result rows.
    bool exec(const std::string& sql);

    // Returns the last SQLite error message
    std::string lastError() const;

    // Executes a query and iterates rows via callback(lambda receives argc, argv, colNames).
    bool query(const std::string& sql,
               const std::function<void(int, char**, char**)> &rowCallback);

    // Begins a transaction. Nesting is not supported.
    bool beginTransaction();
    bool commit();
    bool rollback();

    // Backup current DB to `simple.db.bak` in the same directory.
    bool backup();

    // Attempt to restore from backup if main DB is corrupted/missing.
    bool restoreIfNeeded();

    // Absolute paths
    std::string dataRoot() const;              // e.g., ShioRIS3_Data
    std::string databaseDir() const;           // ShioRIS3_Data/Database
    std::string databasePath() const;          // ShioRIS3_Data/Database/simple.db
    std::string schemaPath() const;            // resources/sql/create_tables.sql (resolved from cwd)

    // Change data root directory. Closes any open DB. Call open() again afterwards.
    void setDataRoot(const std::string& newRoot);

private:
    bool ensureDirectories();
    bool ensureSchema();

    std::string m_dataRoot;
    sqlite3* m_db {nullptr};
    std::mutex m_mutex;
    bool m_inTx {false};
};
