#include "database/database_manager.h"

#include <sqlite3.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <filesystem>
#include <set>

#if defined(_WIN32)
#  include <direct.h>
#  define MKDIR(path) _mkdir(path)
#else
#  include <unistd.h>
#  define MKDIR(path) mkdir(path, 0755)
#endif

namespace {
static bool fileExists(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}
static bool dirExists(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    return fs::is_directory(p, ec);
}
static bool ensureDir(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (dirExists(p)) return true;
    fs::create_directories(p, ec);
    return dirExists(p);
}
}

DatabaseManager::DatabaseManager(const std::string& dataRoot)
    : m_dataRoot(dataRoot) {}

DatabaseManager::~DatabaseManager() {
    if (m_db) {
        sqlite3_close(m_db);
        m_db = nullptr;
    }
}

bool DatabaseManager::open() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!ensureDirectories()) return false;

    if (sqlite3_open(databasePath().c_str(), &m_db) != SQLITE_OK) {
        // Try restore then reopen
        restoreIfNeeded();
        if (sqlite3_open(databasePath().c_str(), &m_db) != SQLITE_OK) {
            return false;
        }
    }
    // Pragmas for better reliability/speed
    exec("PRAGMA journal_mode=WAL;");
    exec("PRAGMA synchronous=NORMAL;");
    return ensureSchema();
}

bool DatabaseManager::isOpen() const { return m_db != nullptr; }

std::string DatabaseManager::lastError() const {
    if (!m_db) return "Database not open";
    const char* errMsg = sqlite3_errmsg(m_db);
    return errMsg ? std::string(errMsg) : "Unknown error";
}

bool DatabaseManager::exec(const std::string& sql) {
    if (!m_db) return false;
    char* err = nullptr;
    int rc = sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (err) sqlite3_free(err);
        return false;
    }
    return true;
}

bool DatabaseManager::query(const std::string& sql,
                            const std::function<void(int, char**, char**)> &rowCallback) {
    if (!m_db) return false;
    char* err = nullptr;
    auto cb = [](void* ud, int argc, char** argv, char** colNames)->int {
        auto* fn = static_cast<const std::function<void(int, char**, char**)>*>(ud);
        (*fn)(argc, argv, colNames);
        return 0;
    };
    int rc = sqlite3_exec(m_db, sql.c_str(), cb, (void*)&rowCallback, &err);
    if (rc != SQLITE_OK) {
        if (err) sqlite3_free(err);
        return false;
    }
    return true;
}

bool DatabaseManager::beginTransaction() {
    if (m_inTx) return false;
    if (!exec("BEGIN TRANSACTION;")) return false;
    m_inTx = true;
    return true;
}

bool DatabaseManager::commit() {
    if (!m_inTx) return false;
    bool ok = exec("COMMIT;");
    m_inTx = false;
    return ok;
}

bool DatabaseManager::rollback() {
    if (!m_inTx) return false;
    bool ok = exec("ROLLBACK;");
    m_inTx = false;
    return ok;
}

bool DatabaseManager::backup() {
    if (!m_db) return false;
    std::string src = databasePath();
    std::string dst = databaseDir() + "/simple.db.bak";
    std::ifstream in(src, std::ios::binary);
    std::ofstream out(dst, std::ios::binary);
    if (!in || !out) return false;
    out << in.rdbuf();
    return true;
}

bool DatabaseManager::restoreIfNeeded() {
    // If main DB missing but backup exists, restore.
    std::string src = databaseDir() + "/simple.db.bak";
    std::string dst = databasePath();
    if (!fileExists(dst) && fileExists(src)) {
        std::ifstream in(src, std::ios::binary);
        std::ofstream out(dst, std::ios::binary);
        if (!in || !out) return false;
        out << in.rdbuf();
        return true;
    }
    return true;
}

std::string DatabaseManager::dataRoot() const { return m_dataRoot; }
std::string DatabaseManager::databaseDir() const { return m_dataRoot + "/Database"; }
std::string DatabaseManager::databasePath() const { return databaseDir() + "/simple.db"; }
std::string DatabaseManager::schemaPath() const { return std::string("resources/sql/create_tables.sql"); }

static std::string getEnv(const char* key) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
}

std::string DatabaseManager::defaultDataRoot() {
    // Allow override via environment variable for flexibility
    if (auto env = getEnv("SHIORIS3_DATA_ROOT"); !env.empty()) return env;
#if defined(_WIN32)
    std::string home = getEnv("USERPROFILE");
    if (home.empty()) home = getEnv("HOME");
    if (home.empty()) home = ".";
    return home + "/ShioRIS3_Data";
#else
    std::string home = getEnv("HOME");
    if (home.empty()) home = ".";
    return home + "/ShioRIS3_Data";
#endif
}

void DatabaseManager::setDataRoot(const std::string& newRoot) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_db) {
        sqlite3_close(m_db);
        m_db = nullptr;
        m_inTx = false;
    }
    m_dataRoot = newRoot;
}

bool DatabaseManager::ensureDirectories() {
    if (!ensureDir(m_dataRoot)) return false;
    if (!ensureDir(databaseDir())) return false;
    if (!ensureDir(m_dataRoot + "/Patients")) return false;
    return true;
}

bool DatabaseManager::ensureSchema() {
    // Load schema SQL and execute
    std::ifstream fin(schemaPath());
    bool schemaOk = false;
    if (fin) {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        schemaOk = exec(buffer.str());
    } else {
        // Fallback minimal schema
        const char* fallback = R"SQL(
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version(version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
CREATE TABLE IF NOT EXISTS patients (
    patient_key TEXT PRIMARY KEY,
    name TEXT,
    created_at INTEGER,
    info_path TEXT
);
CREATE TABLE IF NOT EXISTS studies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_key TEXT NOT NULL,
    modality TEXT NOT NULL,
    study_date TEXT,
    study_name TEXT,
    path TEXT NOT NULL,
    frame_uid TEXT,
    series_uid TEXT,
    series_description TEXT,
    FOREIGN KEY(patient_key) REFERENCES patients(patient_key) ON DELETE CASCADE
);
DROP INDEX IF EXISTS idx_studies_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_studies_unique ON studies(patient_key, modality, path, series_uid);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id INTEGER NOT NULL,
    relative_path TEXT NOT NULL,
    size_bytes INTEGER,
    mtime INTEGER,
    checksum TEXT,
    file_type TEXT,
    preview_path TEXT,
    UNIQUE(study_id, relative_path),
    FOREIGN KEY(study_id) REFERENCES studies(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS CyberKnifeBeamData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type TEXT NOT NULL,
    collimator_size REAL,
    depth REAL,
    radius REAL,
    factor_value REAL NOT NULL,
    file_source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
    )SQL";
        schemaOk = exec(fallback);
    }
    if (!schemaOk) return false;

    std::set<std::string> studyColumns;
    if (!query("PRAGMA table_info(studies);", [&](int argc, char** argv, char**){
        if (argc >= 2 && argv[1]) studyColumns.insert(argv[1]);
    })) {
        return false;
    }

    auto ensureColumn = [&](const std::string& name, const std::string& typeWithDefault) -> bool {
        if (studyColumns.find(name) != studyColumns.end()) return true;
        std::stringstream alter;
        alter << "ALTER TABLE studies ADD COLUMN " << name << " " << typeWithDefault << ";";
        if (!exec(alter.str())) return false;
        studyColumns.insert(name);
        return true;
    };

    if (!ensureColumn("series_uid", "TEXT DEFAULT ''")) return false;
    if (!ensureColumn("series_description", "TEXT DEFAULT ''")) return false;

    if (!exec("UPDATE studies SET series_uid='' WHERE series_uid IS NULL;")) return false;
    if (!exec("UPDATE studies SET series_description='' WHERE series_description IS NULL;")) return false;
    if (!exec("DROP INDEX IF EXISTS idx_studies_unique;")) return false;
    if (!exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_studies_unique ON studies(patient_key, modality, path, series_uid);")) return false;

    // Ensure dose_volumes table exists (added for CyberKnife dose storage)
    const char* doseVolumesSql = R"SQL(
CREATE TABLE IF NOT EXISTS dose_volumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_key TEXT,
    study_id INTEGER,
    calculation_type TEXT NOT NULL,
    beam_count INTEGER,
    file_path TEXT NOT NULL,
    format TEXT NOT NULL DEFAULT 'RTDOSE',
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    spacing_x REAL NOT NULL,
    spacing_y REAL NOT NULL,
    spacing_z REAL NOT NULL,
    origin_x REAL,
    origin_y REAL,
    origin_z REAL,
    max_dose REAL,
    frame_uid TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    FOREIGN KEY(patient_key) REFERENCES patients(patient_key) ON DELETE CASCADE,
    FOREIGN KEY(study_id) REFERENCES studies(id) ON DELETE SET NULL
);
    )SQL";
    if (!exec(doseVolumesSql)) return false;

    return true;
}
