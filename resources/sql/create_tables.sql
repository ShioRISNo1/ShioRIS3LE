-- Schema for ShioRIS3 simple.db
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

INSERT INTO schema_version(version)
    SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);

-- Patients: key is patient_key like "Name_ID"
CREATE TABLE IF NOT EXISTS patients (
    patient_key TEXT PRIMARY KEY,
    name TEXT,
    created_at INTEGER,
    info_path TEXT
);

-- Studies per modality and date under a patient
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

-- Ensure uniqueness to avoid duplicate study rows on rescans
DROP INDEX IF EXISTS idx_studies_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_studies_unique
ON studies(patient_key, modality, path, series_uid);

-- Files tracked for each study
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

-- Relationships between files (e.g., DICOM-RT to RTSTRUCT)
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    related_file_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
    FOREIGN KEY(related_file_id) REFERENCES files(id) ON DELETE CASCADE
);

-- AI results top-level entries (optional convenience)
CREATE TABLE IF NOT EXISTS ai_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_key TEXT NOT NULL,
    category TEXT,
    date TEXT,
    model TEXT,
    path TEXT NOT NULL,
    FOREIGN KEY(patient_key) REFERENCES patients(patient_key) ON DELETE CASCADE
);

-- Simple change log
CREATE TABLE IF NOT EXISTS change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time INTEGER NOT NULL,
    action TEXT NOT NULL,
    path TEXT,
    details TEXT
);

-- Dose volumes table for tracking saved dose distributions
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
