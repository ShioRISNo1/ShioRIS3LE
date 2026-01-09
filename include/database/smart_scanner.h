#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class DatabaseManager;

// SmartScanner: Scans ShioRIS3_Data directory structure and syncs to DB.
class SmartScanner {
public:
    explicit SmartScanner(DatabaseManager& db);

    // Perform full scan and attempt to repair inconsistencies.
    bool fullScanAndRepair();

    // Scan a specific patient or path (incremental).
    bool scanPath(const std::string& absPath);

private:
    bool ensurePatient(const std::string& patientKey,
                       const std::string& name,
                       const std::string& infoPath);

    bool upsertStudy(const std::string& patientKey,
                     const std::string& modality,
                     const std::string& studyDate,
                     const std::string& studyName,
                     const std::string& absPath,
                     const std::string& frameUID,
                     const std::string& seriesUID,
                     const std::string& seriesDescription,
                     int& outStudyId);

    bool upsertFile(int studyId,
                    const std::string& baseDir,
                    const std::string& filePath);

    bool clearStudiesForPath(const std::string& patientKey,
                             const std::string& absPath);

    std::string extractDateFromFolder(const std::string& folderName) const;
    std::string extractNameFromFolder(const std::string& folderName) const;
    std::string toRelative(const std::string& base, const std::string& abs) const;

    DatabaseManager& m_db;
};

