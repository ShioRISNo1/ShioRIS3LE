#pragma once

#include <string>

class DatabaseManager;

// FileStructureManager: Manages patient folder creation and standard layout.
class FileStructureManager {
public:
    explicit FileStructureManager(DatabaseManager& db);

    // Ensure patient folder exists with standard subdirectories and patient_info.txt
    // Returns absolute patient directory path.
    std::string ensurePatientStructure(const std::string& patientName,
                                       const std::string& patientId);

    // Finds or creates a patient folder by patient name, regardless of differing IDs.
    // If a folder with prefix "Name_" exists, reuse it; otherwise create Name_ID.
    std::string ensurePatientFolderFor(const std::string& patientName,
                                       const std::string& patientId);

private:
    bool ensureDir(const std::string& path);
    bool ensureFile(const std::string& path, const std::string& contentIfCreate);

    DatabaseManager& m_db;
};
