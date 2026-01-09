#include "data/file_structure_manager.h"
#include "database/database_manager.h"

#include <fstream>
#include <filesystem>

namespace {
static bool dirExists(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    return fs::is_directory(p, ec);
}
}

FileStructureManager::FileStructureManager(DatabaseManager& db) : m_db(db) {}

bool FileStructureManager::ensureDir(const std::string& path) {
    if (dirExists(path)) return true;
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return dirExists(path);
}

bool FileStructureManager::ensureFile(const std::string& path, const std::string& contentIfCreate) {
    std::ifstream fin(path);
    if (fin.good()) return true;
    fin.close();
    std::ofstream out(path);
    if (!out) return false;
    out << contentIfCreate;
    return true;
}

std::string FileStructureManager::ensurePatientStructure(const std::string& patientName,
                                                         const std::string& patientId) {
    const std::string key = patientName + "_" + patientId;
    const std::string patientDir = m_db.dataRoot() + "/Patients/" + key;

    ensureDir(m_db.dataRoot());
    ensureDir(m_db.dataRoot() + "/Patients");
    ensureDir(patientDir);

    // Images
    ensureDir(patientDir + "/Images");
    ensureDir(patientDir + "/Images/CT");
    ensureDir(patientDir + "/Images/MRI");
    ensureDir(patientDir + "/Images/PET");
    ensureDir(patientDir + "/Images/Others");
    ensureDir(patientDir + "/Images/Fusion");
    ensureDir(patientDir + "/Images/Fusion/CT");
    ensureDir(patientDir + "/Images/Fusion/MRI");

    // RT_Data
    ensureDir(patientDir + "/RT_Data");
    ensureDir(patientDir + "/RT_Data/Structures");
    ensureDir(patientDir + "/RT_Data/Plans");
    ensureDir(patientDir + "/RT_Data/Doses");
    ensureDir(patientDir + "/RT_Data/Analysis");

    // AI_Results
    ensureDir(patientDir + "/AI_Results");
    ensureDir(patientDir + "/AI_Results/Segmentation");
    ensureDir(patientDir + "/AI_Results/Analysis");

    // patient_info.txt
    const std::string info = patientDir + "/patient_info.txt";
    ensureFile(info,
               "# Patient Info\n"
               "Name: " + patientName + "\n"
               "ID: " + patientId + "\n"
               "Created: (auto)\n");

    return patientDir;
}

std::string FileStructureManager::ensurePatientFolderFor(const std::string& patientName,
                                                         const std::string& patientId) {
    // Search for existing folder by name prefix
    const std::string patientsRoot = m_db.dataRoot() + "/Patients";
    ensureDir(m_db.dataRoot());
    ensureDir(patientsRoot);

    // Scan for existing folder starting with "Name_"
    std::string existing;
#if __has_include(<filesystem>)
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(patientsRoot, ec)) {
        if (ec) break;
        if (!e.is_directory()) continue;
        auto fname = e.path().filename().string();
        if (fname.rfind(patientName + "_", 0) == 0) { existing = e.path().string(); break; }
    }
#endif
    if (!existing.empty()) {
        // Ensure standard structure inside
        const std::string key = existing.substr(existing.find_last_of("/") + 1);
        const std::string name = patientName; (void)name; (void)key; // name kept for clarity
        // Build subdirs if missing
        ensureDir(existing);
        ensureDir(existing + "/Images");
        ensureDir(existing + "/Images/CT");
        ensureDir(existing + "/Images/MRI");
        ensureDir(existing + "/Images/PET");
        ensureDir(existing + "/Images/Others");
        ensureDir(existing + "/Images/Fusion");
        ensureDir(existing + "/Images/Fusion/CT");
        ensureDir(existing + "/Images/Fusion/MRI");
        ensureDir(existing + "/RT_Data");
        ensureDir(existing + "/RT_Data/Structures");
        ensureDir(existing + "/RT_Data/Plans");
        ensureDir(existing + "/RT_Data/Doses");
        ensureDir(existing + "/RT_Data/Analysis");
        ensureDir(existing + "/AI_Results");
        ensureDir(existing + "/AI_Results/Segmentation");
        ensureDir(existing + "/AI_Results/Analysis");
        ensureFile(existing + "/patient_info.txt",
                   "# Patient Info\nName: " + patientName + "\nID: " + patientId + "\n");
        return existing;
    }
    return ensurePatientStructure(patientName, patientId);
}
