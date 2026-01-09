#pragma once

#include <string>

class DatabaseManager;

// MetadataGenerator: Creates index.json and simple preview images for supported files.
class MetadataGenerator {
public:
    explicit MetadataGenerator(DatabaseManager& db);

    // Regenerates global index.json under dataRoot/Database/index.json
    bool writeGlobalIndex();

    // Generate a preview image for an absolute file path if supported.
    // Returns preview absolute path or empty.
    std::string generatePreview(const std::string& absFilePath);

private:
    DatabaseManager& m_db;
};

