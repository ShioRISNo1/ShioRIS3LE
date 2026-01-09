#include "data/metadata_generator.h"
#include "database/database_manager.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

MetadataGenerator::MetadataGenerator(DatabaseManager& db) : m_db(db) {}

bool MetadataGenerator::writeGlobalIndex() {
    if (!m_db.isOpen()) return false;
    std::stringstream json;
    json << "{\n  \"patients\": [\n";
    bool firstPatient = true;
    m_db.query("SELECT patient_key, name FROM patients ORDER BY name;",
               [&](int argc, char** argv, char**){
        if (argc < 2) return;
        std::string key = argv[0] ? argv[0] : "";
        std::string name = argv[1] ? argv[1] : "";
        if (!firstPatient) json << ",\n";
        firstPatient = false;
        json << "    {\n      \"patient_key\": \"" << key << "\",\n      \"name\": \"" << name << "\",\n      \"studies\": [";

        bool firstStudy = true;
        std::stringstream q;
        q << "SELECT id, modality, study_date, study_name, path FROM studies WHERE patient_key='" << key << "' ORDER BY id;";
        m_db.query(q.str(), [&](int argc2, char** argv2, char**){
            if (argc2 < 5) return;
            if (!firstStudy) json << ",";
            firstStudy = false;
            json << "{\"id\":" << (argv2[0] ? argv2[0] : "0")
                 << ",\"modality\":\"" << (argv2[1] ? argv2[1] : "")
                 << "\",\"study_date\":\"" << (argv2[2] ? argv2[2] : "")
                 << "\",\"study_name\":\"" << (argv2[3] ? argv2[3] : "")
                 << "\",\"path\":\"" << (argv2[4] ? argv2[4] : "")
                 << "\"}";
        });
        json << "]\n    }";
    });
    json << "\n  ]\n}\n";

    const std::string outPath = m_db.databaseDir() + "/index.json";
    std::ofstream out(outPath);
    if (!out) return false;
    out << json.str();
    return true;
}

std::string MetadataGenerator::generatePreview(const std::string& absFilePath) {
    // Only support common images for now
    auto ext = fs::path(absFilePath).extension().string();
    for (auto& c : ext) c = std::tolower(c);
    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") {
        return std::string();
    }

    cv::Mat img = cv::imread(absFilePath, cv::IMREAD_COLOR);
    if (img.empty()) return std::string();

    int w = img.cols, h = img.rows;
    const int maxW = 256;
    if (w > maxW) {
        int nh = static_cast<int>(h * (maxW / static_cast<double>(w)));
        cv::resize(img, img, cv::Size(maxW, nh));
    }

    const std::string prevDir = m_db.databaseDir() + "/previews";
    std::error_code ec;
    fs::create_directories(prevDir, ec);
    const std::string outPath = prevDir + "/" + fs::path(absFilePath).filename().string() + ".jpg";
    cv::imwrite(outPath, img);
    return outPath;
}
