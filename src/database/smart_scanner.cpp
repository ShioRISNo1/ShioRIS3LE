#include "database/database_manager.h"
#include "database/smart_scanner.h"

#include <filesystem>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <dcmtk/dcmdata/dctk.h>

using namespace std;
namespace fs = std::filesystem;

// Helper to extract modality and FrameOfReferenceUID from a DICOM file
static bool readDicomMeta(const std::string& file,
                         std::string& modality,
                         std::string& studyDate,
                         std::string& studyName,
                         std::string& frameUID,
                         std::string& seriesUID,
                         std::string& seriesDescription) {
    DcmFileFormat ff;
    if (ff.loadFile(file.c_str()).bad()) return false;
    DcmDataset* ds = ff.getDataset();
    OFString val;
    if (ds->findAndGetOFString(DCM_Modality, val).good()) modality = val.c_str();
    if (ds->findAndGetOFString(DCM_StudyDate, val).good()) studyDate = val.c_str();
    if (ds->findAndGetOFString(DCM_StudyDescription, val).good()) studyName = val.c_str();
    if (ds->findAndGetOFString(DCM_FrameOfReferenceUID, val).good()) {
        frameUID = val.c_str();
    } else {
        // RTSTRUCTなどではReferencedFrameOfReferenceSequence内にUIDがある場合がある
        DcmItem* seqItem = nullptr;
        if (ds->findAndGetSequenceItem(DCM_ReferencedFrameOfReferenceSequence, seqItem).good() && seqItem) {
            if (seqItem->findAndGetOFString(DCM_FrameOfReferenceUID, val).good()) frameUID = val.c_str();
        }
    }
    if (ds->findAndGetOFString(DCM_SeriesInstanceUID, val).good()) seriesUID = val.c_str();
    if (ds->findAndGetOFString(DCM_SeriesDescription, val).good()) seriesDescription = val.c_str();
    return !modality.empty();
}

static std::string normalizeModality(const std::string& m) {
    std::string up = m;
    std::transform(up.begin(), up.end(), up.begin(), ::toupper);
    if (up == "MR") return "MRI";
    if (up == "PT") return "PET";
    if (up == "CT" || up == "MRI" || up == "PET") return up;
    if (up.rfind("RT", 0) == 0) return up; // RTPLAN, RTSTRUCT, RTDOSE
    return "OTHERS";
}

static inline long long toUnixTime(std::filesystem::file_time_type tp) {
    using namespace std::chrono;
    auto sctp = time_point_cast<system_clock::duration>(tp - fs::file_time_type::clock::now()
                                                        + system_clock::now());
    return duration_cast<seconds>(sctp.time_since_epoch()).count();
}

static std::string escapeSql(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\'') {
            o += "''";
        } else {
            o += c;
        }
    }
    return o;
}

SmartScanner::SmartScanner(DatabaseManager& db) : m_db(db) {}

bool SmartScanner::fullScanAndRepair() {
    if (!m_db.isOpen()) return false;
    // Consolidate duplicate studies if any (keep min id)
    m_db.beginTransaction();
    m_db.exec("WITH d AS (\n"
              "  SELECT MIN(id) AS keep_id, patient_key, modality, path, COALESCE(series_uid, '') AS suid\n"
              "  FROM studies\n"
              "  GROUP BY patient_key, modality, path, COALESCE(series_uid, '')\n"
              ")\n"
              "UPDATE files SET study_id = (\n"
              "  SELECT keep_id FROM d JOIN studies s ON s.id = files.study_id\n"
              "  WHERE d.patient_key = s.patient_key AND d.modality = s.modality AND d.path = s.path\n"
              "    AND d.suid = COALESCE(s.series_uid, '')\n"
              ");");
    m_db.exec("DELETE FROM studies WHERE id NOT IN (\n"
              "  SELECT MIN(id) FROM studies GROUP BY patient_key, modality, path, COALESCE(series_uid, '')\n"
              ");");

    // Clean up old RTDOSE studies that use SeriesInstanceUID instead of FrameOfReferenceUID
    // This fixes databases created before the frame-of-reference grouping was implemented
    std::clog << "[SmartScanner] Cleaning up old RTDOSE records with incorrect series_uid..." << std::endl;
    m_db.exec("DELETE FROM studies WHERE modality='RTDOSE' "
              "AND series_uid IS NOT NULL AND frame_uid IS NOT NULL "
              "AND series_uid != frame_uid;");

    m_db.commit();
    const std::string patientsDir = m_db.dataRoot() + "/Patients";
    if (!fs::exists(patientsDir)) return true;

    for (const auto& entry : fs::directory_iterator(patientsDir)) {
        if (!entry.is_directory()) continue;
        const string patientFolder = entry.path().filename().string();
        const string infoPath = (entry.path() / "patient_info.txt").string();

        // Extract patient name and key
        string patientKey = patientFolder; // expected format: Name_ID
        string patientName = patientFolder;
        auto pos = patientFolder.rfind('_');
        if (pos != string::npos) patientName = patientFolder.substr(0, pos);

        if (!ensurePatient(patientKey, patientName, infoPath)) return false;

        // Images/{CT,MRI,PET,Others}/[Date_StudyName]
        const fs::path imagesDir = entry.path() / "Images";
        if (fs::exists(imagesDir)) {
            for (const auto& modEntry : fs::directory_iterator(imagesDir)) {
                if (!modEntry.is_directory()) continue;
                const string modalityFolder = modEntry.path().filename().string();
                for (const auto& studyEntry : fs::directory_iterator(modEntry.path())) {
                    if (!studyEntry.is_directory()) continue;
                    const string folderName = studyEntry.path().filename().string();
                    const string studyDate = extractDateFromFolder(folderName);
                    const string studyName = extractNameFromFolder(folderName);
                    if (!clearStudiesForPath(patientKey, studyEntry.path().string()))
                        return false;
                    std::unordered_map<std::string, int> seriesStudies;
                    int firstStudyId = 0;
                    for (auto const& f : fs::recursive_directory_iterator(studyEntry.path())) {
                        if (!f.is_regular_file()) continue;
                        std::string modalityTag, dateTag, nameTag, frameUID, seriesUID, seriesDesc;
                        if (readDicomMeta(f.path().string(), modalityTag, dateTag, nameTag, frameUID, seriesUID, seriesDesc)) {
                            std::string normalized = normalizeModality(!modalityTag.empty() ? modalityTag : modalityFolder);
                            std::string useDate = !dateTag.empty() ? dateTag : studyDate;
                            std::string useName = !nameTag.empty() ? nameTag : studyName;
                            std::string storedSeriesUid;
                            if (normalized == "RTDOSE") {
                                // For RTDOSE, use frameUID to group doses by frame of reference
                                storedSeriesUid = !frameUID.empty() ? frameUID : seriesUID;
                            } else {
                                storedSeriesUid = !seriesUID.empty() ? seriesUID : frameUID;
                            }
                            if (storedSeriesUid.empty()) storedSeriesUid = f.path().filename().string();
                            std::string key = normalized + "|" + storedSeriesUid;
                            int sid = 0;
                            auto it = seriesStudies.find(key);
                            if (it == seriesStudies.end()) {
                                if (!upsertStudy(patientKey, normalized, useDate, useName,
                                                 studyEntry.path().string(), frameUID, storedSeriesUid,
                                                 seriesDesc, sid))
                                    return false;
                                seriesStudies.emplace(key, sid);
                                if (firstStudyId == 0) firstStudyId = sid;
                            } else {
                                sid = it->second;
                            }
                            if (!upsertFile(sid, studyEntry.path().string(), f.path().string())) return false;
                        } else if (firstStudyId > 0) {
                            if (!upsertFile(firstStudyId, studyEntry.path().string(), f.path().string())) return false;
                        }
                    }
                }
            }
        }

        // RT_Data/{Structures,Plans,Doses,Analysis}/[Date_Name]
        const fs::path rtDir = entry.path() / "RT_Data";
        if (fs::exists(rtDir)) {
            for (const auto& catEntry : fs::directory_iterator(rtDir)) {
                if (!catEntry.is_directory()) continue;
                string modality = catEntry.path().filename().string();
                // Map category to RT modality for matching
                if (modality == "Structures") modality = "RTSTRUCT";
                else if (modality == "Plans") modality = "RTPLAN";
                else if (modality == "Doses") modality = "RTDOSE";
                else if (modality == "Analysis") modality = "RTANALYSIS";
                for (const auto& studyEntry : fs::directory_iterator(catEntry.path())) {
                    if (!studyEntry.is_directory()) continue;
                    const string folderName = studyEntry.path().filename().string();
                    const string studyDate = extractDateFromFolder(folderName);
                    const string studyName = extractNameFromFolder(folderName);
                    if (!clearStudiesForPath(patientKey, studyEntry.path().string()))
                        return false;
                    std::unordered_map<std::string, int> seriesStudies;
                    int firstStudyId = 0;
                    for (auto const& f : fs::recursive_directory_iterator(studyEntry.path())) {
                        if (!f.is_regular_file()) continue;
                        std::string modalityTag, dateTag, nameTag, frameUID, seriesUID, seriesDesc;
                        if (readDicomMeta(f.path().string(), modalityTag, dateTag, nameTag, frameUID, seriesUID, seriesDesc)) {
                            std::string normalized = normalizeModality(!modalityTag.empty() ? modalityTag : modality);
                            std::string useDate = !dateTag.empty() ? dateTag : studyDate;
                            std::string useName = !nameTag.empty() ? nameTag : studyName;
                            std::string storedSeriesUid;
                            if (normalized == "RTDOSE") {
                                // For RTDOSE, use frameUID to group doses by frame of reference
                                storedSeriesUid = !frameUID.empty() ? frameUID : seriesUID;
                            } else {
                                storedSeriesUid = !seriesUID.empty() ? seriesUID : frameUID;
                            }
                            if (storedSeriesUid.empty()) storedSeriesUid = f.path().filename().string();
                            std::string key = normalized + "|" + storedSeriesUid;
                            int sid = 0;
                            auto it = seriesStudies.find(key);
                            if (it == seriesStudies.end()) {
                                if (!upsertStudy(patientKey, normalized, useDate, useName,
                                                 studyEntry.path().string(), frameUID, storedSeriesUid,
                                                 seriesDesc, sid))
                                    return false;
                                seriesStudies.emplace(key, sid);
                                if (normalized.rfind("RT", 0) == 0) {
                                    std::clog << "[SmartScanner] Detected " << normalized
                                              << " series UID=" << (seriesUID.empty() ? storedSeriesUid : seriesUID)
                                              << " desc=" << seriesDesc
                                              << " path=" << studyEntry.path().string() << std::endl;
                                }
                                if (firstStudyId == 0) firstStudyId = sid;
                            } else {
                                sid = it->second;
                            }
                            if (!upsertFile(sid, studyEntry.path().string(), f.path().string())) return false;
                        } else if (firstStudyId > 0) {
                            if (!upsertFile(firstStudyId, studyEntry.path().string(), f.path().string())) return false;
                        }
                    }
                }
            }
        }

        // AI_Results/{Segmentation,Analysis}/[Date_ModelName]
        const fs::path aiDir = entry.path() / "AI_Results";
        if (fs::exists(aiDir)) {
            for (const auto& catEntry : fs::directory_iterator(aiDir)) {
                if (!catEntry.is_directory()) continue;
                const string modality = catEntry.path().filename().string();
                for (const auto& studyEntry : fs::directory_iterator(catEntry.path())) {
                    if (!studyEntry.is_directory()) continue;
                    const string folderName = studyEntry.path().filename().string();
                    const string studyDate = extractDateFromFolder(folderName);
                    const string studyName = extractNameFromFolder(folderName);
                    int studyId = 0;
                    std::string frameUID; // AI results not necessarily DICOM
                    std::string seriesUid;
                    std::string seriesDesc;
                    if (!upsertStudy(patientKey, modality, studyDate, studyName,
                                     studyEntry.path().string(), frameUID, seriesUid,
                                     seriesDesc, studyId))
                        return false;
                    for (auto const& f : fs::recursive_directory_iterator(studyEntry.path())) {
                        if (!f.is_regular_file()) continue;
                        if (!upsertFile(studyId, studyEntry.path().string(), f.path().string())) return false;
                    }
                }
            }
        }

        // RTDOSE_Calculated - ShioRIS3 calculated doses stored at patient root level
        const fs::path rtdoseCalcDir = entry.path() / "RTDOSE_Calculated";
        if (fs::exists(rtdoseCalcDir) && fs::is_directory(rtdoseCalcDir)) {
            // Clear existing studies for this path before scanning
            if (!clearStudiesForPath(patientKey, rtdoseCalcDir.string()))
                return false;

            std::unordered_map<std::string, int> seriesStudies;
            int firstStudyId = 0;

            for (auto const& f : fs::recursive_directory_iterator(rtdoseCalcDir)) {
                if (!f.is_regular_file()) continue;
                std::string modalityTag, dateTag, nameTag, frameUID, seriesUID, seriesDesc;
                if (readDicomMeta(f.path().string(), modalityTag, dateTag, nameTag, frameUID, seriesUID, seriesDesc)) {
                    std::string normalized = normalizeModality(modalityTag);
                    if (normalized != "RTDOSE") continue; // Only process RTDOSE files

                    std::string useDate = dateTag;
                    std::string useName = "ShioRIS3 Calculated Dose";

                    // For RTDOSE, use frameUID to group doses by frame of reference
                    std::string storedSeriesUid = !frameUID.empty() ? frameUID : seriesUID;
                    if (storedSeriesUid.empty()) storedSeriesUid = f.path().filename().string();

                    std::string key = normalized + "|" + storedSeriesUid;
                    int sid = 0;
                    auto it = seriesStudies.find(key);
                    if (it == seriesStudies.end()) {
                        if (!upsertStudy(patientKey, normalized, useDate, useName,
                                         rtdoseCalcDir.string(), frameUID, storedSeriesUid,
                                         seriesDesc, sid))
                            return false;
                        seriesStudies.emplace(key, sid);
                        if (firstStudyId == 0) firstStudyId = sid;
                        std::clog << "[SmartScanner] Detected ShioRIS3 Calculated " << normalized
                                  << " frame_uid=" << frameUID
                                  << " series_uid=" << seriesUID
                                  << " path=" << rtdoseCalcDir.string() << std::endl;
                    } else {
                        sid = it->second;
                    }
                    if (!upsertFile(sid, rtdoseCalcDir.string(), f.path().string())) return false;
                } else if (firstStudyId > 0) {
                    // Non-DICOM file: associate with first study
                    if (!upsertFile(firstStudyId, rtdoseCalcDir.string(), f.path().string())) return false;
                }
            }
        }
    }

    // Clean up orphaned file records (files that exist in DB but not on filesystem)
    std::vector<int> orphanedFileIds;
    std::stringstream fileQuery;
    fileQuery << "SELECT f.id, s.path, f.relative_path FROM files f JOIN studies s ON f.study_id = s.id;";
    if (!m_db.query(fileQuery.str(), [&](int argc, char** argv, char**) {
        if (argc >= 3 && argv[0] && argv[1] && argv[2]) {
            int fileId = std::stoi(argv[0]);
            std::string studyPath = argv[1];
            std::string relativePath = argv[2];
            std::string absolutePath = studyPath + "/" + relativePath;

            std::error_code ec;
            if (!fs::exists(absolutePath, ec) || ec) {
                orphanedFileIds.push_back(fileId);
            }
        }
    })) {
        return false;
    }

    // Delete orphaned file records
    if (!orphanedFileIds.empty()) {
        m_db.beginTransaction();
        for (int fileId : orphanedFileIds) {
            std::stringstream delFile;
            delFile << "DELETE FROM files WHERE id=" << fileId << ";";
            if (!m_db.exec(delFile.str())) {
                m_db.rollback();
                return false;
            }
        }

        // Delete studies that no longer have any files
        if (!m_db.exec("DELETE FROM studies WHERE id NOT IN (SELECT DISTINCT study_id FROM files);")) {
            m_db.rollback();
            return false;
        }

        m_db.commit();
        std::clog << "[SmartScanner] Removed " << orphanedFileIds.size()
                  << " orphaned file records during fullScanAndRepair" << std::endl;
    }

    return true;
}

bool SmartScanner::scanPath(const std::string& absPath) {
    if (!m_db.isOpen()) return false;
    fs::path p(absPath);

    // Detect patientKey from path: .../Patients/Name_ID/...
    string patientKey;
    for (auto cur = p; !cur.empty(); cur = cur.parent_path()) {
        if (cur.filename() == "Patients") {
            fs::path child = fs::path(absPath).lexically_relative(cur);
            if (!child.empty()) patientKey = child.begin()->string();
            break;
        }
    }
    if (patientKey.empty()) return false;

    std::vector<std::string> patientStudyPaths;
    {
        std::stringstream q;
        q << "SELECT path FROM studies WHERE patient_key='" << escapeSql(patientKey) << "';";
        if (!m_db.query(q.str(), [&](int argc, char** argv, char**){
                if (argc >= 1 && argv[0]) patientStudyPaths.emplace_back(argv[0]);
            }))
            return false;
    }

    auto trimSeparators = [](std::string value) {
        while (!value.empty() && (value.back() == '/' || value.back() == '\\')) value.pop_back();
        return value;
    };
    auto isSameOrSubPath = [](const std::string& base, const std::string& candidate) {
        if (base.empty()) return false;
        if (candidate.size() < base.size()) return false;
        if (candidate.compare(0, base.size(), base) != 0) return false;
        if (candidate.size() == base.size()) return true;
        char next = candidate[base.size()];
        return next == '/' || next == '\\';
    };
    std::unordered_set<std::string> removedPaths;
    auto rememberRemoval = [&](const std::string& path) {
        std::string normalized = trimSeparators(path);
        if (normalized.empty()) normalized = path;
        if (!normalized.empty()) removedPaths.insert(normalized);
    };
    auto findStudyPathFor = [&](const std::string& target) -> std::string {
        std::string normalizedTarget = trimSeparators(target);
        if (normalizedTarget.empty()) normalizedTarget = target;
        std::string bestMatch;
        size_t bestLen = 0;
        for (const auto& candidate : patientStudyPaths) {
            std::string normalizedCandidate = trimSeparators(candidate);
            if (normalizedCandidate.empty()) normalizedCandidate = candidate;
            if (normalizedCandidate.empty()) continue;
            if (normalizedTarget.size() < normalizedCandidate.size()) continue;
            if (normalizedTarget.compare(0, normalizedCandidate.size(), normalizedCandidate) != 0) continue;
            if (normalizedTarget.size() > normalizedCandidate.size()) {
                char next = normalizedTarget[normalizedCandidate.size()];
                if (next != '/' && next != '\\') continue;
            }
            if (normalizedCandidate.size() > bestLen) {
                bestLen = normalizedCandidate.size();
                bestMatch = candidate;
            }
        }
        return bestMatch;
    };
    auto clearMissingUnder = [&](const fs::path& base) -> bool {
        std::string baseNormalized = trimSeparators(base.string());
        if (baseNormalized.empty()) return true;
        std::unordered_set<std::string> seen;
        for (const auto& dbPath : patientStudyPaths) {
            std::string normalized = trimSeparators(dbPath);
            if (normalized.empty()) normalized = dbPath;
            if (normalized.empty()) continue;
            if (!isSameOrSubPath(baseNormalized, normalized)) continue;
            if (removedPaths.count(normalized)) continue;
            std::error_code ec;
            if (!fs::exists(fs::path(dbPath), ec) || ec) {
                if (seen.insert(normalized).second) {
                    if (!clearStudiesForPath(patientKey, dbPath)) return false;
                    rememberRemoval(dbPath);
                }
            }
        }
        return true;
    };

    std::string normalizedPath = trimSeparators(p.string());
    if (normalizedPath.empty()) normalizedPath = p.string();

    std::error_code existsEc;
    bool pathExists = fs::exists(p, existsEc);
    if (existsEc) pathExists = false;

    if (!pathExists) {
        std::string studyPath = findStudyPathFor(normalizedPath);
        std::string removalTarget = !studyPath.empty() ? studyPath : normalizedPath;
        if (!removalTarget.empty() && !removedPaths.count(trimSeparators(removalTarget))) {
            if (!clearStudiesForPath(patientKey, removalTarget)) return false;
            rememberRemoval(removalTarget);
        }
        if (!clearMissingUnder(p)) return false;
        fs::path parent = p.parent_path();
        if (!clearMissingUnder(parent)) return false;
        return true;
    }

    // Ensure patient exists
    fs::path patientDir = fs::path(m_db.dataRoot()) / "Patients" / patientKey;
    string infoPath = (patientDir / "patient_info.txt").string();
    string patientName = patientKey;
    auto pos = patientKey.rfind('_');
    if (pos != string::npos) patientName = patientKey.substr(0, pos);
    if (!ensurePatient(patientKey, patientName, infoPath)) return false;

    fs::path studyDir = fs::is_directory(p) ? p : p.parent_path();
    string folderName = studyDir.filename().string();
    string defaultDate = extractDateFromFolder(folderName);
    string defaultName = extractNameFromFolder(folderName);

    if (fs::is_directory(p)) {
        if (!clearMissingUnder(p)) return false;
    } else {
        if (!clearMissingUnder(studyDir)) return false;
        if (!clearMissingUnder(studyDir.parent_path())) return false;
    }

    if (fs::is_regular_file(p)) {
        std::string modality, dateTag, nameTag, frameUID, seriesUID, seriesDesc;
        if (!readDicomMeta(p.string(), modality, dateTag, nameTag, frameUID, seriesUID, seriesDesc)) return false;
        modality = normalizeModality(modality);
        if (!dateTag.empty()) defaultDate = dateTag;
        if (!nameTag.empty()) defaultName = nameTag;
        std::string storedSeriesUid;
        if (modality == "RTDOSE") {
            // For RTDOSE, use frameUID to group doses by frame of reference
            storedSeriesUid = !frameUID.empty() ? frameUID : seriesUID;
        } else {
            storedSeriesUid = !seriesUID.empty() ? seriesUID : frameUID;
        }
        if (storedSeriesUid.empty()) storedSeriesUid = p.filename().string();
        int studyId = 0;
        if (!upsertStudy(patientKey, modality, defaultDate, defaultName, studyDir.string(), frameUID, storedSeriesUid, seriesDesc, studyId))
            return false;
        return upsertFile(studyId, studyDir.string(), p.string());
    }

    // Directory: may contain multiple modalities
    std::unordered_map<std::string, int> studyIds;
    int firstStudyId = 0;
    for (auto const& f : fs::recursive_directory_iterator(p)) {
        if (!f.is_regular_file()) continue;
        std::string modality, dateTag, nameTag, frameUID, seriesUID, seriesDesc;
        if (readDicomMeta(f.path().string(), modality, dateTag, nameTag, frameUID, seriesUID, seriesDesc)) {
            modality = normalizeModality(modality);
            std::string useDate = !dateTag.empty() ? dateTag : defaultDate;
            std::string useName = !nameTag.empty() ? nameTag : defaultName;
            std::string storedSeriesUid;
            if (modality == "RTDOSE") {
                // For RTDOSE, use frameUID to group doses by frame of reference
                storedSeriesUid = !frameUID.empty() ? frameUID : seriesUID;
            } else {
                storedSeriesUid = !seriesUID.empty() ? seriesUID : frameUID;
            }
            if (storedSeriesUid.empty()) storedSeriesUid = f.path().filename().string();
            std::string key = modality + "|" + storedSeriesUid;
            int sid = 0;
            auto it = studyIds.find(key);
            if (it == studyIds.end()) {
                if (!upsertStudy(patientKey, modality, useDate, useName, p.string(), frameUID, storedSeriesUid, seriesDesc, sid))
                    return false;
                studyIds[key] = sid;
                if (firstStudyId == 0) firstStudyId = sid;
                if (modality.rfind("RT", 0) == 0) {
                    std::clog << "[SmartScanner] Detected " << modality
                              << " series UID=" << (seriesUID.empty() ? storedSeriesUid : seriesUID)
                              << " desc=" << seriesDesc
                              << " path=" << p.string() << std::endl;
                }
            } else {
                sid = it->second;
            }
            if (!upsertFile(sid, p.string(), f.path().string())) return false;
        } else if (firstStudyId > 0) {
            // Non-DICOM file: associate with first study
            if (!upsertFile(firstStudyId, p.string(), f.path().string())) return false;
        }
    }
    return !studyIds.empty();
}

bool SmartScanner::ensurePatient(const std::string& patientKey,
                                 const std::string& name,
                                 const std::string& infoPath) {
    std::stringstream ss;
    ss << "INSERT INTO patients(patient_key, name, created_at, info_path)\n"
       << "VALUES('" << escapeSql(patientKey) << "','" << escapeSql(name)
       << "', strftime('%s','now'), '" << escapeSql(infoPath) << "')\n"
       << "ON CONFLICT(patient_key) DO UPDATE SET name=excluded.name, info_path=excluded.info_path;";
    return m_db.exec(ss.str());
}

bool SmartScanner::upsertStudy(const std::string& patientKey,
                               const std::string& modality,
                               const std::string& studyDate,
                               const std::string& studyName,
                               const std::string& absPath,
                               const std::string& frameUID,
                               const std::string& seriesUID,
                               const std::string& seriesDescription,
                               int& outStudyId) {
    std::string safeSeriesUID = seriesUID;
    if (safeSeriesUID.empty()) safeSeriesUID.clear();

    std::stringstream ss;
    ss << "INSERT INTO studies(patient_key, modality, study_date, study_name, path, frame_uid, series_uid, series_description)\n"
       << "VALUES('" << escapeSql(patientKey) << "','" << escapeSql(modality) << "','" << escapeSql(studyDate)
       << "','" << escapeSql(studyName) << "','" << escapeSql(absPath) << "','" << escapeSql(frameUID) << "','"
       << escapeSql(safeSeriesUID) << "','" << escapeSql(seriesDescription) << "')\n"
       << "ON CONFLICT(patient_key, modality, path, series_uid) DO UPDATE SET\n"
       << "  study_date=excluded.study_date,\n"
       << "  study_name=CASE WHEN studies.study_name='ShioRIS3 Calculated Dose' THEN studies.study_name ELSE excluded.study_name END,\n"
       << "  frame_uid=excluded.frame_uid,\n"
       << "  series_description=excluded.series_description;";
    if (!m_db.exec(ss.str())) return false;

    int foundId = 0;
    std::stringstream q;
    q << "SELECT id FROM studies WHERE patient_key='" << escapeSql(patientKey) << "' AND modality='"
      << escapeSql(modality) << "' AND path='" << escapeSql(absPath) << "' AND series_uid='"
      << escapeSql(safeSeriesUID) << "' LIMIT 1;";
    bool ok = m_db.query(q.str(), [&](int argc, char** argv, char**){
        if (argc >= 1 && argv[0]) foundId = std::stoi(argv[0]);
    });
    if (!ok || foundId == 0) return false;
    outStudyId = foundId;
    return true;
}

bool SmartScanner::upsertFile(int studyId,
                              const std::string& baseDir,
                              const std::string& filePath) {
    std::error_code ec;
    auto sz = fs::file_size(filePath, ec);
    if (ec) sz = 0;
    auto ft = fs::last_write_time(filePath, ec);
    long long mtime = ec ? 0 : toUnixTime(ft);
    std::string rel = toRelative(baseDir, filePath);

    std::stringstream ss;
    ss << "INSERT INTO files(study_id, relative_path, size_bytes, mtime, file_type)\n"
       << "VALUES(" << studyId << ", '" << escapeSql(rel) << "', " << sz << ", " << mtime << ", '')\n"
       << "ON CONFLICT(study_id, relative_path) DO UPDATE SET size_bytes=excluded.size_bytes, mtime=excluded.mtime;";
    return m_db.exec(ss.str());
}

bool SmartScanner::clearStudiesForPath(const std::string& patientKey,
                                       const std::string& absPath) {
    std::stringstream ss;
    ss << "DELETE FROM studies WHERE patient_key='" << escapeSql(patientKey)
       << "' AND path='" << escapeSql(absPath) << "';";
    return m_db.exec(ss.str());
}

std::string SmartScanner::extractDateFromFolder(const std::string& folderName) const {
    // Expect: YYYYMMDD_something; otherwise empty
    if (folderName.size() >= 8) {
        bool digits = true;
        for (int i = 0; i < 8; ++i) if (!isdigit(folderName[i])) { digits = false; break; }
        if (digits) return folderName.substr(0, 8);
    }
    return "";
}

std::string SmartScanner::extractNameFromFolder(const std::string& folderName) const {
    auto pos = folderName.find('_');
    if (pos == std::string::npos) return folderName;
    return folderName.substr(pos + 1);
}

std::string SmartScanner::toRelative(const std::string& base, const std::string& abs) const {
    fs::path rel = fs::path(abs).lexically_relative(base);
    return rel.string();
}
