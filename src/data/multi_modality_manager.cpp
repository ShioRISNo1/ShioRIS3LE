#include "data/multi_modality_manager.h"
#include <algorithm>

const std::vector<std::string>& MultiModalityManager::imageModalities() {
    static const std::vector<std::string> k = {"CT", "MRI", "PET", "Others"};
    return k;
}

const std::vector<std::string>& MultiModalityManager::rtCategories() {
    static const std::vector<std::string> k = {"Structures", "Plans", "Doses", "Analysis"};
    return k;
}

const std::vector<std::string>& MultiModalityManager::aiCategories() {
    static const std::vector<std::string> k = {"Segmentation", "Analysis"};
    return k;
}

static inline std::string toUpper(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c){ return std::toupper(c); });
    return r;
}

std::string MultiModalityManager::normalize(const std::string& value,
                                            const std::vector<std::string>& domain) {
    auto v = toUpper(value);
    for (const auto& d : domain) {
        if (toUpper(d) == v) return d;
    }
    return value;
}
