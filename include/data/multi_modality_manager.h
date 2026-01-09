#pragma once

#include <string>
#include <vector>

// MultiModalityManager: Utilities to work with modalities and categories.
class MultiModalityManager {
public:
    static const std::vector<std::string>& imageModalities();     // CT, MRI, PET, Others
    static const std::vector<std::string>& rtCategories();        // Structures, Plans, Doses, Analysis
    static const std::vector<std::string>& aiCategories();        // Segmentation, Analysis

    // Normalizes a modality/category string to canonical form (case-insensitive match).
    static std::string normalize(const std::string& value,
                                 const std::vector<std::string>& domain);
};

