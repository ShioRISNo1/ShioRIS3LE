#ifndef DPSD_CALCULATOR_H
#define DPSD_CALCULATOR_H

#include "dicom/dicom_volume.h"
#include "dicom/dose_resampled_volume.h"
#include "dicom/rtstruct.h"
#include "dicom/rtdose_volume.h"
#include <atomic>
#include <vector>

class DPSDCalculator {
public:
  struct Result {
    std::vector<double> distancesMm;
    std::vector<double> minDoseGy;
    std::vector<double> maxDoseGy;
    std::vector<double> meanDoseGy;
  };

  enum class Mode { Mode2D, Mode3D };

  static Result calculate(const DicomVolume &ctVolume,
                          const DoseResampledVolume &doseVolume,
                          const RTStructureSet &structures, int roiIndex,
                          double startDistanceMm = -20.0,
                          double endDistanceMm = 50.0, double stepMm = 2.0,
                          Mode mode = Mode::Mode3D,
                          int sampleRoiIndex = -1,
                          std::atomic_bool *cancel = nullptr);

  // 直接RTDoseからサンプリングする経路（検証用/代替）
  static Result calculateFromRTDose(const DicomVolume &ctVolume,
                                    const RTDoseVolume &rtDose,
                                    const RTStructureSet &structures,
                                    int roiIndex,
                                    double startDistanceMm = -20.0,
                                    double endDistanceMm = 50.0,
                                    double stepMm = 2.0,
                                    Mode mode = Mode::Mode3D,
                                    int sampleRoiIndex = -1,
                                    std::atomic_bool *cancel = nullptr);
};

#endif // DPSD_CALCULATOR_H
