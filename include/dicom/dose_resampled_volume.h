#ifndef DOSE_RESAMPLED_VOLUME_H
#define DOSE_RESAMPLED_VOLUME_H

#include "dicom/dicom_volume.h"
#include "dicom/rtdose_volume.h"
#include <opencv2/core.hpp>
#include <QColor>
#include <QImage>
#include <QPointF>
#include <QVector>
#include <QVector3D>
#include <functional>

class DoseResampledVolume
{
public:
    // 線量表示モード
    enum class DoseDisplayMode {
        Simple,     // 従来の赤色表示
        Colorful,   // カラフルなHSVグラデーション
        Isodose,    // 等線量曲線風表示（塗りつぶし）
        IsodoseLines, // 等線量線
        Hot         // 黒→赤→黄→白のグラデーション
    };

    struct IsodoseLinePath {
        QVector<QPointF> points;
        QColor color;
    };
    using IsodoseLineList = QVector<IsodoseLinePath>;

private:
    cv::Mat m_resampledVolume; // CV_32F, depth x height x width
    int m_width{0};
    int m_height{0};
    int m_depth{0};
    double m_spacingX{1.0};
    double m_spacingY{1.0};
    double m_spacingZ{1.0};
    double m_originX{0.0};
    double m_originY{0.0};
    double m_originZ{0.0};
    double m_maxDose{0.0};
    bool m_isResampled{false};

public:
    DoseResampledVolume();
    ~DoseResampledVolume();

    bool resampleFromRTDose(const DicomVolume& ctVol,
                            const RTDoseVolume& rtDose,
                            std::function<void(int, int)> progressCallback = {},
                            bool useNativeDoseGeometry = true);
    
    // デフォルトはカラフル表示
    QImage getSlice(int index, DicomVolume::Orientation ori,
                    double minDose, double maxDose,
                    DoseDisplayMode mode = DoseDisplayMode::Colorful,
                    double referenceDose = 1.0,
                    std::function<double(double)> transform = {}) const;

    IsodoseLineList getIsodoseLines(int index, DicomVolume::Orientation ori,
                                    double minDose, double maxDose,
                                    double referenceDose = 1.0,
                                    std::function<double(double)> transform = {}) const;

    size_t getMemoryUsage() const;
    bool isResampled() const { return m_isResampled; }
    void clear();

    // 最大線量値を取得
    double maxDose() const { return m_maxDose; }

    // ジオメトリ情報へのアクセサ
    int width() const { return m_width; }
    int height() const { return m_height; }
    int depth() const { return m_depth; }
    double spacingX() const { return m_spacingX; }
    double spacingY() const { return m_spacingY; }
    double spacingZ() const { return m_spacingZ; }
    double originX() const { return m_originX; }
    double originY() const { return m_originY; }
    double originZ() const { return m_originZ; }

    // 内部データへのアクセス
    const cv::Mat &data() const { return m_resampledVolume; }
    cv::Mat &data() { return m_resampledVolume; }

    // 外部データからの設定
    void setFromMat(const cv::Mat &vol, double spacingX, double spacingY,
                    double spacingZ, double originX, double originY,
                    double originZ);
    void updateMaxDose();

    // 指定ボクセルの線量値を取得 (Gy)
    float voxelDose(int x, int y, int z) const;
};

#endif // DOSE_RESAMPLED_VOLUME_H
