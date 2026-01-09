#ifndef DOSE_ISOSURFACE_H
#define DOSE_ISOSURFACE_H

#include <QVector3D>
#include <QVector>
#include <QColor>
#include <opencv2/core.hpp>

class DicomVolume; // Forward declaration

// 3Dメッシュの三角形
struct DoseTriangle {
    QVector3D vertices[3];  // 3頂点（患者座標系、mm単位）
    QVector3D normal;       // 法線ベクトル
};

// 等線量サーフェスメッシュ
class DoseIsosurface {
public:
    DoseIsosurface();

    // 指定した線量レベル（Gy）でアイソサーフェスを生成
    // Marching Cubesアルゴリズムを使用
    void generateIsosurface(const cv::Mat& doseVolume,
                           double isoValue,  // Gy単位
                           double originX, double originY, double originZ,
                           double spacingX, double spacingY, double spacingZ,
                           const double rowDir[3],
                           const double colDir[3],
                           const double sliceDir[3]);

    // 生成されたメッシュを取得
    const QVector<DoseTriangle>& triangles() const { return m_triangles; }

    // メッシュの色と透明度を設定
    void setColor(const QColor& color) { m_color = color; }
    QColor color() const { return m_color; }

    void setOpacity(float opacity) { m_opacity = opacity; }
    float opacity() const { return m_opacity; }

    // メッシュのクリア
    void clear() { m_triangles.clear(); }

    // メッシュが空かどうか
    bool isEmpty() const { return m_triangles.isEmpty(); }

    // 統計情報
    int triangleCount() const { return m_triangles.size(); }

    // バウンディングボックスの中心点を取得（ソート用）
    QVector3D center() const;

    // 患者座標系から3Dウィジェット座標系（画像中心が原点）に変換
    void transformTo3DWidgetSpace(const DicomVolume& refVolume);

private:
    // Marching Cubesルックアップテーブル
    static const int edgeTable[256];
    static const int triTable[256][16];

    // ボクセル座標から患者座標への変換
    QVector3D voxelToPatient(double x, double y, double z) const;

    // エッジ上の補間点を計算
    QVector3D interpolateVertex(const QVector3D& p1, const QVector3D& p2,
                               float val1, float val2, float isoValue) const;

    // 三角形の法線を計算
    QVector3D calculateNormal(const QVector3D& v0, const QVector3D& v1, const QVector3D& v2) const;

    // ボリュームデータへの安全なアクセス
    float getVoxelValue(const cv::Mat& volume, int x, int y, int z) const;

    QVector<DoseTriangle> m_triangles;
    QColor m_color;
    float m_opacity;

    // ジオメトリ情報（最後の生成時の情報を保持）
    double m_originX, m_originY, m_originZ;
    double m_spacingX, m_spacingY, m_spacingZ;
    double m_rowDir[3], m_colDir[3], m_sliceDir[3];
    int m_volumeWidth, m_volumeHeight, m_volumeDepth;
};

#endif // DOSE_ISOSURFACE_H
