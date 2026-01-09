#ifndef STRUCTURE_SURFACE_H
#define STRUCTURE_SURFACE_H

#include <QVector3D>
#include <QVector>
#include <QColor>
#include <opencv2/core.hpp>

class DicomVolume; // Forward declaration

// 3Dメッシュの三角形
struct StructureTriangle {
    QVector3D vertices[3];  // 3頂点（患者座標系、mm単位）
    QVector3D normal;       // 法線ベクトル
};

// Structureサーフェスメッシュ
class StructureSurface {
public:
    StructureSurface();

    // 輪郭データからアイソサーフェスを生成
    // Marching Cubesアルゴリズムを使用
    void generateFromContours(const QVector<QVector<QVector3D>>& contours,
                             const DicomVolume& refVolume);

    // 生成されたメッシュを取得
    const QVector<StructureTriangle>& triangles() const { return m_triangles; }

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

    // 輪郭からボリュームを生成（ボクセル化）
    cv::Mat voxelizeContours(const QVector<QVector<QVector3D>>& contours,
                            const DicomVolume& refVolume,
                            QVector3D& outOrigin,
                            QVector3D& outSpacing,
                            int& outWidth, int& outHeight, int& outDepth);

    // ボクセル座標から患者座標への変換
    QVector3D voxelToPatient(double x, double y, double z,
                            const QVector3D& origin,
                            const QVector3D& spacing) const;

    // エッジ上の補間点を計算
    QVector3D interpolateVertex(const QVector3D& p1, const QVector3D& p2,
                               float val1, float val2, float isoValue) const;

    // 三角形の法線を計算
    QVector3D calculateNormal(const QVector3D& v0, const QVector3D& v1, const QVector3D& v2) const;

    // ボリュームデータへの安全なアクセス
    float getVoxelValue(const cv::Mat& volume, int x, int y, int z) const;

    // Marching Cubesでサーフェスを生成
    void marchingCubes(const cv::Mat& volume, float isoValue,
                      const QVector3D& origin, const QVector3D& spacing);

    QVector<StructureTriangle> m_triangles;
    QColor m_color;
    float m_opacity;
};

#endif // STRUCTURE_SURFACE_H
