#ifndef VOLUME_VIEWER_H
#define VOLUME_VIEWER_H

#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QSpinBox>
#include <QPushButton>

#include "dicom/dicom_volume.h"

class VolumeViewer : public QWidget
{
    Q_OBJECT
public:
    explicit VolumeViewer(QWidget* parent = nullptr);

    bool loadVolume(const QString& directory);

private slots:
    void onWindowChanged(int value);
    void onLevelChanged(int value);
    void onSliceChanged(int value);

private:
    void updateImages();

    DicomVolume m_volume;
    double m_window;
    double m_level;

    QLabel* m_axLabel;
    QLabel* m_sagLabel;
    QLabel* m_corLabel;
    QSlider* m_sliceSlider;
    QSlider* m_windowSlider;
    QSlider* m_levelSlider;
};

#endif // VOLUME_VIEWER_H
