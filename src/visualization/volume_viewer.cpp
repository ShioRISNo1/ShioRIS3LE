#include "visualization/volume_viewer.h"
#include <QGridLayout>
#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <QPainter>

VolumeViewer::VolumeViewer(QWidget* parent)
    : QWidget(parent)
    , m_window(256)
    , m_level(128)
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    QHBoxLayout* imageLayout = new QHBoxLayout();
    m_axLabel = new QLabel("Axial");
    m_sagLabel = new QLabel("Sagittal");
    m_corLabel = new QLabel("Coronal");
    m_axLabel->setAlignment(Qt::AlignCenter);
    m_sagLabel->setAlignment(Qt::AlignCenter);
    m_corLabel->setAlignment(Qt::AlignCenter);
    imageLayout->addWidget(m_axLabel);
    imageLayout->addWidget(m_sagLabel);
    imageLayout->addWidget(m_corLabel);

    QHBoxLayout* controlLayout = new QHBoxLayout();
    m_sliceSlider = new QSlider(Qt::Horizontal);
    m_windowSlider = new QSlider(Qt::Horizontal);
    m_levelSlider = new QSlider(Qt::Horizontal);
    m_sliceSlider->setRange(0,0);
    m_windowSlider->setRange(1,4096);
    m_levelSlider->setRange(-1024,3072);
    m_windowSlider->setValue(256);
    m_levelSlider->setValue(128);
    controlLayout->addWidget(new QLabel("Slice"));
    controlLayout->addWidget(m_sliceSlider);
    controlLayout->addWidget(new QLabel("W"));
    controlLayout->addWidget(m_windowSlider);
    controlLayout->addWidget(new QLabel("L"));
    controlLayout->addWidget(m_levelSlider);

    mainLayout->addLayout(imageLayout,1);
    mainLayout->addLayout(controlLayout);

    connect(m_sliceSlider,&QSlider::valueChanged,this,&VolumeViewer::onSliceChanged);
    connect(m_windowSlider,&QSlider::valueChanged,this,&VolumeViewer::onWindowChanged);
    connect(m_levelSlider,&QSlider::valueChanged,this,&VolumeViewer::onLevelChanged);
}

bool VolumeViewer::loadVolume(const QString& directory)
{
    if (!m_volume.loadFromDirectory(directory)) return false;
    m_sliceSlider->setRange(0, m_volume.depth()-1);
    int mid = m_volume.depth() > 0 ? m_volume.depth() / 2 : 0;
    m_sliceSlider->setValue(mid);
    updateImages();
    return true;
}

void VolumeViewer::onWindowChanged(int value)
{
    m_window = value;
    updateImages();
}

void VolumeViewer::onLevelChanged(int value)
{
    m_level = value;
    updateImages();
}

void VolumeViewer::onSliceChanged(int value)
{
    Q_UNUSED(value);
    updateImages();
}

void VolumeViewer::updateImages()
{
    int index = m_sliceSlider->value();
    QImage ax = m_volume.getSlice(index, DicomVolume::Orientation::Axial, m_window, m_level);
    QImage sag = m_volume.getSlice(index, DicomVolume::Orientation::Sagittal, m_window, m_level);
    QImage cor = m_volume.getSlice(index, DicomVolume::Orientation::Coronal, m_window, m_level);

    auto drawSliceLines = [](QImage &img, int countY) {
        if (img.isNull() || countY <= 0) return;
        QPainter p(&img);
        p.setPen(QPen(Qt::gray));
        for (int y = 0; y < countY; ++y) {
            int cy = static_cast<int>((y + 0.5) * img.height() / countY);
            p.drawLine(0, cy, img.width(), cy);
        }
    };

    drawSliceLines(ax, m_volume.height());
    drawSliceLines(sag, m_volume.depth());
    drawSliceLines(cor, m_volume.depth());

    m_axLabel->setPixmap(QPixmap::fromImage(ax));
    m_sagLabel->setPixmap(QPixmap::fromImage(sag));
    m_corLabel->setPixmap(QPixmap::fromImage(cor));
}

