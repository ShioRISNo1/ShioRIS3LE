#include "visualization/quad_viewer.h"
#include <QSizePolicy>

QuadViewer::QuadViewer(QWidget* parent)
    : QWidget(parent)
{
    m_layout = new QGridLayout(this);
    m_layout->setSpacing(10);
    m_layout->setContentsMargins(10, 10, 10, 10);

    m_layout->setRowStretch(0, 1);
    m_layout->setRowStretch(1, 1);
    m_layout->setColumnStretch(0, 1);
    m_layout->setColumnStretch(1, 1);

    for (int i = 0; i < 4; ++i) {
        m_viewers[i] = new DicomViewer(this, false);
        m_viewers[i]->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        m_layout->addWidget(m_viewers[i], i / 2, i % 2);
    }
    for (int i = 1; i < 4; ++i) {
        connect(m_viewers[0], &DicomViewer::windowLevelChanged,
                m_viewers[i], &DicomViewer::setWindowLevel);
    }
}

bool QuadViewer::loadDicomFile(const QString& filename)
{
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        ok = m_viewers[i]->loadDicomFile(filename) && ok;
    }
    return ok;
}

bool QuadViewer::loadDicomDirectory(const QString& directory)
{
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        ok = m_viewers[i]->loadDicomDirectory(directory) && ok;
    }
    return ok;
}
