#ifndef BRACHY_SOURCE_H
#define BRACHY_SOURCE_H

#include <QVector3D>

// Brachytherapy source information
class BrachySource {
public:
    BrachySource() = default;
    BrachySource(const QVector3D &pos, const QVector3D &dir,
                 double dwell, int ch)
        : m_position(pos), m_direction(dir), m_dwellTimeSec(dwell),
          m_channel(ch) {}

    const QVector3D &position() const { return m_position; }
    const QVector3D &direction() const { return m_direction; }
    double dwellTime() const { return m_dwellTimeSec; }
    int channel() const { return m_channel; }

    void setPosition(const QVector3D &p) { m_position = p; }
    void setDirection(const QVector3D &d) { m_direction = d; }
    void setDwellTime(double t) { m_dwellTimeSec = t; }
    void setChannel(int c) { m_channel = c; }

private:
    QVector3D m_position{0, 0, 0};
    QVector3D m_direction{0, 0, 0};
    double m_dwellTimeSec{0.0};
    int m_channel{0};
};

#endif // BRACHY_SOURCE_H

