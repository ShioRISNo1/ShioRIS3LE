#include "dicom/brachy_plan.h"
#include <dcmtk/dcmdata/dctk.h>
#include <QMap>
#include <QStringList>
#include <QRandomGenerator>
#include <QDebug>
#include <cmath>

bool BrachyPlan::loadFromFile(const QString &filename) {
    m_sources.clear();
    m_referencePoints.clear();
    DcmFileFormat file;
    if (file.loadFile(filename.toLocal8Bit().data()).bad()) {
        return false;
    }
    DcmDataset *ds = file.getDataset();
    DcmSequenceOfItems *appSeq = nullptr;
    // Try Brachy Application Setup Sequence (300A,0013),
    // then fallback to Application Setup Sequence (300A,0230).
    OFCondition cond = ds->findAndGetSequence(DcmTagKey(0x300A, 0x0013), appSeq);
    if (cond.bad() || !appSeq) {
        cond = ds->findAndGetSequence(DcmTagKey(0x300A, 0x0230), appSeq);
    }
    if (cond.bad() || !appSeq) {
        return false;
    }
    for (unsigned long i = 0; i < appSeq->card(); ++i) {
        DcmItem *appItem = appSeq->getItem(i);
        DcmSequenceOfItems *chanSeq = nullptr;
        if (appItem->findAndGetSequence(DCM_ChannelSequence, chanSeq).good() && chanSeq) {
            for (unsigned long j = 0; j < chanSeq->card(); ++j) {
                DcmItem *chanItem = chanSeq->getItem(j);
                Sint32 channelNum = 0;
                chanItem->findAndGetSint32(DCM_ChannelNumber, channelNum);
                double totalTime = 0.0;
                chanItem->findAndGetFloat64(DCM_ChannelTotalTime, totalTime);
                DcmSequenceOfItems *cpSeq = nullptr;
                if (chanItem->findAndGetSequence(DCM_BrachyControlPointSequence, cpSeq).good() && cpSeq) {
                    double prevWeight = 0.0;
                    QVector<BrachySource> channelSources;
                    for (unsigned long k = 0; k < cpSeq->card(); ++k) {
                        DcmItem *cpItem = cpSeq->getItem(k);
                        OFString posStr;
                        QVector3D pos;
                        if (cpItem->findAndGetOFStringArray(DCM_ControlPoint3DPosition, posStr).good()) {
                            QStringList nums =
                                QString::fromLatin1(posStr.c_str())
                                    .split(QLatin1Char('\\'), Qt::SkipEmptyParts);
                            if (nums.size() >= 3) {
                                pos.setX(nums[0].toDouble());
                                pos.setY(nums[1].toDouble());
                                pos.setZ(nums[2].toDouble());
                            }
                        }
                        double weight = 0.0;
                        cpItem->findAndGetFloat64(DCM_CumulativeTimeWeight, weight);
                        double dwell = 0.0;
                        double deltaW = 0.0;
                        if (k > 0) {
                            deltaW = weight - prevWeight;
                            if (deltaW < 0.0) deltaW = 0.0; // guard against malformed data
                            dwell = (totalTime > 0.0) ? (deltaW * totalTime) : deltaW;

                            // DEBUG: Log weight ratio and calculated dwell time
                            if (k <= 3) {
                                qDebug() << QString("  Control Point %1: cumWeight=%2, deltaW=%3, totalTime=%4 s, dwellTime=%5 s")
                                    .arg(k).arg(weight, 0, 'f', 4).arg(deltaW, 0, 'f', 4).arg(totalTime, 0, 'f', 2).arg(dwell, 0, 'f', 3);
                            }
                        }
                        BrachySource src;
                        src.setPosition(pos);
                        src.setDwellTime(dwell);
                        src.setChannel(channelNum);
                        channelSources.append(src);
                        prevWeight = weight;
                    }
                    // compute direction vectors within the channel
                    for (int k = 0; k < channelSources.size(); ++k) {
                        QVector3D dir(0, 0, 0);
                        if (k > 0)
                            dir += channelSources[k].position() - channelSources[k - 1].position();
                        if (k + 1 < channelSources.size())
                            dir += channelSources[k + 1].position() - channelSources[k].position();
                        if (dir.lengthSquared() > 0.0f)
                            channelSources[k].setDirection(dir.normalized());
                        m_sources.append(channelSources[k]);
                    }
                }
            }
        }
    }

    // Load reference points from Dose Reference Sequence (0x300A, 0x0010)
    DcmSequenceOfItems *doseRefSeq = nullptr;
    if (ds->findAndGetSequence(DcmTagKey(0x300A, 0x0010), doseRefSeq).good() && doseRefSeq) {
        for (unsigned long i = 0; i < doseRefSeq->card(); ++i) {
            DcmItem *doseRefItem = doseRefSeq->getItem(i);

            // Get reference point coordinates (0x300A, 0x0018)
            OFString posStr;
            QVector3D pos(0.0, 0.0, 0.0);
            if (doseRefItem->findAndGetOFStringArray(DcmTagKey(0x300A, 0x0018), posStr).good()) {
                QStringList nums = QString::fromLatin1(posStr.c_str())
                    .split(QLatin1Char('\\'), Qt::SkipEmptyParts);
                if (nums.size() >= 3) {
                    pos.setX(nums[0].toDouble());
                    pos.setY(nums[1].toDouble());
                    pos.setZ(nums[2].toDouble());
                }
            }

            // Get prescribed dose (0x300A, 0x0026) - Target Prescription Dose
            double dose = 0.0;
            doseRefItem->findAndGetFloat64(DcmTagKey(0x300A, 0x0026), dose);

            // Get label/description (0x300A, 0x0016) - Dose Reference Description
            OFString descStr;
            QString label;
            if (doseRefItem->findAndGetOFString(DcmTagKey(0x300A, 0x0016), descStr).good()) {
                label = QString::fromLatin1(descStr.c_str());
            }

            // Only add if we have valid coordinates or dose
            if (pos.lengthSquared() > 0.0 || dose > 0.0) {
                ReferencePoint refPoint(pos, dose, label);
                m_referencePoints.append(refPoint);
                qDebug() << "Loaded reference point:" << pos << "dose:" << dose << "Gy" << "label:" << label;
            }
        }
    }

    // DEBUG: Log loaded dwell times and their ratios per channel
    if (!m_sources.isEmpty()) {
        qDebug() << "=== Loaded Brachy Plan: Dwell Times and Ratios ===";
        QMap<int, QVector<double>> channelTimes;
        for (const auto& src : m_sources) {
            channelTimes[src.channel()].append(src.dwellTime());
        }

        for (auto it = channelTimes.cbegin(); it != channelTimes.cend(); ++it) {
            int channel = it.key();
            const auto& times = it.value();
            double totalTime = 0.0;
            for (double t : times) {
                totalTime += t;
            }

            qDebug() << QString("Channel %1 (Total: %2 s):").arg(channel).arg(totalTime, 0, 'f', 3);
            for (int i = 0; i < qMin(5, times.size()); ++i) {
                double ratio = (totalTime > 0.0) ? (times[i] / totalTime) : 0.0;
                qDebug() << QString("  Source %1: %2 s (ratio: %3)")
                    .arg(i).arg(times[i], 0, 'f', 3).arg(ratio, 0, 'f', 4);
            }
            if (times.size() > 5) {
                qDebug() << QString("  ... (%1 more sources)").arg(times.size() - 5);
            }
        }
        qDebug() << "=================================================";
    }

    return !m_sources.isEmpty();
}

QStringList BrachyPlan::dwellTimeStrings() const {
    QMap<int, QList<double>> byChannel;
    for (const auto &s : m_sources) {
        byChannel[s.channel()].append(s.dwellTime());
    }
    QStringList lines;
    for (auto it = byChannel.cbegin(); it != byChannel.cend(); ++it) {
        QStringList times;

        // Calculate total time for this channel
        double totalTime = 0.0;
        for (double t : it.value()) {
            totalTime += t;
        }

        // Display both absolute times and ratios
        for (int i = 0; i < it.value().size(); ++i) {
            double t = it.value()[i];
            double ratio = (totalTime > 0.0) ? (t / totalTime) : 0.0;
            times << QString("%1s(%2%)").arg(t, 0, 'f', 2).arg(ratio * 100.0, 0, 'f', 1);
        }
        lines << QString("Channel %1: %2 [Total: %3s]")
                    .arg(it.key())
                    .arg(times.join(", "))
                    .arg(totalTime, 0, 'f', 2);
    }
    return lines;
}

void BrachyPlan::generateRandomSources(int count, double spatialRange,
                                       double minDwellTime, double maxDwellTime) {
    m_sources.clear();
    QRandomGenerator *rng = QRandomGenerator::global();

    for (int i = 0; i < count; ++i) {
        // Random position within [-spatialRange, +spatialRange]
        double x = rng->bounded(2.0 * spatialRange) - spatialRange;
        double y = rng->bounded(2.0 * spatialRange) - spatialRange;
        double z = rng->bounded(2.0 * spatialRange) - spatialRange;
        QVector3D position(x, y, z);

        // Random direction (uniform on unit sphere)
        // Using spherical coordinates
        double theta = rng->bounded(2.0 * M_PI);  // azimuthal angle [0, 2π]
        double phi = std::acos(2.0 * rng->bounded(1.0) - 1.0);  // polar angle [0, π]
        QVector3D direction(
            std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)
        );

        // Random dwell time
        double dwellTime = minDwellTime + rng->bounded(maxDwellTime - minDwellTime);

        // Channel 1 for all random sources
        int channel = 1;

        BrachySource source(position, direction, dwellTime, channel);
        m_sources.append(source);
    }
}

void BrachyPlan::clearSources() {
    m_sources.clear();
}

void BrachyPlan::addSource(const BrachySource &source) {
    m_sources.append(source);
}

void BrachyPlan::generateTestSourceAtOrigin() {
    m_sources.clear();

    // Single source at origin with Z-axis direction
    BrachySource testSource(
        QVector3D(0, 0, 0),      // Position: origin
        QVector3D(0, 0, 1),      // Direction: Z-axis (normalized)
        10.0,                     // Dwell time: 10 seconds
        1                         // Channel: 1
    );

    m_sources.append(testSource);

    qDebug() << "Generated test source at origin:"
             << "pos=" << testSource.position()
             << "dir=" << testSource.direction()
             << "dwell=" << testSource.dwellTime() << "s";
}

void BrachyPlan::addEvaluationPoint(const DoseEvaluationPoint& point) {
    m_evaluationPoints.append(point);
}

void BrachyPlan::clearEvaluationPoints() {
    m_evaluationPoints.clear();
}

void BrachyPlan::setDwellTimes(const QVector<double>& dwellTimes) {
    if (dwellTimes.size() != m_sources.size()) {
        qWarning() << "setDwellTimes: size mismatch" << dwellTimes.size() << "vs" << m_sources.size();
        return;
    }

    // DEBUG: Log old and new ratios per channel
    qDebug() << "=== setDwellTimes: Comparing Old vs New Ratios ===";

    // Group by channel and calculate old ratios
    QMap<int, QVector<int>> channelIndices;  // channel -> source indices
    for (int i = 0; i < m_sources.size(); ++i) {
        channelIndices[m_sources[i].channel()].append(i);
    }

    for (auto it = channelIndices.cbegin(); it != channelIndices.cend(); ++it) {
        int channel = it.key();
        const auto& indices = it.value();

        // Calculate old total time and ratios
        double oldTotal = 0.0;
        for (int idx : indices) {
            oldTotal += m_sources[idx].dwellTime();
        }

        // Calculate new total time
        double newTotal = 0.0;
        for (int idx : indices) {
            newTotal += dwellTimes[idx];
        }

        qDebug() << QString("Channel %1:").arg(channel);
        qDebug() << QString("  Old total: %1 s, New total: %2 s").arg(oldTotal, 0, 'f', 3).arg(newTotal, 0, 'f', 3);

        // Log first few sources
        for (int j = 0; j < qMin(3, indices.size()); ++j) {
            int idx = indices[j];
            double oldTime = m_sources[idx].dwellTime();
            double newTime = dwellTimes[idx];
            double oldRatio = (oldTotal > 0.0) ? (oldTime / oldTotal) : 0.0;
            double newRatio = (newTotal > 0.0) ? (newTime / newTotal) : 0.0;

            qDebug() << QString("  Source %1: %2s(%3%) -> %4s(%5%)")
                .arg(idx)
                .arg(oldTime, 0, 'f', 3).arg(oldRatio * 100.0, 0, 'f', 1)
                .arg(newTime, 0, 'f', 3).arg(newRatio * 100.0, 0, 'f', 1);
        }
        if (indices.size() > 3) {
            qDebug() << QString("  ... (%1 more sources)").arg(indices.size() - 3);
        }
    }
    qDebug() << "=================================================";

    // Actually update the dwell times
    for (int i = 0; i < m_sources.size(); ++i) {
        m_sources[i].setDwellTime(dwellTimes[i]);
    }
}

QVector<double> BrachyPlan::getDwellTimes() const {
    QVector<double> times;
    times.reserve(m_sources.size());
    for (const auto& source : m_sources) {
        times.append(source.dwellTime());
    }
    return times;
}

void BrachyPlan::addReferencePoint(const ReferencePoint& point) {
    m_referencePoints.append(point);
}

void BrachyPlan::clearReferencePoints() {
    m_referencePoints.clear();
}
