#ifndef QCUSTOMPLOT_H
#define QCUSTOMPLOT_H

#include <QWidget>
#include <QObject>
#include <QVector>
#include <QPen>
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>
#include <QColor>
#include <QBrush>
#include <QRect>
#include <QString>
#include <QPointF>
#include <QFont>
#include <QFontMetrics>
#include <QPolygon>
#include <algorithm>
#include <limits>

#include "theme_manager.h"

class QCustomPlot;

class QCPAxis {
public:
    explicit QCPAxis(QCustomPlot* parentPlot);
    void setLabel(const QString& label) { m_label = label; }
    void setLabelFont(const QFont& font) { m_labelFont = font; }
    QFont labelFont() const { return m_labelFont; }
    void setRange(double min, double max) { m_min = min; m_max = max; }
    void setAutoTicks(bool on) { m_autoTicks = on; }
    void setTickVector(const QVector<double>& ticks) { m_ticks = ticks; }
    void setTickLabelFont(const QFont& font) { m_tickLabelFont = font; }
    QFont tickLabelFont() const { return m_tickLabelFont; }
    QString label() const { return m_label; }
    double min() const { return m_min; }
    double max() const { return m_max; }
    double range() const { return m_max - m_min; }
    bool autoTicks() const { return m_autoTicks; }
    const QVector<double>& tickVector() const { return m_ticks; }
    double pixelToCoord(int pixel) const;

private:
    QString m_label;
    QFont m_labelFont;
    QFont m_tickLabelFont;
    double m_min{0.0};
    double m_max{100.0};
    QCustomPlot* m_parent{nullptr};
    bool m_autoTicks{true};
    QVector<double> m_ticks;
};

class QCPGraph {
public:
    void setData(const QVector<double>& x, const QVector<double>& y) {
        m_xData = x;
        m_yData = y;
    }
    void setPen(const QPen& pen) { m_pen = pen; }
    void setVisible(bool visible) { m_visible = visible; }
    bool visible() const { return m_visible; }
    const QVector<double>& xData() const { return m_xData; }
    const QVector<double>& yData() const { return m_yData; }
    const QPen& pen() const { return m_pen; }
    void setBrush(const QBrush& brush) { m_brush = brush; }
    const QBrush& brush() const { return m_brush; }
    void setChannelFillGraph(QCPGraph* graph) { m_channelFillGraph = graph; }
    QCPGraph* channelFillGraph() const { return m_channelFillGraph; }

private:
    QVector<double> m_xData;
    QVector<double> m_yData;
    QPen m_pen{Qt::blue, 2};
    QBrush m_brush{Qt::NoBrush};
    QCPGraph* m_channelFillGraph{nullptr};
    bool m_visible{true};
};

class QCPItemPosition {
public:
    enum PositionType { ptAbsolute, ptViewportRatio, ptAxisRectRatio, ptPlotCoords };
    void setCoords(double x, double y) { m_x = x; m_y = y; }
    void setType(PositionType type) { m_type = type; }
    void setAxes(QCPAxis* keyAxis, QCPAxis* valueAxis) { m_keyAxis = keyAxis; m_valueAxis = valueAxis; }
    double x() const { return m_x; }
    double y() const { return m_y; }
private:
    double m_x{0.0};
    double m_y{0.0};
    PositionType m_type{ptAbsolute};
    QCPAxis* m_keyAxis{nullptr};
    QCPAxis* m_valueAxis{nullptr};
};

class QCPItemLine {
public:
    explicit QCPItemLine(QCustomPlot* parentPlot);
    ~QCPItemLine();
    void setPen(const QPen& pen) { m_pen = pen; }
    void setVisible(bool visible) { m_visible = visible; }
    bool visible() const { return m_visible; }
    const QPen& pen() const { return m_pen; }

    QCPItemPosition* start{&m_start};
    QCPItemPosition* end{&m_end};

private:
    QCustomPlot* m_parent{nullptr};
    QCPItemPosition m_start;
    QCPItemPosition m_end;
    QPen m_pen{Qt::black, 1};
    bool m_visible{false};
};

class QCPItemText {
public:
    explicit QCPItemText(QCustomPlot* parentPlot);
    ~QCPItemText();
    void setText(const QString& text) { m_text = text; }
    void setColor(const QColor& color) {
        m_color = color;
        m_useThemeColor = false;
    }
    void applyThemeColor(const QColor &color) {
        if (m_useThemeColor)
            m_color = color;
    }
    void setPositionAlignment(Qt::Alignment align) { m_align = align; }
    Qt::Alignment positionAlignment() const { return m_align; }
    void setVisible(bool visible) { m_visible = visible; }
    bool visible() const { return m_visible; }
    const QString& text() const { return m_text; }
    QColor color() const { return m_color; }
    void setFont(const QFont& font) { m_font = font; }
    QFont font() const { return m_font; }

    QCPItemPosition* position{&m_pos};

private:
    QCustomPlot* m_parent{nullptr};
    QCPItemPosition m_pos;
    QString m_text;
    QColor m_color{ThemeManager::instance().textColor()};
    bool m_visible{false};
    Qt::Alignment m_align{Qt::AlignCenter};
    QFont m_font;
    bool m_useThemeColor{true};
};

class QCustomPlot : public QWidget {
public:
    explicit QCustomPlot(QWidget* parent = nullptr)
        : QWidget(parent),
          xAxis(new QCPAxis(this)),
          yAxis(new QCPAxis(this)) {
        setMinimumSize(100, 80);
        ThemeManager &theme = ThemeManager::instance();
        theme.applyTextColor(
            this,
            QStringLiteral(
                "QCustomPlot { background-color: #222; border: 1px solid gray; color: %1; }"));
        QObject::connect(&theme, &ThemeManager::textColorChanged, this,
                         [this](const QColor &color) {
                             for (auto *item : m_itemTexts) {
                                 if (item)
                                     item->applyThemeColor(color);
                             }
                             update();
                         });
    }

    ~QCustomPlot() {
        clearGraphs();
        QVector<QCPItemLine*> lines = m_itemLines;
        m_itemLines.clear();
        for (auto* line : lines)
            delete line;
        QVector<QCPItemText*> texts = m_itemTexts;
        m_itemTexts.clear();
        for (auto* text : texts)
            delete text;
    }

    QCPGraph* addGraph() {
        QCPGraph* g = new QCPGraph();
        m_graphs.append(g);
        return g;
    }

    void clearGraphs() {
        for (auto* g : m_graphs)
            delete g;
        m_graphs.clear();
        update();
    }

    int graphCount() const { return m_graphs.size(); }

    QCPGraph* graph(int index) {
        if (index < 0 || index >= m_graphs.size()) return nullptr;
        return m_graphs[index];
    }

    void rescaleAxes() {
        if (m_graphs.isEmpty()) return;
        double xMin = std::numeric_limits<double>::max();
        double xMax = std::numeric_limits<double>::lowest();
        double yMin = std::numeric_limits<double>::max();
        double yMax = std::numeric_limits<double>::lowest();
        bool hasData = false;
        for (const auto* g : m_graphs) {
            if (!g->visible() || g->xData().isEmpty()) continue;
            const auto& xs = g->xData();
            const auto& ys = g->yData();
            auto xMinMax = std::minmax_element(xs.begin(), xs.end());
            auto yMinMax = std::minmax_element(ys.begin(), ys.end());
            xMin = std::min(xMin, *xMinMax.first);
            xMax = std::max(xMax, *xMinMax.second);
            yMin = std::min(yMin, *yMinMax.first);
            yMax = std::max(yMax, *yMinMax.second);
            hasData = true;
        }
        if (hasData) {
            double xMargin = (xMax - xMin) * 0.05;
            double yMargin = (yMax - yMin) * 0.05;
            xAxis->setRange(xMin - xMargin, xMax + xMargin);
            yAxis->setRange(yMin - yMargin, yMax + yMargin);
        }
    }

    void replot() { update(); }

    bool savePng(const QString& file) { return grab().save(file, "PNG"); }
    bool savePdf(const QString& file) { Q_UNUSED(file); return false; }

    void setPlotMargins(int left, int top, int right, int bottom)
    {
        m_leftMargin = std::max(0, left);
        m_topMargin = std::max(0, top);
        m_rightMargin = std::max(0, right);
        m_bottomMargin = std::max(0, bottom);
        update();
    }

    QRect plotRect() const
    {
        return rect().adjusted(m_leftMargin, m_topMargin, -m_rightMargin, -m_bottomMargin);
    }

    QPointF coordToPixel(double x, double y) const;

    void registerItem(QCPItemLine* item) { m_itemLines.append(item); }
    void registerItem(QCPItemText* item) { m_itemTexts.append(item); }
    void unregisterItem(QCPItemLine* item) { m_itemLines.removeOne(item); }
    void unregisterItem(QCPItemText* item) { m_itemTexts.removeOne(item); }

    QCPAxis* xAxis;
    QCPAxis* yAxis;

protected:
    void paintEvent(QPaintEvent* event) override {
        Q_UNUSED(event);
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        painter.fillRect(rect(), QColor(34, 34, 34));
        const QColor textColor = ThemeManager::instance().textColor();
        painter.setPen(QPen(textColor, 1));
        painter.drawRect(rect().adjusted(0, 0, -1, -1));

        QRect pr = plotRect();
        if (pr.width() <= 0 || pr.height() <= 0) return;

        double xRange = xAxis->range();
        double yRange = yAxis->range();
        if (xRange <= 0 || yRange <= 0) return;

        QVector<double> xTicks;
        QVector<double> yTicks;
        if (xAxis->autoTicks()) {
            for (int i = 0; i <= 10; ++i)
                xTicks << xAxis->min() + xRange * i / 10.0;
        } else {
            xTicks = xAxis->tickVector();
        }
        if (yAxis->autoTicks()) {
            for (int i = 0; i <= 10; ++i)
                yTicks << yAxis->min() + yRange * i / 10.0;
        } else {
            yTicks = yAxis->tickVector();
        }

        painter.setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
        for (int i = 1; i < xTicks.size() - 1; ++i) {
            double nx = (xTicks[i] - xAxis->min()) / xRange;
            int x = pr.left() + static_cast<int>(nx * pr.width());
            painter.drawLine(x, pr.top(), x, pr.bottom());
        }
        for (int i = 1; i < yTicks.size() - 1; ++i) {
            double ny = (yTicks[i] - yAxis->min()) / yRange;
            int y = pr.bottom() - static_cast<int>(ny * pr.height());
            painter.drawLine(pr.left(), y, pr.right(), y);
        }

        painter.setPen(QPen(textColor, 2));
        painter.drawLine(pr.bottomLeft(), pr.bottomRight());
        painter.drawLine(pr.bottomLeft(), pr.topLeft());

        QFont originalFont = painter.font();

        painter.setPen(textColor);
        painter.setFont(xAxis->tickLabelFont());
        auto formatTick = [](double v) {
            QString text = QString::number(v, 'f', 2);
            while (text.endsWith('0'))
                text.chop(1);
            if (text.endsWith('.'))
                text.chop(1);
            return text;
        };
        for (double xv : xTicks) {
            double nx = (xv - xAxis->min()) / xRange;
            int x = pr.left() + static_cast<int>(nx * pr.width());
            painter.drawLine(x, pr.bottom(), x, pr.bottom() + 5);
            painter.drawText(x - 20, pr.bottom() + 5, 40, 15,
                             Qt::AlignHCenter | Qt::AlignTop,
                             formatTick(xv));
        }
        painter.setFont(yAxis->tickLabelFont());
        for (double yv : yTicks) {
            double ny = (yv - yAxis->min()) / yRange;
            int y = pr.bottom() - static_cast<int>(ny * pr.height());
            painter.drawLine(pr.left() - 5, y, pr.left(), y);
            painter.drawText(pr.left() - 45, y - 7, 40, 15,
                             Qt::AlignRight | Qt::AlignVCenter,
                             formatTick(yv));
        }

        painter.setFont(originalFont);

        auto toPixel = [&](double x, double y) {
            double nx = (x - xAxis->min()) / xRange;
            double ny = 1.0 - (y - yAxis->min()) / yRange;
            int px = pr.left() + static_cast<int>(nx * pr.width());
            int py = pr.top() + static_cast<int>(ny * pr.height());
            return QPoint(px, py);
        };

        for (const auto* g : m_graphs) {
            if (!g->visible()) continue;
            const auto& xs = g->xData();
            const auto& ys = g->yData();
            if (xs.size() != ys.size() || xs.isEmpty()) continue;
            if (g->brush().style() != Qt::NoBrush && g->channelFillGraph()) {
                const auto* fill = g->channelFillGraph();
                const auto& xs2 = fill->xData();
                const auto& ys2 = fill->yData();
                if (xs2.size() == ys2.size() && xs2.size() == xs.size()) {
                    QPolygon poly;
                    poly.reserve(xs.size() + xs2.size());
                    for (int i = 0; i < xs.size(); ++i)
                        poly << toPixel(xs[i], ys[i]);
                    for (int i = xs2.size() - 1; i >= 0; --i)
                        poly << toPixel(xs2[i], ys2[i]);
                    painter.save();
                    painter.setPen(Qt::NoPen);
                    painter.setBrush(g->brush());
                    painter.drawPolygon(poly);
                    painter.restore();
                }
            }
            painter.setPen(g->pen());
            QVector<QPoint> pts;
            pts.reserve(xs.size());
            for (int i = 0; i < xs.size(); ++i)
                pts.append(toPixel(xs[i], ys[i]));
            if (pts.size() > 1) painter.drawPolyline(pts);
        }

        for (auto* line : m_itemLines) {
            if (!line || !line->visible()) continue;
            painter.setPen(line->pen());
            QPoint p1 = toPixel(line->start->x(), line->start->y());
            QPoint p2 = toPixel(line->end->x(), line->end->y());
            painter.drawLine(p1, p2);
        }

        for (auto* text : m_itemTexts) {
            if (!text || !text->visible()) continue;
            painter.save();
            painter.setPen(QPen(text->color()));
            painter.setFont(text->font());
            QFontMetrics fm(painter.font());
            QPoint p = toPixel(text->position->x(), text->position->y());
            // Support multi-line text by splitting on '\n'
            QStringList lines = text->text().split('\n');
            int lineH = fm.height();
            int maxW = 0;
            for (const QString &ln : lines)
                maxW = std::max(maxW, fm.horizontalAdvance(ln));
            int lineCount = std::max(1, static_cast<int>(lines.size()));
            int step = std::max(1, (lineH * 3) / 4); // ~75% of line height
            int totalH = lineH + (lineCount - 1) * step;
            QSize sz(maxW, totalH);
            QPoint topLeft = p;
            if (text->positionAlignment() & Qt::AlignHCenter)
                topLeft.setX(p.x() - sz.width() / 2);
            else if (text->positionAlignment() & Qt::AlignRight)
                topLeft.setX(p.x() - sz.width());
            if (text->positionAlignment() & Qt::AlignVCenter)
                topLeft.setY(p.y() - sz.height() / 2);
            else if (text->positionAlignment() & Qt::AlignBottom)
                topLeft.setY(p.y() - sz.height());
            // Draw each line stacked
            for (int i = 0; i < lines.size(); ++i) {
                QPoint linePos = topLeft + QPoint(0, i * step);
                painter.drawText(QRect(linePos, QSize(sz.width(), lineH)), Qt::AlignLeft | Qt::AlignTop, lines[i]);
            }
            painter.restore();
        }

        painter.setPen(textColor);
        painter.setFont(xAxis->labelFont());
        int bottomSpan = std::max(20, m_bottomMargin);
        painter.drawText(QRect(0, height() - bottomSpan, width(), bottomSpan), Qt::AlignCenter,
                         xAxis->label());
        painter.save();
        painter.translate(std::max(10, m_leftMargin / 2), height() / 2);
        painter.rotate(-90);
        painter.setFont(yAxis->labelFont());
        int leftSpan = std::max(20, m_leftMargin);
        painter.drawText(QRect(-leftSpan / 2, -10, leftSpan, 20), Qt::AlignCenter, yAxis->label());
        painter.restore();

        painter.setFont(originalFont);
    }

private:
    int m_leftMargin{60};
    int m_topMargin{40};
    int m_rightMargin{40};
    int m_bottomMargin{60};
    QVector<QCPGraph*> m_graphs;
    QVector<QCPItemLine*> m_itemLines;
    QVector<QCPItemText*> m_itemTexts;
};

inline double QCPAxis::pixelToCoord(int pixel) const {
    if (!m_parent) return 0.0;
    QRect pr = m_parent->plotRect();
    if (pr.width() <= 0) return m_min;
    double ratio = static_cast<double>(pixel - pr.left()) / pr.width();
    return m_min + ratio * (m_max - m_min);
}

inline QCPAxis::QCPAxis(QCustomPlot* parentPlot)
    : m_labelFont(parentPlot ? parentPlot->font() : QFont()),
      m_tickLabelFont(parentPlot ? parentPlot->font() : QFont()),
      m_parent(parentPlot) {}

inline QCPItemLine::QCPItemLine(QCustomPlot* parentPlot)
    : m_parent(parentPlot) {
    if (m_parent) m_parent->registerItem(this);
}

inline QCPItemLine::~QCPItemLine() {
    if (m_parent) m_parent->unregisterItem(this);
}

inline QCPItemText::QCPItemText(QCustomPlot* parentPlot)
    : m_parent(parentPlot),
      m_font(parentPlot ? parentPlot->font() : QFont()) {
    if (m_parent) m_parent->registerItem(this);
}

inline QCPItemText::~QCPItemText() {
    if (m_parent) m_parent->unregisterItem(this);
}

inline QPointF QCustomPlot::coordToPixel(double x, double y) const {
    double xRange = xAxis->range();
    double yRange = yAxis->range();
    if (xRange <= 0 || yRange <= 0) return QPointF();
    QRect pr = plotRect();
    double nx = (x - xAxis->min()) / xRange;
    double ny = 1.0 - (y - yAxis->min()) / yRange;
    return QPointF(pr.left() + nx * pr.width(), pr.top() + ny * pr.height());
}

#endif // QCUSTOMPLOT_H
