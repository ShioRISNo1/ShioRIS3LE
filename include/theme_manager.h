#pragma once

#include <QObject>
#include <QColor>

class QWidget;

class ThemeManager : public QObject {
    Q_OBJECT
public:
    enum class TextTheme { DefaultWhite = 0, Green = 1, DarkRed = 2, Custom = 99 };

    static ThemeManager &instance();

    TextTheme currentTheme() const;
    QColor textColor() const;
    QString textColorCss() const;

    void setTextTheme(TextTheme theme);
    void setCustomTextColor(int r, int g, int b);
    void setCustomTextColor(const QColor &color);
    void applyTextColor(QWidget *widget);
    void applyTextColor(QWidget *widget, const QString &styleTemplate);

    static TextTheme themeFromInt(int value);
    static int toInt(TextTheme theme);

signals:
    void themeChanged(ThemeManager::TextTheme theme);
    void textColorChanged(const QColor &color);

private:
    ThemeManager();
    QColor colorForTheme(TextTheme theme) const;
    void applyToApplication();

    TextTheme m_theme;
    QColor m_textColor;
    bool m_initialized;
};

