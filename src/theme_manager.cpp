#include "theme_manager.h"

#include <QApplication>
#include <QPalette>
#include <QSettings>
#include <QString>
#include <QWidget>

ThemeManager &ThemeManager::instance() {
    static ThemeManager manager;
    return manager;
}

ThemeManager::ThemeManager()
    : QObject(nullptr),
      m_theme(TextTheme::DefaultWhite),
      m_textColor(colorForTheme(TextTheme::DefaultWhite)),
      m_initialized(false) {}

ThemeManager::TextTheme ThemeManager::currentTheme() const {
    return m_theme;
}

QColor ThemeManager::textColor() const {
    return m_textColor;
}

QString ThemeManager::textColorCss() const {
    return m_textColor.name(QColor::HexRgb);
}

void ThemeManager::setTextTheme(TextTheme theme) {
    if (theme == m_theme && m_initialized) {
        return;
    }

    m_theme = theme;
    m_textColor = colorForTheme(theme);
    applyToApplication();
    m_initialized = true;

    emit themeChanged(m_theme);
    emit textColorChanged(m_textColor);

    QSettings settings("ShioRIS3", "ShioRIS3");
    settings.setValue("appearance/textTheme", toInt(m_theme));
}

void ThemeManager::setCustomTextColor(int r, int g, int b) {
    setCustomTextColor(QColor(r, g, b));
}

void ThemeManager::setCustomTextColor(const QColor &color) {
    if (!color.isValid()) {
        return;
    }

    m_theme = TextTheme::Custom;
    m_textColor = color;
    applyToApplication();
    m_initialized = true;

    emit themeChanged(m_theme);
    emit textColorChanged(m_textColor);

    QSettings settings("ShioRIS3", "ShioRIS3");
    settings.setValue("appearance/textTheme", toInt(m_theme));
    settings.setValue("appearance/customTextColor", color.name(QColor::HexRgb));
}

void ThemeManager::applyTextColor(QWidget *widget) {
    applyTextColor(widget, QStringLiteral("color: %1;"));
}

void ThemeManager::applyTextColor(QWidget *widget, const QString &styleTemplate) {
    if (!widget) {
        return;
    }

    widget->setStyleSheet(styleTemplate.arg(textColorCss()));

    QObject::connect(this, &ThemeManager::textColorChanged, widget,
                     [widget, styleTemplate](const QColor &color) {
                         widget->setStyleSheet(
                             styleTemplate.arg(color.name(QColor::HexRgb)));
                     });
}

ThemeManager::TextTheme ThemeManager::themeFromInt(int value) {
    switch (value) {
    case 1:
        return TextTheme::Green;
    case 2:
        return TextTheme::DarkRed;
    case 99:
        return TextTheme::Custom;
    default:
        return TextTheme::DefaultWhite;
    }
}

int ThemeManager::toInt(TextTheme theme) {
    switch (theme) {
    case TextTheme::Green:
        return 1;
    case TextTheme::DarkRed:
        return 2;
    case TextTheme::Custom:
        return 99;
    case TextTheme::DefaultWhite:
    default:
        return 0;
    }
}

QColor ThemeManager::colorForTheme(TextTheme theme) const {
    switch (theme) {
    case TextTheme::Green:
        return QColor(0, 255, 0);
    case TextTheme::DarkRed:
        return QColor(204, 85, 85);
    case TextTheme::Custom: {
        // Load custom color from settings
        QSettings settings("ShioRIS3", "ShioRIS3");
        QString colorName = settings.value("appearance/customTextColor", "#ffffff").toString();
        return QColor(colorName);
    }
    case TextTheme::DefaultWhite:
    default:
        return QColor(255, 255, 255);
    }
}

void ThemeManager::applyToApplication() {
    if (!qApp) {
        return;
    }

    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::WindowText, m_textColor);
    darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
    darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, m_textColor);
    darkPalette.setColor(QPalette::Text, m_textColor);
    darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ButtonText, m_textColor);
    darkPalette.setColor(QPalette::BrightText, QColor(255, 85, 85));
    darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::HighlightedText, Qt::black);
    qApp->setPalette(darkPalette);

    qApp->setStyleSheet(
        QStringLiteral("* { color: %1; }").arg(textColorCss()));
}

