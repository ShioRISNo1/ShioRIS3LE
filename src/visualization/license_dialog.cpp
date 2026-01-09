#include "visualization/license_dialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSizePolicy>
#include <QLabel>
#include <QFont>
#include <QPixmap>
#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QStringList>
#include <QScrollBar>

LicenseDialog::LicenseDialog(QWidget *parent)
    : QDialog(parent)
    , m_textBrowser(new QTextBrowser(this))
    , m_closeButton(new QPushButton(tr("Close"), this))
    , m_scrollTimer(new QTimer(this))
    , m_closeTimer(new QTimer(this))
{
    setupUi();
    // Auto-scroll disabled per user request
    // startAutoScroll();
    // Auto-close disabled per user request
    // startAutoClose();
}

void LicenseDialog::setupUi()
{
    setWindowTitle(tr("License Information"));
    setMinimumWidth(820);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(20, 20, 20, 10);
    mainLayout->setSpacing(10);

    QHBoxLayout *contentLayout = new QHBoxLayout();
    contentLayout->setSpacing(20);

    // Logo image
    QLabel *logoLabel = new QLabel(this);
    logoLabel->setFixedSize(350, 350);
    logoLabel->setAlignment(Qt::AlignCenter);

    auto scaledLogo = [](const QPixmap &pixmap) {
        return pixmap.scaled(350, 350, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    };

    QPixmap logo(":/images/ShioRIS3_Logo.png");
    if (logo.isNull()) {
        const QString appDir = QCoreApplication::applicationDirPath();
        const QStringList candidatePaths = {
            // macOS app bundle: Contents/Resources/images/
            appDir + "/../Resources/images/ShioRIS3_Logo.png",
            // Linux/Windows: executable directory
            appDir + QDir::separator() + "resources/images/ShioRIS3_Logo.png",
            QDir(appDir).absoluteFilePath("ShioRIS3_Logo.png"),
            QDir::current().absoluteFilePath("resources/images/ShioRIS3_Logo.png")
        };

        for (const QString &path : candidatePaths) {
            if (QFile::exists(path)) {
                logo.load(path);
                if (!logo.isNull()) {
                    break;
                }
            }
        }
    }

    if (!logo.isNull()) {
        logoLabel->setPixmap(scaledLogo(logo));
    }

    contentLayout->addWidget(logoLabel, 0, Qt::AlignTop);

    QVBoxLayout *infoLayout = new QVBoxLayout();
    infoLayout->setSpacing(10);

    // Header (title + version) as a compact two-line block
    QLabel *headerLabel = new QLabel(this);
    headerLabel->setTextFormat(Qt::RichText);
    const QString headerTitle = tr("ShioRIS3 - Open Source Licenses");
    const QString headerVersion = tr("Version: %1").arg(QApplication::applicationVersion());
    const QString headerHtml = QStringLiteral("<div style='line-height:1.15;'>"
                                             "<span style='font-size:14px; font-weight:600;'>%1</span><br>"
                                             "<span style='font-size:11px;'>%2</span>"
                                             "</div>")
                                 .arg(headerTitle, headerVersion);
    headerLabel->setText(headerHtml);
    headerLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    headerLabel->setContentsMargins(0, 0, 0, 0);
    headerLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    infoLayout->addWidget(headerLabel);

    // License information text browser
    m_textBrowser->setOpenExternalLinks(true);
    m_textBrowser->setHtml(generateLicenseInfo());
    m_textBrowser->setMinimumHeight(320);
    m_textBrowser->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    infoLayout->addWidget(m_textBrowser);

    contentLayout->addLayout(infoLayout, 1);
    mainLayout->addLayout(contentLayout);

    // Close button
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setContentsMargins(0, 10, 0, 0);
    buttonLayout->addStretch();
    buttonLayout->addWidget(m_closeButton);
    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);
    adjustSize();
}

QString LicenseDialog::generateLicenseInfo()
{
    QString html;

    html += "<html><head><style>";
    html += "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 15px; line-height: 1.1; letter-spacing: -0.1px; font-size: 11px; }";
    html += "h2 { color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 5px; margin-top: 15px; margin-bottom: 10px; font-size: 14px; }";
    html += "h3 { color: #f57c00; margin-top: 0; margin-bottom: 5px; font-size: 12px; }";
    html += ".library { padding: 10px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #1976d2; border: 1px solid #cccccc; }";
    html += ".license-type { color: #388e3c; font-weight: bold; }";
    html += "a { color: #1976d2; text-decoration: none; }";
    html += "a:hover { text-decoration: underline; }";
    html += "p { line-height: 1.1; margin: 3px 0; letter-spacing: -0.1px; font-size: 11px; }";
    html += ".note { color: #757575; font-style: italic; font-size: 10px; }";
    html += "</style></head><body>";

    html += "<h2>About ShioRIS3</h2>";
    html += "<p>ShioRIS3 is a prototype DICOM viewer and radiation treatment planning system ";
    html += "built with Qt6, DCMTK, and OpenCV.</p>";

    html += "<h2>Open Source Libraries</h2>";

    // Qt6
    html += "<div class='library'>";
    html += "<h3>Qt6</h3>";
    html += "<p><span class='license-type'>License:</span> LGPL v3 / GPL v3 / Commercial</p>";
    html += "<p>Qt is a cross-platform application development framework.</p>";
    html += "<p>Website: <a href='https://www.qt.io/'>https://www.qt.io/</a></p>";
    html += "<p>Components used: Core, Widgets, OpenGLWidgets, Concurrent, Network, Multimedia, Core5Compat</p>";
    html += "</div>";

    // DCMTK
    html += "<div class='library'>";
    html += "<h3>DCMTK (DICOM Toolkit)</h3>";
    html += "<p><span class='license-type'>License:</span> Modified BSD License</p>";
    html += "<p>DCMTK is a collection of libraries and applications for working with DICOM images and communication.</p>";
    html += "<p>Website: <a href='https://dicom.offis.de/dcmtk'>https://dicom.offis.de/dcmtk</a></p>";
    html += "<p>Components used: ofstd, oflog, dcmdata, dcmimgle, dcmimage</p>";
    html += "</div>";

    // OpenCV
    html += "<div class='library'>";
    html += "<h3>OpenCV</h3>";
    html += "<p><span class='license-type'>License:</span> Apache License 2.0</p>";
    html += "<p>OpenCV is an open-source computer vision and machine learning library.</p>";
    html += "<p>Website: <a href='https://opencv.org/'>https://opencv.org/</a></p>";
    html += "</div>";

    // SQLite
    html += "<div class='library'>";
    html += "<h3>SQLite</h3>";
    html += "<p><span class='license-type'>License:</span> Public Domain</p>";
    html += "<p>SQLite is a lightweight, self-contained database engine.</p>";
    html += "<p>Website: <a href='https://www.sqlite.org/'>https://www.sqlite.org/</a></p>";
    html += "</div>";

    // ONNX Runtime
    html += "<div class='library'>";
    html += "<h3>ONNX Runtime</h3>";
    html += "<p><span class='license-type'>License:</span> MIT License</p>";
    html += "<p>ONNX Runtime is a cross-platform inference engine for machine learning models.</p>";
    html += "<p>Website: <a href='https://onnxruntime.ai/'>https://onnxruntime.ai/</a></p>";
    html += "<p class='note'>Note: Used when AI features are enabled.</p>";
    html += "</div>";

    // whisper.cpp
    html += "<div class='library'>";
    html += "<h3>whisper.cpp</h3>";
    html += "<p><span class='license-type'>License:</span> MIT License</p>";
    html += "<p>whisper.cpp is a C/C++ implementation of OpenAI's Whisper speech recognition model.</p>";
    html += "<p>Website: <a href='https://github.com/ggerganov/whisper.cpp'>https://github.com/ggerganov/whisper.cpp</a></p>";
    html += "</div>";

    // OpenCL
    html += "<div class='library'>";
    html += "<h3>OpenCL</h3>";
    html += "<p><span class='license-type'>License:</span> Apache License 2.0</p>";
    html += "<p>OpenCL is a framework for parallel programming across heterogeneous platforms.</p>";
    html += "<p>Website: <a href='https://www.khronos.org/opencl/'>https://www.khronos.org/opencl/</a></p>";
    html += "<p class='note'>Note: Used when GPU acceleration is enabled.</p>";
    html += "</div>";

    // Metal (macOS)
    html += "<div class='library'>";
    html += "<h3>Metal Framework</h3>";
    html += "<p><span class='license-type'>License:</span> Apple SDK License</p>";
    html += "<p>Metal is a low-level graphics API for Apple platforms.</p>";
    html += "<p class='note'>Note: Used on macOS when GPU acceleration is enabled.</p>";
    html += "</div>";

    // QCustomPlot
    html += "<div class='library'>";
    html += "<h3>QCustomPlot</h3>";
    html += "<p><span class='license-type'>License:</span> GPL v3</p>";
    html += "<p>QCustomPlot is a plotting widget for Qt applications.</p>";
    html += "<p>Website: <a href='https://www.qcustomplot.com/'>https://www.qcustomplot.com/</a></p>";
    html += "</div>";

    html += "<h2>Copyright and License Notice</h2>";
    html += "<p>This software uses the above open-source libraries. ";
    html += "Each library is distributed under its respective license terms.</p>";
    html += "<p>For detailed license terms, please refer to the official website of each library.</p>";

    html += "<hr style='margin-top: 30px; border: none; border-top: 1px solid #4a4a4a;'>";
    html += "<p style='text-align: center; color: #888888;'>";
    html += "© 2024 ShioRIS3 Development Team<br>";
    html += "This software is a research and development prototype.";
    html += "</p>";

    html += "</body></html>";

    return html;
}

void LicenseDialog::startAutoScroll()
{
    // Start auto-scrolling after a short delay (100ms intervals)
    connect(m_scrollTimer, &QTimer::timeout, this, &LicenseDialog::scrollDown);
    m_scrollTimer->setInterval(100); // Scroll every 100ms
    m_scrollTimer->setSingleShot(false);

    QTimer::singleShot(1000, this, [this]() {
        m_scrollTimer->start();
    });
}

void LicenseDialog::startAutoClose()
{
    // Auto-close after 5 seconds
    connect(m_closeTimer, &QTimer::timeout, this, &QDialog::accept);
    m_closeTimer->setSingleShot(true);
    m_closeTimer->start(5000);  // 5000ms = 5 seconds
}

void LicenseDialog::scrollDown()
{
    QScrollBar *scrollBar = m_textBrowser->verticalScrollBar();
    if (scrollBar) {
        int currentValue = scrollBar->value();
        int maxValue = scrollBar->maximum();

        // Scroll down by 20 pixels each time
        if (currentValue < maxValue) {
            scrollBar->setValue(currentValue + 20);
        } else {
            // Stop scrolling when we reach the bottom
            m_scrollTimer->stop();
        }
    }
}
