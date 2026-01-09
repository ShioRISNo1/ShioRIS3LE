#include <QApplication>
#include <QStyleFactory>
#include <QDir>
#include <QStandardPaths>
#include <QDebug>
#include <QSettings>

#include "mainwindow.h"
#include "theme_manager.h"
#include "visualization/license_dialog.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // アプリケーション情報設定
    app.setApplicationName("ShioRIS3");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("ShioRIS3 Development Team");
    app.setOrganizationDomain("shioris3.org");

    // スタイル設定（オプション）
    app.setStyle(QStyleFactory::create("Fusion"));

    ThemeManager &themeManager = ThemeManager::instance();
    QSettings settings("ShioRIS3", "ShioRIS3");
    const int savedTheme = settings.value(
                                "appearance/textTheme",
                                ThemeManager::toInt(
                                    ThemeManager::TextTheme::DefaultWhite))
                                .toInt();
    themeManager.setTextTheme(ThemeManager::themeFromInt(savedTheme));

    // DCMTKの初期化
    qDebug() << "Initializing DCMTK...";

    // アプリケーションデータディレクトリの作成
    QString appDataPath =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(appDataPath);
    qDebug() << "Application data path:" << appDataPath;

    // Qt6.9では高DPI対応は自動的に有効
    // 以下の属性は非推奨のため削除
    // app.setAttribute(Qt::AA_EnableHighDpiScaling);
    // app.setAttribute(Qt::AA_UseHighDpiPixmaps);

    // ライセンス情報を最初に表示（モーダルダイアログ）
    LicenseDialog licenseDialog;
    licenseDialog.exec();

    // メインウィンドウの作成と表示（ダイアログが閉じられた後）
    MainWindow window;
    window.show();

    qDebug() << "ShioRIS3 application started successfully";

    return app.exec();
}