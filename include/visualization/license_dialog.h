#ifndef LICENSE_DIALOG_H
#define LICENSE_DIALOG_H

#include <QDialog>
#include <QTextBrowser>
#include <QPushButton>
#include <QVBoxLayout>
#include <QTimer>

class LicenseDialog : public QDialog
{
    Q_OBJECT

public:
    explicit LicenseDialog(QWidget *parent = nullptr);
    ~LicenseDialog() override = default;

private:
    void setupUi();
    QString generateLicenseInfo();
    void startAutoScroll();
    void startAutoClose();

private slots:
    void scrollDown();

private:
    QTextBrowser *m_textBrowser;
    QPushButton *m_closeButton;
    QTimer *m_scrollTimer;
    QTimer *m_closeTimer;
};

#endif // LICENSE_DIALOG_H
