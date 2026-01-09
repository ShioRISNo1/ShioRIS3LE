#pragma once

#include <QCloseEvent>
#include <QHash>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QVector>
#include <QWidget>
#include <QStringList>

class DatabaseManager;
class SmartScanner;
class MetadataGenerator;
class FileStructureManager;

// DataWindow: Lightweight DB browser to load/organize studies and show
// previews.
class DataWindow : public QWidget {
  Q_OBJECT
public:
  DataWindow(DatabaseManager &db, SmartScanner &scanner,
             MetadataGenerator &meta, QWidget *parent = nullptr);

public slots:
  void refresh();

signals:
  void openStudyRequested(const QStringList &imageDirs,
                          const QStringList &modalities,
                          const QStringList &rtssPaths,
                          const QStringList &rtdosePaths);
  void changeDataRootRequested(const QString &newRoot);
  void openDicomFileRequested(const QString &filePath);
  void openDicomFolderRequested(const QString &folderPath);

private slots:
  void onOpenSelected();
  void onRescan();
  void onOpenDataFolder();
  void onCreatePatient();
  void onSelectionChanged();
  void onChangeDataFolder();
  void onImportDicomFile();
  void onImportDicomFolder();
  void onImportImageFiles();
  void onDeleteSelected();

protected:
  void closeEvent(QCloseEvent *event) override;

private:
  void setupUi();
  void loadPatients();
  void loadStudies(const QString &patientKey, QTreeWidgetItem *parent);
  void ensureStudyMetadataInfo();
  void updatePreviewForSelected();
  void clearPreview();
  void buildListsForPatient(const QString &patientName);
  void onPatientSelected();
  QStringList dicomFilesForStudy(const QString &groupKey,
                                 const QVector<int> &studyIds,
                                 const QString &studyPath);
  int dicomFileCountForStudy(const QString &groupKey,
                             const QVector<int> &studyIds,
                             const QString &studyPath);

  DatabaseManager &m_db;
  SmartScanner &m_scanner;
  MetadataGenerator &m_meta;

  // Lists: patients and image studies with RT children
  QTreeWidget *m_patientList{nullptr};
  QTreeWidget *m_imageList{nullptr};
  QLabel *m_preview{nullptr};
  QTimer *m_previewTimer{nullptr};
  QStringList m_previewQueue;
  int m_previewIndex{0};
  QLineEdit *m_patientFilter{nullptr};
  QPushButton *m_searchBtn{nullptr};
  // deprecated in new layout
  QTreeWidget *m_tree{nullptr};
  QTreeWidget *m_matchList{nullptr};
  QPushButton *m_openBtn{nullptr};
  QPushButton *m_rescanBtn{nullptr};
  QPushButton *m_openFolderBtn{nullptr};
  QPushButton *m_createPatientBtn{nullptr};
  QPushButton *m_changeDataFolderBtn{nullptr};
  QPushButton *m_openDicomFileBtn{nullptr};
  QPushButton *m_openDicomFolderBtn{nullptr};
  QPushButton *m_importDicomFileBtn{nullptr};
  QPushButton *m_importDicomFolderBtn{nullptr};
  QPushButton *m_importImageFilesBtn{nullptr};
  QPushButton *m_deleteSelectedBtn{nullptr};

  QHash<QString, int> m_dicomCountCache;
  QHash<QString, QStringList> m_dicomFilesCache;
  bool m_checkedStudyColumns{false};
  bool m_hasSeriesMetadata{false};
};
