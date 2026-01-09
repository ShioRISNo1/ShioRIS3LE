#include "visualization/data_window.h"
#include "data/file_structure_manager.h"
#include "data/metadata_generator.h"
#include "database/database_manager.h"
#include "database/smart_scanner.h"
#include "dicom/dicom_reader.h"

#include <QDate>
#include <QDesktopServices>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QHash>
#include <QImage>
#include <QInputDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QSet>
#include <QStringList>
#include <QVector>
#include <QUrl>
#include <dcmtk/dcmdata/dctk.h>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <utility>
#include <vector>
#include <string>

namespace {
QString canonicalModality(const QString &raw) {
  QString mod = raw.trimmed().toUpper();
  if (mod.isEmpty())
    return QStringLiteral("OTHERS");

  if (mod == "MR")
    return QStringLiteral("MRI");
  if (mod == "PT")
    return QStringLiteral("PET");
  if (mod == "CT" || mod == "MRI" || mod == "PET" || mod == "OTHERS")
    return mod;

  if (mod.startsWith("FUSION")) {
    QString suffix = mod.mid(QStringLiteral("FUSION").size());
    suffix = suffix.trimmed();
    if (suffix.startsWith(QLatin1Char('/')))
      suffix.remove(0, 1);
    if (suffix.isEmpty())
      suffix = QStringLiteral("OTHERS");
    return QStringLiteral("Fusion/%1").arg(suffix);
  }

  QString normalized = mod;
  normalized.replace('-', ' ');
  normalized.replace('_', ' ');
  normalized = normalized.simplified();
  const QStringList tokens =
      normalized.split(' ', Qt::SkipEmptyParts);

  auto hasToken = [&](std::initializer_list<const char *> words) {
    for (const QString &token : tokens) {
      for (const char *word : words) {
        if (token == QLatin1String(word))
          return true;
      }
    }
    return false;
  };

  if (hasToken({"RTSTRUCT", "RTSS", "STRUCTURE", "STRUCTURES", "STRUCTS"}))
    return QStringLiteral("RTSTRUCT");
  if (hasToken({"FUSION"})) {
    QString target;
    if (hasToken({"CT"}))
      target = QStringLiteral("CT");
    else if (hasToken({"MRI", "MR"}))
      target = QStringLiteral("MRI");
    else if (hasToken({"PET"}))
      target = QStringLiteral("PET");
    else
      target = QStringLiteral("OTHERS");
    return QStringLiteral("Fusion/%1").arg(target);
  }
  if (hasToken({"RTDOSE", "DOSE", "DOSES"}))
    return QStringLiteral("RTDOSE");
  if (hasToken({"RTPLAN", "PLAN", "PLANS"}))
    return QStringLiteral("RTPLAN");
  if (hasToken({"RTIMAGE"}) ||
      (hasToken({"RT"}) && hasToken({"IMAGE", "IMAGES"})))
    return QStringLiteral("RTIMAGE");
  if (hasToken({"RTRECORD"}) ||
      (hasToken({"RT"}) && hasToken({"RECORD", "RECORDS"})))
    return QStringLiteral("RTRECORD");
  if (hasToken({"RTANALYSIS", "ANALYSIS"}))
    return QStringLiteral("RTANALYSIS");
  if (mod.startsWith("RT"))
    return mod;
  return QStringLiteral("OTHERS");
}

bool isImagingModality(const QString &modality) {
  return modality == "CT" || modality == "MRI" || modality == "PET" ||
         modality == "OTHERS" || modality.startsWith(QLatin1String("Fusion/"));
}

QString fallbackStudyName(const QString &name, const QString &path) {
  QString trimmed = name.trimmed();
  if (!trimmed.isEmpty())
    return trimmed;
  QFileInfo info(path);
  trimmed = info.fileName();
  if (!trimmed.isEmpty())
    return trimmed;
  return QString();
}

QString friendlyRtModality(const QString &modality) {
  if (modality == "RTDOSE")
    return QStringLiteral("RT Dose");
  if (modality == "RTSTRUCT")
    return QStringLiteral("RT Structure Set");
  if (modality == "RTPLAN")
    return QStringLiteral("RT Plan");
  if (modality == "RTIMAGE")
    return QStringLiteral("RT Image");
  if (modality == "RTRECORD")
    return QStringLiteral("RT Record");
  if (modality == "RTANALYSIS")
    return QStringLiteral("RT Analysis");
  return modality;
}

std::string sqlEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '\'')
      out += "''";
    else
      out += c;
  }
  return out;
}

QString studyDisplayLabel(const QString &modality, const QString &name,
                          const QString &path, int dicomCount,
                          const QString &seriesDescription) {
  const bool isRt = modality.startsWith(QLatin1String("RT"));
  QString baseName = fallbackStudyName(name, path);
  QString label;
  if (isRt) {
    const QString friendly = friendlyRtModality(modality);
    QString detail = seriesDescription.trimmed();
    if (detail.isEmpty())
      detail = baseName;
    if (!detail.isEmpty() &&
        detail.compare(friendly, Qt::CaseInsensitive) != 0)
      label = QStringLiteral("%1 - %2").arg(friendly, detail);
    else
      label = friendly;
  } else {
    if (baseName.isEmpty() ||
        baseName.compare(modality, Qt::CaseInsensitive) == 0) {
      label = modality;
    } else {
      label = QStringLiteral("%1 - %2").arg(modality, baseName);
    }
  }
  if (dicomCount >= 0) {
    if (isRt) {
      if (dicomCount == 1)
        label = QStringLiteral("%1 (1 file)").arg(label);
      else
        label = QStringLiteral("%1 (%2 files)").arg(label).arg(dicomCount);
    } else {
      if (dicomCount == 1)
        label = QStringLiteral("%1 (1 image)").arg(label);
      else
        label = QStringLiteral("%1 (%2 images)").arg(label).arg(dicomCount);
    }
  }
  return label;
}

bool isDicomFile(const QString &filePath) {
  const QByteArray encoded = QFile::encodeName(filePath);
  DcmFileFormat ff;
  return ff.loadFile(encoded.constData()).good();
}

QString firstDicomFileIn(const QString &path) {
  QFileInfo info(path);
  if (info.isDir()) {
    QDirIterator it(path, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
      const QString f = it.next();
      if (isDicomFile(f))
        return f;
    }
    QDir dir(path);
    const QFileInfoList files = dir.entryInfoList(QDir::Files, QDir::Name);
    if (!files.isEmpty())
      return files.first().absoluteFilePath();
  }
  return path;
}
} // namespace

DataWindow::DataWindow(DatabaseManager &db, SmartScanner &scanner,
                       MetadataGenerator &meta, QWidget *parent)
    : QWidget(parent), m_db(db), m_scanner(scanner), m_meta(meta) {
  setupUi();
  // Make it a top-level window even if a parent is provided (for ownership)
  setWindowFlag(Qt::Window, true);
  setAttribute(Qt::WA_DeleteOnClose, false);
  refresh();
}

void DataWindow::setupUi() {
  auto *root = new QVBoxLayout(this);

  // Patient filter bar
  auto *topBar = new QHBoxLayout();
  topBar->addWidget(new QLabel("Patient Name:", this));
  m_patientFilter = new QLineEdit(this);
  m_searchBtn = new QPushButton("Search", this);
  topBar->addWidget(m_patientFilter);
  topBar->addWidget(m_searchBtn);
  root->addLayout(topBar);

  // 1) Patient list (names only) with a square preview to the right
  auto *patientRow = new QHBoxLayout();
  m_patientList = new QTreeWidget(this);
  m_patientList->setColumnCount(1);
  m_patientList->setHeaderLabels({"Patient Name"});
  m_patientList->header()->setSectionResizeMode(0, QHeaderView::Stretch);
  connect(m_patientList, &QTreeWidget::itemSelectionChanged, this,
          &DataWindow::onPatientSelected);
  patientRow->addWidget(m_patientList, 1);

  auto *previewLayout = new QVBoxLayout();
  m_preview = new QLabel(this);
  m_preview->setFixedSize(256, 256);
  m_preview->setAlignment(Qt::AlignCenter);
  m_preview->setText("No Preview");
  previewLayout->addWidget(m_preview);
  previewLayout->addStretch();
  patientRow->addLayout(previewLayout);
  root->addLayout(patientRow, 1);

  // 2) Image studies; RT items will appear as children under each image study
  m_imageList = new QTreeWidget(this);
  m_imageList->setColumnCount(4);
  m_imageList->setHeaderLabels({"Study/RT", "Modality", "Date/UID", "Path"});
  // Narrow columns resize to contents, Path stretches
  m_imageList->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  m_imageList->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
  m_imageList->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
  m_imageList->header()->setSectionResizeMode(3, QHeaderView::Stretch);
  m_imageList->setTextElideMode(Qt::ElideMiddle);
  connect(m_imageList, &QTreeWidget::itemSelectionChanged, this,
          &DataWindow::onSelectionChanged);
  root->addWidget(m_imageList, 2);

  // Controls: two-row layout for better button sizing
  auto *controls = new QVBoxLayout();
  auto *row1 = new QHBoxLayout();
  auto *row2 = new QHBoxLayout();
  m_openBtn = new QPushButton("Open Selected", this);
  m_openDicomFileBtn = new QPushButton("Open DICOM File...", this);
  m_openDicomFolderBtn = new QPushButton("Open DICOM Folder...", this);
  m_openFolderBtn = new QPushButton("Open Data Folder", this);
  row1->addWidget(m_openBtn);
  row1->addWidget(m_openDicomFileBtn);
  row1->addWidget(m_openDicomFolderBtn);
  row1->addWidget(m_openFolderBtn);
  row1->addStretch();

  m_importDicomFileBtn = new QPushButton("Import DICOM File to DB", this);
  m_importDicomFolderBtn = new QPushButton("Import DICOM Folder to DB", this);
  m_importImageFilesBtn = new QPushButton("Import Images to DB", this);
  m_createPatientBtn = new QPushButton("Create Patient", this);
  m_changeDataFolderBtn = new QPushButton("Change Data Folder", this);
  m_rescanBtn = new QPushButton("Rescan", this);
  m_deleteSelectedBtn = new QPushButton("Delete Selected", this);
  row2->addWidget(m_importDicomFileBtn);
  row2->addWidget(m_importDicomFolderBtn);
  row2->addWidget(m_importImageFilesBtn);
  row2->addWidget(m_createPatientBtn);
  row2->addWidget(m_changeDataFolderBtn);
  row2->addWidget(m_rescanBtn);
  row2->addWidget(m_deleteSelectedBtn);
  row2->addStretch();

  controls->addLayout(row1);
  controls->addLayout(row2);
  root->addLayout(controls);

  connect(m_openBtn, &QPushButton::clicked, this, &DataWindow::onOpenSelected);
  connect(m_rescanBtn, &QPushButton::clicked, this, &DataWindow::onRescan);
  connect(m_openFolderBtn, &QPushButton::clicked, this,
          &DataWindow::onOpenDataFolder);
  connect(m_createPatientBtn, &QPushButton::clicked, this,
          &DataWindow::onCreatePatient);
  connect(m_changeDataFolderBtn, &QPushButton::clicked, this,
          &DataWindow::onChangeDataFolder);
  connect(m_openDicomFileBtn, &QPushButton::clicked, this, [this] {
    const QString file = QFileDialog::getOpenFileName(
        this, "Open DICOM File", QDir::homePath(),
        "All Files (*.*);;DICOM Files (*.dcm *.DCM *.dicom *.DICOM)");
    if (!file.isEmpty())
      emit openDicomFileRequested(file);
  });
  connect(m_openDicomFolderBtn, &QPushButton::clicked, this, [this] {
    const QString dir = QFileDialog::getExistingDirectory(
        this, "Open DICOM Folder", QDir::homePath());
    if (!dir.isEmpty())
      emit openDicomFolderRequested(dir);
  });
  connect(m_importDicomFileBtn, &QPushButton::clicked, this,
          &DataWindow::onImportDicomFile);
  connect(m_importDicomFolderBtn, &QPushButton::clicked, this,
          &DataWindow::onImportDicomFolder);
  connect(m_importImageFilesBtn, &QPushButton::clicked, this,
          &DataWindow::onImportImageFiles);
  connect(m_deleteSelectedBtn, &QPushButton::clicked, this,
          &DataWindow::onDeleteSelected);
  connect(m_searchBtn, &QPushButton::clicked, this,
          [this] { buildListsForPatient(m_patientFilter->text()); });
  connect(m_patientFilter, &QLineEdit::returnPressed, this,
          [this] { buildListsForPatient(m_patientFilter->text()); });

  // Preview timer
  m_previewTimer = new QTimer(this);
  m_previewTimer->setInterval(800);
  connect(m_previewTimer, &QTimer::timeout, this, [this] {

    
    if (m_previewQueue.isEmpty())
      return;
    if (m_previewIndex < 0 || m_previewIndex >= m_previewQueue.size())
      m_previewIndex = 0;
    const QString f = m_previewQueue.at(m_previewIndex);
    ++m_previewIndex;
    // Try DICOM first then image
    QImage img;
    {
      DicomReader r;
      if (r.loadDicomFile(f))
        img = r.getImage();
    }
    if (img.isNull())
      img.load(f);
    if (!img.isNull())
      m_preview->setPixmap(QPixmap::fromImage(
          img.scaled(256, 256, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
  });
}

void DataWindow::refresh() { loadPatients(); }

void DataWindow::loadPatients() {
  if (!m_patientList)
    return;
  m_patientList->clear();
  m_db.query("SELECT patient_key, name FROM patients ORDER BY name;",
             [&](int argc, char **argv, char **) {
               if (argc < 2)
                 return;

               const QString patientKey =
                   argv[0] ? QString::fromUtf8(argv[0]) : QString();
               const QString patientName =
                   argv[1] ? QString::fromUtf8(argv[1]) : QString();
               const QString trimmedName = patientName.trimmed();

               if (patientKey == QStringLiteral(".") ||
                   trimmedName.isEmpty() ||
                   trimmedName == QStringLiteral(".")) {
                 return;
               }

               auto *item = new QTreeWidgetItem();
               item->setText(0, patientName);
               item->setData(0, Qt::UserRole, patientKey);
               m_patientList->addTopLevelItem(item);
             });
  m_patientList->expandToDepth(0);
}

void DataWindow::ensureStudyMetadataInfo() {
  if (m_checkedStudyColumns)
    return;
  m_checkedStudyColumns = true;
  bool hasSeriesUid = false;
  bool hasSeriesDescription = false;
  m_db.query("PRAGMA table_info(studies);",
             [&](int argc, char **argv, char **) {
               if (argc < 2 || !argv[1])
                 return;
               const QString column = QString::fromUtf8(argv[1]);
               if (column.compare(QStringLiteral("series_uid"),
                                  Qt::CaseInsensitive) == 0)
                 hasSeriesUid = true;
               else if (column.compare(QStringLiteral("series_description"),
                                       Qt::CaseInsensitive) == 0)
                 hasSeriesDescription = true;
             });
  m_hasSeriesMetadata = hasSeriesUid && hasSeriesDescription;
}

void DataWindow::loadStudies(const QString &patientKey,
                             QTreeWidgetItem *parent) {
  if (!parent) {
    m_dicomCountCache.clear();
    m_dicomFilesCache.clear();
  }

  ensureStudyMetadataInfo();

  struct StudyRow {
    int id{0};
    QVector<int> studyIds;
    QString modality;
    QString date;
    QString name;
    QString path;
    QString dbPath;
    QString frame;
    QString seriesUid;
    QString seriesDescription;
    QString key;
  };

  QVector<StudyRow> orderedStudies;
  QHash<QString, int> keyToIndex;

  const std::string escapedKey = sqlEscape(patientKey.toStdString());

  auto runQuery = [&](bool includeSeries) -> bool {
    std::stringstream q;
    if (includeSeries) {
      q << "SELECT id, modality, study_date, study_name, path, frame_uid, "
           "series_uid, series_description FROM studies WHERE patient_key='"
        << escapedKey << "' ORDER BY id;";
    } else {
      q << "SELECT id, modality, study_date, study_name, path, frame_uid "
           "FROM studies WHERE patient_key='" << escapedKey
        << "' ORDER BY id;";
    }
    const int expectedCols = includeSeries ? 8 : 6;
    bool ok = m_db.query(q.str(), [&](int argc, char **argv, char **) {
      if (argc < expectedCols)
        return;

      StudyRow row;
      row.id = std::atoi(argv[0] ? argv[0] : "0");
      row.modality =
          canonicalModality(QString::fromUtf8(argv[1] ? argv[1] : ""));
      row.date = QString::fromUtf8(argv[2] ? argv[2] : "").trimmed();
      row.name = QString::fromUtf8(argv[3] ? argv[3] : "").trimmed();
      QString rawPath = QString::fromUtf8(argv[4] ? argv[4] : "");
      if (!rawPath.isEmpty()) {
        QString normalized = QDir::fromNativeSeparators(rawPath);
        QDir dir(normalized);
        normalized = QDir::cleanPath(dir.absolutePath());
        row.path = normalized;
      } else {
        row.path = rawPath;
      }
      row.dbPath = rawPath;
      if (row.dbPath.isEmpty())
        row.dbPath = row.path;
      row.frame = QString::fromUtf8(argv[5] ? argv[5] : "").trimmed();
      if (includeSeries) {
        row.seriesUid = QString::fromUtf8(argv[6] ? argv[6] : "").trimmed();
        row.seriesDescription =
            QString::fromUtf8(argv[7] ? argv[7] : "").trimmed();
      } else {
        row.seriesUid.clear();
        row.seriesDescription.clear();
      }
      QString pathForKey = row.path;
      if (!pathForKey.isEmpty()) {
        QDir pathDir(pathForKey);
        pathForKey = QDir::cleanPath(pathDir.absolutePath()).toLower();
      }
      QString key = row.modality + QLatin1Char('\x1F');
      if (row.modality.startsWith(QLatin1String("RT"))) {
        key += row.frame + QLatin1Char('\x1F');
        if (includeSeries)
          key += row.seriesUid + QLatin1Char('\x1F');
      }
      key += pathForKey;
      row.key = key;

      if (keyToIndex.contains(row.key)) {
        StudyRow &existing = orderedStudies[keyToIndex.value(row.key)];
        existing.studyIds.append(row.id);
        if (existing.id <= 0 || row.id < existing.id)
          existing.id = row.id;
        if (existing.date.isEmpty() && !row.date.isEmpty())
          existing.date = row.date;
        if (existing.name.isEmpty() && !row.name.isEmpty())
          existing.name = row.name;
        if (existing.frame.isEmpty() && !row.frame.isEmpty())
          existing.frame = row.frame;
        if (includeSeries) {
          if (existing.seriesUid.isEmpty() && !row.seriesUid.isEmpty())
            existing.seriesUid = row.seriesUid;
          if (existing.seriesDescription.isEmpty() &&
              !row.seriesDescription.isEmpty())
            existing.seriesDescription = row.seriesDescription;
        }
        return;
      }

      row.studyIds.append(row.id);
      int insertIndex = orderedStudies.size();
      keyToIndex.insert(row.key, insertIndex);
      orderedStudies.push_back(row);
    });
    return ok;
  };

  bool needFallback = !m_hasSeriesMetadata;
  if (m_hasSeriesMetadata) {
    if (!runQuery(true)) {
      orderedStudies.clear();
      keyToIndex.clear();
      m_hasSeriesMetadata = false;
      needFallback = true;
    }
  }
  if (needFallback) {
    if (!runQuery(false))
      return;
  }

  auto normalizePathKey = [](const QString &path) {
    QString normalized = QDir::fromNativeSeparators(path);
    normalized = QDir::cleanPath(normalized);
    if (normalized == QLatin1String("."))
      normalized.clear();
    return normalized.toLower();
  };

  auto mergeStudyRow = [](StudyRow &dst, const StudyRow &src) {
    for (int id : src.studyIds) {
      if (id > 0 && !dst.studyIds.contains(id))
        dst.studyIds.append(id);
    }
    if ((dst.id <= 0 || (src.id > 0 && src.id < dst.id)))
      dst.id = src.id;
    if (dst.date.isEmpty() && !src.date.isEmpty())
      dst.date = src.date;
    if (dst.name.isEmpty() && !src.name.isEmpty())
      dst.name = src.name;
    if (dst.path.isEmpty() && !src.path.isEmpty())
      dst.path = src.path;
    if (dst.dbPath.isEmpty() && !src.dbPath.isEmpty())
      dst.dbPath = src.dbPath;
    if (dst.frame.isEmpty() && !src.frame.isEmpty())
      dst.frame = src.frame;
    if (dst.seriesUid.isEmpty() && !src.seriesUid.isEmpty())
      dst.seriesUid = src.seriesUid;
    if (dst.seriesDescription.isEmpty() && !src.seriesDescription.isEmpty())
      dst.seriesDescription = src.seriesDescription;
  };

  QVector<StudyRow> imageStudies;
  QVector<StudyRow> rtStudies;
  QVector<StudyRow> otherStudies;
  QHash<QString, int> imageIndex;
  QHash<QString, int> rtIndex;
  QHash<QString, int> otherIndex;

  for (const StudyRow &row : std::as_const(orderedStudies)) {
    if (isImagingModality(row.modality)) {
      const QString normalizedPath = normalizePathKey(row.path);
      QString imagingKey;
      if (!normalizedPath.isEmpty())
        imagingKey = row.modality + QLatin1Char('\x1F') + normalizedPath;
      else if (!row.key.isEmpty())
        imagingKey = row.key;
      else
        imagingKey = row.modality + QLatin1Char('\x1F');
      if (imageIndex.contains(imagingKey)) {
        mergeStudyRow(imageStudies[imageIndex.value(imagingKey)], row);
      } else {
        StudyRow copy = row;
        copy.key = imagingKey;
        imageIndex.insert(imagingKey, imageStudies.size());
        imageStudies.push_back(copy);
      }
    } else if (row.modality.startsWith("RT")) {
      QString rtKey = row.key;
      if (rtKey.isEmpty()) {
        const QString normalizedPath = normalizePathKey(row.path);
        rtKey = row.modality + QLatin1Char('\x1F') + row.frame +
                QLatin1Char('\x1F') + normalizedPath;
      }
      if (rtIndex.contains(rtKey)) {
        mergeStudyRow(rtStudies[rtIndex.value(rtKey)], row);
      } else {
        StudyRow copy = row;
        copy.key = rtKey;
        rtIndex.insert(rtKey, rtStudies.size());
        rtStudies.push_back(copy);
      }
    } else {
      QString otherKey = row.key;
      if (otherKey.isEmpty()) {
        const QString normalizedPath = normalizePathKey(row.path);
        otherKey = row.modality + QLatin1Char('\x1F') + normalizedPath;
      }
      if (otherIndex.contains(otherKey)) {
        mergeStudyRow(otherStudies[otherIndex.value(otherKey)], row);
      } else {
        StudyRow copy = row;
        copy.key = otherKey;
        otherIndex.insert(otherKey, otherStudies.size());
        otherStudies.push_back(copy);
      }
    }
  }

  QHash<QString, QVector<int>> rtByFrame;
  for (int i = 0; i < rtStudies.size(); ++i)
    rtByFrame[rtStudies[i].frame].append(i);

  QSet<QString> seenRtDicomFiles;
  QSet<QString> seenRtFallbacks;
  QSet<QString> topLevelRtModalityChecked;
  QHash<QTreeWidgetItem *, QSet<QString>> rtModalityCheckedByParent;

  auto shouldCheckRtModality = [&](QTreeWidgetItem *parentItem,
                                   const QString &modality) {
    if (modality == QLatin1String("RTSTRUCT") ||
        modality == QLatin1String("RTDOSE")) {
      QSet<QString> *checkedSet = nullptr;
      if (parentItem)
        checkedSet = &rtModalityCheckedByParent[parentItem];
      else
        checkedSet = &topLevelRtModalityChecked;
      if (checkedSet->contains(modality))
        return false;
      checkedSet->insert(modality);
    }
    return true;
  };

  auto addRtItem = [&](const StudyRow &rt, QTreeWidgetItem *parentItem) {
    const QStringList dicomFiles =
        dicomFilesForStudy(rt.key, rt.studyIds, rt.path);
    QString baseLabel =
        studyDisplayLabel(rt.modality, rt.name, rt.path, -1, rt.seriesDescription);
    if (dicomFiles.isEmpty()) {
      const QString fallbackKey =
          rt.modality + QLatin1Char('\x1F') + normalizePathKey(rt.path);
      if (seenRtFallbacks.contains(fallbackKey))
        return;
      seenRtFallbacks.insert(fallbackKey);
      const bool shouldCheck = shouldCheckRtModality(parentItem, rt.modality);
      QTreeWidgetItem *item = parentItem ? new QTreeWidgetItem(parentItem)
                                         : new QTreeWidgetItem();
      if (!parentItem)
        m_imageList->addTopLevelItem(item);
      item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
      item->setCheckState(0, shouldCheck ? Qt::Checked : Qt::Unchecked);

      QString label = studyDisplayLabel(rt.modality, rt.name, rt.path, 0, rt.seriesDescription);

      // Check if this is a ShioRIS3 calculated dose
      bool isCalculatedDose = rt.name.contains("ShioRIS3 Calculated Dose", Qt::CaseInsensitive);
      if (isCalculatedDose) {
        label = QString("[ShioRIS3] %1").arg(label);
        item->setForeground(0, QColor(0, 150, 0)); // Dark green text
        item->setForeground(1, QColor(0, 150, 0));
      }

      item->setText(0, label);
      item->setText(1, rt.modality);
      item->setText(2, rt.frame);
      item->setText(3, rt.path);
      item->setData(3, Qt::UserRole, rt.path);
      const QString dbPath = !rt.dbPath.isEmpty() ? rt.dbPath : rt.path;
      item->setData(3, Qt::UserRole + 3, dbPath);
      item->setData(3, Qt::UserRole + 1, rt.frame);
      return;
    }

    QStringList uniqueFiles;
    uniqueFiles.reserve(dicomFiles.size());
    for (const QString &dicomPath : dicomFiles) {
      const QString normalizedFile = normalizePathKey(dicomPath);
      if (seenRtDicomFiles.contains(normalizedFile))
        continue;
      seenRtDicomFiles.insert(normalizedFile);
      uniqueFiles.append(dicomPath);
    }

    if (uniqueFiles.isEmpty())
      return;

    // Create parent item for the study with file count
    const bool shouldCheck = shouldCheckRtModality(parentItem, rt.modality);
    QTreeWidgetItem *studyItem = parentItem ? new QTreeWidgetItem(parentItem)
                                            : new QTreeWidgetItem();
    if (!parentItem)
      m_imageList->addTopLevelItem(studyItem);
    // Parent item does not have checkbox - only children are checkable

    QString studyLabel = QStringLiteral("%1 (%2 file%3)")
        .arg(baseLabel)
        .arg(uniqueFiles.size())
        .arg(uniqueFiles.size() > 1 ? "s" : "");

    // Check if this is a ShioRIS3 calculated dose
    bool isCalculatedDose = rt.name.contains("ShioRIS3 Calculated Dose", Qt::CaseInsensitive);
    if (isCalculatedDose) {
      studyLabel = QString("[ShioRIS3] %1").arg(studyLabel);
      studyItem->setForeground(0, QColor(0, 150, 0)); // Dark green text
      studyItem->setForeground(1, QColor(0, 150, 0));
    }

    studyItem->setText(0, studyLabel);
    studyItem->setText(1, rt.modality);
    studyItem->setText(2, rt.frame);
    studyItem->setText(3, rt.path);
    studyItem->setData(3, Qt::UserRole, rt.path);
    const QString dbPath = !rt.dbPath.isEmpty() ? rt.dbPath : rt.path;
    studyItem->setData(3, Qt::UserRole + 3, dbPath);
    studyItem->setData(3, Qt::UserRole + 1, rt.frame);

    // Add each file as a child item
    for (const QString &dicomPath : std::as_const(uniqueFiles)) {
      QTreeWidgetItem *fileItem = new QTreeWidgetItem(studyItem);
      fileItem->setFlags(fileItem->flags() | Qt::ItemIsUserCheckable);
      fileItem->setCheckState(0, shouldCheck ? Qt::Checked : Qt::Unchecked);

      const QString fileName = QFileInfo(dicomPath).fileName();
      fileItem->setText(0, fileName);
      fileItem->setText(1, rt.modality);
      fileItem->setText(2, rt.frame);
      fileItem->setText(3, dicomPath);
      fileItem->setData(3, Qt::UserRole, dicomPath);
      fileItem->setData(3, Qt::UserRole + 3, dicomPath);
      fileItem->setData(3, Qt::UserRole + 1, rt.frame);

      if (isCalculatedDose) {
        fileItem->setForeground(0, QColor(0, 150, 0));
        fileItem->setForeground(1, QColor(0, 150, 0));
      }
    }
  };

  auto attachRtChildren = [&](QTreeWidgetItem *rootItem,
                              const QString &frameKey) {
    const QVector<int> indices = rtByFrame.take(frameKey);
    if (indices.isEmpty())
      return;
    for (int idx : indices)
      addRtItem(rtStudies[idx], rootItem);
    rootItem->setExpanded(true);
  };

  QSet<QString> seenStudyPrimaryKeys;

  auto addStudyItem = [&](const StudyRow &study, QTreeWidgetItem *parentItem,
                          bool makeCheckable,
                          Qt::CheckState initialState) -> QTreeWidgetItem * {
    const QStringList dicomFiles =
        dicomFilesForStudy(study.key, study.studyIds, study.path);
    int dicomCount = dicomFiles.size();
    QString primaryKey;
    if (!dicomFiles.isEmpty())
      primaryKey = normalizePathKey(dicomFiles.first());
    if (primaryKey.isEmpty())
      primaryKey = normalizePathKey(study.path);
    QString dedupKey =
        study.modality + QLatin1Char('\x1F') + primaryKey;
    if (seenStudyPrimaryKeys.contains(dedupKey))
      return nullptr;
    seenStudyPrimaryKeys.insert(dedupKey);

    QTreeWidgetItem *row = parentItem ? new QTreeWidgetItem(parentItem)
                                      : new QTreeWidgetItem();
    if (!parentItem)
      m_imageList->addTopLevelItem(row);

    if (makeCheckable) {
      row->setFlags(row->flags() | Qt::ItemIsUserCheckable);
      row->setCheckState(0, initialState);
    }

    QString firstDicomPath;
    if (!dicomFiles.isEmpty())
      firstDicomPath = dicomFiles.first();
    else if (!study.path.isEmpty())
      firstDicomPath = firstDicomFileIn(study.path);

    row->setText(0, studyDisplayLabel(study.modality, study.name, study.path,
                                      dicomCount, study.seriesDescription));
    row->setText(1, study.modality);
    row->setText(2, study.date);
    row->setText(3, study.path);
    row->setData(3, Qt::UserRole, study.path);
    const QString dbPath = !study.dbPath.isEmpty() ? study.dbPath : study.path;
    row->setData(3, Qt::UserRole + 3, dbPath);
    row->setData(3, Qt::UserRole + 1, study.frame);
    row->setData(3, Qt::UserRole + 2, firstDicomPath);
    return row;
  };

  QHash<QString, QVector<int>> frameToImageIndices;
  for (int i = 0; i < imageStudies.size(); ++i)
    frameToImageIndices[imageStudies[i].frame].append(i);

  QVector<bool> imageHandled(imageStudies.size(), false);

  auto handleImageStudy = [&](int index) {
    const StudyRow &study = imageStudies[index];
    const bool isCtOrMri =
        study.modality == QLatin1String("CT") ||
        study.modality == QLatin1String("MRI");
    const bool isFusion =
        study.modality.startsWith(QLatin1String("Fusion/"));
    Qt::CheckState initialState = Qt::Unchecked;
    if (isCtOrMri && !parent)
      initialState = Qt::Checked;
    const bool makeCheckable = isFusion || (isCtOrMri && parent);
    QTreeWidgetItem *row =
        addStudyItem(study, parent, makeCheckable, initialState);
    imageHandled[index] = true;
    if (row)
      attachRtChildren(row, study.frame);
  };

  auto handleGroupedStudies = [&](const QVector<int> &indices) {
    QVector<int> sorted = indices;
    std::sort(sorted.begin(), sorted.end(), [&](int lhs, int rhs) {
      auto orderRank = [&](const QString &modality) {
        if (modality == QLatin1String("CT"))
          return 0;
        if (modality == QLatin1String("MRI"))
          return 1;
        return 2;
      };
      const StudyRow &a = imageStudies[lhs];
      const StudyRow &b = imageStudies[rhs];
      const int ra = orderRank(a.modality);
      const int rb = orderRank(b.modality);
      if (ra != rb)
        return ra < rb;
      return lhs < rhs;
    });

    QTreeWidgetItem *rootItem = nullptr;
    int rootIndex = -1;
    for (int pos = 0; pos < sorted.size(); ++pos) {
      const int idx = sorted[pos];
      const StudyRow &study = imageStudies[idx];
      const bool isCtOrMri =
          study.modality == QLatin1String("CT") ||
          study.modality == QLatin1String("MRI");
      const bool isFusion =
          study.modality.startsWith(QLatin1String("Fusion/"));
      Qt::CheckState initialState = Qt::Unchecked;
      if (isCtOrMri && !rootItem)
        initialState = Qt::Checked;
      QTreeWidgetItem *parentItem = rootItem ? rootItem : parent;
      const bool makeCheckable = isCtOrMri || isFusion;
      QTreeWidgetItem *item =
          addStudyItem(study, parentItem, makeCheckable, initialState);
      imageHandled[idx] = true;
      if (!item)
        continue;
      if (!rootItem) {
        rootItem = item;
        rootIndex = idx;
        if (isCtOrMri && initialState == Qt::Unchecked)
          rootItem->setCheckState(0, Qt::Checked);
      }
    }

    if (rootItem && rootIndex >= 0)
      attachRtChildren(rootItem, imageStudies[rootIndex].frame);
  };

  for (int i = 0; i < imageStudies.size(); ++i) {
    if (imageHandled[i])
      continue;
    const StudyRow &study = imageStudies[i];
    QVector<int> group = frameToImageIndices.value(study.frame);
    bool hasCt = false;
    bool hasMri = false;
    bool hasFusion = false;
    bool hasNonFusionImaging = false;
    if (!study.frame.isEmpty() && group.size() > 1) {
      for (int idx : group) {
        const QString &mod = imageStudies[idx].modality;
        if (mod == QLatin1String("CT"))
          hasCt = true;
        else if (mod == QLatin1String("MRI"))
          hasMri = true;
        if (mod.startsWith(QLatin1String("Fusion/")))
          hasFusion = true;
        else
          hasNonFusionImaging = true;
      }
    }

    if ((hasCt && hasMri) || (hasFusion && hasNonFusionImaging)) {
      handleGroupedStudies(group);
    } else {
      handleImageStudy(i);
    }
  }

  auto handleOtherStudy = [&](const StudyRow &study) {
    QTreeWidgetItem *row =
        addStudyItem(study, parent, false, Qt::Unchecked);
    if (row)
      attachRtChildren(row, study.frame);
  };

  for (const StudyRow &study : otherStudies)
    handleOtherStudy(study);

  for (int i = 0; i < rtStudies.size(); ++i) {
    auto it = rtByFrame.find(rtStudies[i].frame);
    if (it == rtByFrame.end())
      continue;
    QVector<int> &indices = it.value();
    if (!indices.contains(i))
      continue;
    addRtItem(rtStudies[i], parent);
    indices.removeOne(i);
    if (indices.isEmpty())
      rtByFrame.erase(it);
  }
}

QStringList DataWindow::dicomFilesForStudy(const QString &groupKey,
                                          const QVector<int> &studyIds,
                                          const QString &studyPath) {
  if (groupKey.isEmpty() || studyIds.isEmpty())
    return {};

  if (m_dicomFilesCache.contains(groupKey))
    return m_dicomFilesCache.value(groupKey);

  QFileInfo baseInfo(studyPath);
  QDir baseDir = baseInfo.isDir() ? QDir(baseInfo.absoluteFilePath())
                                  : baseInfo.dir();
  if (!baseDir.exists()) {
    m_dicomFilesCache.insert(groupKey, {});
    return {};
  }

  QStringList orderedPaths;
  QSet<QString> seen;
  for (int id : studyIds) {
    if (id <= 0)
      continue;
    std::stringstream sq;
    sq << "SELECT relative_path FROM files WHERE study_id=" << id << ";";
    m_db.query(sq.str(), [&](int argc, char **argv, char **) {
      if (argc < 1 || !argv[0])
        return;
      const QString rel = QString::fromUtf8(argv[0]);
      const QString absPath =
          QDir::cleanPath(baseDir.absoluteFilePath(rel));
      if (absPath.isEmpty() || seen.contains(absPath))
        return;
      seen.insert(absPath);
      orderedPaths.append(absPath);
    });
  }

  std::sort(orderedPaths.begin(), orderedPaths.end(),
            [](const QString &a, const QString &b) {
              return QString::localeAwareCompare(a, b) < 0;
            });

  QStringList dicomFiles;
  dicomFiles.reserve(orderedPaths.size());
  for (const QString &absPath : std::as_const(orderedPaths)) {
    if (!absPath.isEmpty() && isDicomFile(absPath))
      dicomFiles.push_back(absPath);
  }

  m_dicomFilesCache.insert(groupKey, dicomFiles);
  return dicomFiles;
}

int DataWindow::dicomFileCountForStudy(const QString &groupKey,
                                       const QVector<int> &studyIds,
                                       const QString &studyPath) {
  if (groupKey.isEmpty() || studyIds.isEmpty())
    return -1;

  if (m_dicomCountCache.contains(groupKey))
    return m_dicomCountCache.value(groupKey);

  const QStringList dicomFiles =
      dicomFilesForStudy(groupKey, studyIds, studyPath);
  const int count = dicomFiles.size();
  m_dicomCountCache.insert(groupKey, count);
  return count;
}

void DataWindow::onOpenSelected() {
  auto items = m_imageList->selectedItems();
  if (items.isEmpty())
    return;
  auto *selected = items.first();
  QTreeWidgetItem *groupItem = selected->parent() ? selected->parent() : selected;

  struct ImagingSelection {
    QString modality;
    QString directory;
    QString frame;
  };
  QVector<ImagingSelection> imagingSelections;
  QSet<QString> seenImagingDirs;
  QSet<QString> imagingFrames;
  auto normalizedDirectoryForPath = [](const QString &path) -> QString {
    if (path.isEmpty())
      return {};
    QFileInfo info(path);
    QString dirPath;
    if (info.isDir())
      dirPath = info.absoluteFilePath();
    else if (info.exists())
      dirPath = info.dir().absolutePath();
    else
      dirPath = path;
    QDir dir(dirPath);
    QString cleaned = QDir::cleanPath(dir.absolutePath());
    if (cleaned == QLatin1String("."))
      cleaned.clear();
    return cleaned;
  };
  auto addImagingSelection = [&](const QString &modality, const QString &path,
                                 const QString &frame) {
    QString directory = normalizedDirectoryForPath(path);
    if (directory.isEmpty())
      return;
    const QString key = directory.toLower();
    if (seenImagingDirs.contains(key))
      return;
    seenImagingDirs.insert(key);
    imagingSelections.push_back({modality, directory, frame});
    if (!frame.isEmpty())
      imagingFrames.insert(frame);
  };
  QStringList rtssPaths;
  QStringList rtdosePaths;
  QStringList rtplanPaths;

  auto imagingFileForItem = [&](QTreeWidgetItem *item) -> QString {
    if (!item)
      return {};
    QString firstDicom = item->data(3, Qt::UserRole + 2).toString();
    if (!firstDicom.isEmpty())
      return firstDicom;
    QString path = item->data(3, Qt::UserRole).toString();
    if (path.isEmpty())
      return {};
    return firstDicomFileIn(path);
  };

  struct RtCandidate {
    QString modality;
    QString path;
    QString frame;
  };
  QVector<RtCandidate> branchRtCandidates;

  QSet<QString> seenStructPaths;
  QSet<QString> seenDosePaths;
  QSet<QString> seenPlanPaths;

  auto appendRtPath = [&](const QString &modality, const QString &path) {
    if (path.isEmpty())
      return;
    if (modality == QLatin1String("RTSTRUCT")) {
      if (seenStructPaths.contains(path))
        return;
      seenStructPaths.insert(path);
      rtssPaths << path;
    } else if (modality == QLatin1String("RTDOSE")) {
      if (seenDosePaths.contains(path))
        return;
      seenDosePaths.insert(path);
      rtdosePaths << path;
    } else if (modality == QLatin1String("RTPLAN")) {
      if (seenPlanPaths.contains(path))
        return;
      seenPlanPaths.insert(path);
      rtplanPaths << path;
    }
  };

  std::function<void(QTreeWidgetItem *)> collectBranch =
      [&](QTreeWidgetItem *item) {
        if (!item)
          return;
        const QString mod = item->text(1);
        const bool checkable = (item->flags() & Qt::ItemIsUserCheckable) != 0;
        const bool isChecked = !checkable || item->checkState(0) != Qt::Unchecked;

        if (isImagingModality(mod)) {
          const QString frame = item->data(3, Qt::UserRole + 1).toString();
          if (isChecked || item == selected) {
            if (!frame.isEmpty())
              imagingFrames.insert(frame);
          }
          if (isChecked || (item == selected && imagingSelections.isEmpty())) {
            const QString path = imagingFileForItem(item);
            if (!path.isEmpty())
              addImagingSelection(mod, path, frame);
          }
        } else if (mod.startsWith(QLatin1String("RT")) && isChecked) {
          QString path = item->data(3, Qt::UserRole).toString();
          if (!path.isEmpty()) {
            path = firstDicomFileIn(path);
            if (!path.isEmpty()) {
              RtCandidate entry{mod, path,
                                item->data(3, Qt::UserRole + 1).toString()};
              branchRtCandidates.push_back(entry);
            }
          }
        }

        for (int i = 0; i < item->childCount(); ++i)
          collectBranch(item->child(i));
      };

  collectBranch(groupItem);

  auto branchFrameMatches = [&](const QString &frame) {
    if (imagingFrames.isEmpty())
      return true;
    if (frame.isEmpty())
      return true;
    return imagingFrames.contains(frame);
  };

  for (const RtCandidate &entry : std::as_const(branchRtCandidates)) {
    if (!branchFrameMatches(entry.frame))
      continue;
    appendRtPath(entry.modality, entry.path);
  }

  if (!imagingFrames.isEmpty()) {
    auto globalFrameMatches = [&](const QString &frame) {
      return !frame.isEmpty() && imagingFrames.contains(frame);
    };

    std::function<void(QTreeWidgetItem *)> collectMatchingRt =
        [&](QTreeWidgetItem *item) {
          if (!item)
            return;
          const QString mod = item->text(1);
          const bool checkable =
              (item->flags() & Qt::ItemIsUserCheckable) != 0;
          const bool isChecked =
              !checkable || item->checkState(0) != Qt::Unchecked;
          if (mod.startsWith(QLatin1String("RT")) && isChecked) {
            const QString frame = item->data(3, Qt::UserRole + 1).toString();
            if (globalFrameMatches(frame)) {
              QString path = item->data(3, Qt::UserRole).toString();
              if (!path.isEmpty()) {
                path = firstDicomFileIn(path);
                appendRtPath(mod, path);
              }
            }
          }
          for (int i = 0; i < item->childCount(); ++i)
            collectMatchingRt(item->child(i));
        };

    const int topCount = m_imageList->topLevelItemCount();
    for (int i = 0; i < topCount; ++i)
      collectMatchingRt(m_imageList->topLevelItem(i));
  }

  if (imagingSelections.isEmpty()) {
    auto tryAddFromItem = [&](QTreeWidgetItem *item) {
      if (!item)
        return false;
      const QString mod = item->text(1);
      if (!isImagingModality(mod))
        return false;
      const QString path = imagingFileForItem(item);
      if (path.isEmpty())
        return false;
      const QString frame = item->data(3, Qt::UserRole + 1).toString();
      addImagingSelection(mod, path, frame);
      return !imagingSelections.isEmpty();
    };

    if (!tryAddFromItem(selected))
      tryAddFromItem(groupItem);
  }

  if (!imagingSelections.isEmpty()) {
    std::stable_sort(imagingSelections.begin(), imagingSelections.end(),
                     [](const ImagingSelection &lhs,
                        const ImagingSelection &rhs) {
                       auto orderRank = [](const QString &modality) {
                         if (modality == QLatin1String("CT"))
                           return 0;
                         if (modality == QLatin1String("MRI"))
                           return 1;
                         return 2;
                       };
                       const int ra = orderRank(lhs.modality);
                       const int rb = orderRank(rhs.modality);
                       if (ra != rb)
                         return ra < rb;
                       return false;
                     });
  }

  QStringList imageDirs;
  QStringList imageModalities;
  for (const ImagingSelection &sel : std::as_const(imagingSelections)) {
    imageDirs << sel.directory;
    imageModalities << sel.modality;
  }

  emit openStudyRequested(imageDirs, imageModalities, rtssPaths, rtdosePaths,
                          rtplanPaths);
}

void DataWindow::onRescan() {
  m_scanner.fullScanAndRepair();
  m_meta.writeGlobalIndex();
  refresh();
}

void DataWindow::onOpenDataFolder() {
  const QUrl url = QUrl::fromLocalFile(QString::fromStdString(m_db.dataRoot()));
  QDesktopServices::openUrl(url);
}

void DataWindow::onCreatePatient() {
  bool ok1 = false, ok2 = false;
  const QString name = QInputDialog::getText(this, "Create Patient",
                                             "Patient Name:", QLineEdit::Normal,
                                             QString(), &ok1);
  if (!ok1 || name.isEmpty())
    return;
  const QString id =
      QInputDialog::getText(this, "Create Patient",
                            "Patient ID:", QLineEdit::Normal, QString(), &ok2);
  if (!ok2 || id.isEmpty())
    return;

  FileStructureManager fsm(m_db);
  const std::string dir =
      fsm.ensurePatientFolderFor(name.toStdString(), id.toStdString());
  if (dir.empty()) {
    QMessageBox::warning(this, "Error", "Failed to create patient structure");
    return;
  }
  m_scanner.scanPath(dir);
  m_meta.writeGlobalIndex();
  refresh();
}

void DataWindow::onPatientSelected() {
  auto items = m_patientList->selectedItems();
  if (items.isEmpty())
    return;
  const QString key = items.first()->data(0, Qt::UserRole).toString();
  m_imageList->clear();
  loadStudies(key, nullptr);
}

void DataWindow::onSelectionChanged() { updatePreviewForSelected(); }

void DataWindow::closeEvent(QCloseEvent *event) {
  clearPreview();
  QWidget::closeEvent(event);
}

void DataWindow::clearPreview() {
  if (m_preview)
    m_preview->setPixmap(QPixmap());
  if (m_preview)
    m_preview->setText("No Preview");
  m_previewQueue.clear();
  m_previewIndex = 0;
  if (m_previewTimer)
    m_previewTimer->stop();
}

void DataWindow::updatePreviewForSelected() {
  clearPreview();

  // Skip preview updates if the window is currently hidden.
  if (!isVisible()) {
    qDebug() << "DataWindow is not visible, skipping preview update";
    return;
  }

  auto items = m_imageList->selectedItems();
  if (items.isEmpty())
    return;
  auto *it = items.first();
  QString path = it->data(3, Qt::UserRole).toString();
  if (path.isEmpty() && it->parent())
    path = it->parent()->data(3, Qt::UserRole).toString();
  if (path.isEmpty())
    return;

  // Build a queue of images: detect DICOM first, then common images
  QStringList dcmFiles;
  QStringList imgFiles;
  QDirIterator itFiles(path, QDir::Files, QDirIterator::Subdirectories);
  while (itFiles.hasNext()) {
    const QString f = itFiles.next();
    DcmFileFormat ff;
    if (ff.loadFile(f.toStdString().c_str()).good()) {
      dcmFiles << f;
    } else {
      QImage img;
      if (img.load(f))
        imgFiles << f;
    }
  }

  // Subsample: pick up to ~12 frames
  auto pushSubsample = [&](const QStringList &list) {
    if (list.isEmpty())
      return;
    int n = list.size();
    int target = qMin(12, n);
    int step = qMax(1, n / target);
    for (int i = 0; i < n; i += step)
      m_previewQueue.push_back(list.at(i));
  };
  pushSubsample(dcmFiles);
  if (m_previewQueue.isEmpty())
    pushSubsample(imgFiles);

  if (!m_previewQueue.isEmpty()) {
    qDebug() << "Starting preview timer with" << m_previewQueue.size() << "files";
    m_previewTimer->start();
  }
}

void DataWindow::buildListsForPatient(const QString &patientName) {
  m_imageList->clear();
  if (patientName.isEmpty())
    return;

  std::vector<QString> keys;
  std::stringstream qp;
  qp << "SELECT patient_key FROM patients WHERE name LIKE '%"
     << sqlEscape(patientName.toStdString()) << "%';";
  m_db.query(qp.str(), [&](int argc, char **argv, char **) {
    if (argc >= 1 && argv[0])
      keys.emplace_back(argv[0]);
  });
  if (keys.empty())
    return;

  for (const auto &key : keys) {
    loadStudies(key, nullptr);
  }
}

void DataWindow::onChangeDataFolder() {
  const QString dir = QFileDialog::getExistingDirectory(
      this, "Select Data Folder", QString::fromStdString(m_db.dataRoot()));
  if (dir.isEmpty())
    return;
  emit changeDataRootRequested(dir);
}
static QString sanitizeName(const QString &in) {
  QString s = in.trimmed();
  s.replace('/', '_');
  s.replace('\\', '_');
  if (s.isEmpty())
    s = "Imported";
  return s;
}

void DataWindow::onImportDicomFile() {
  const QString file = QFileDialog::getOpenFileName(
      this, "Import DICOM File", QDir::homePath(),
      "All Files (*.*);;DICOM Files (*.dcm *.DCM *.dicom *.DICOM)");
  if (file.isEmpty())
    return;

  QString name, id, modality, studyDate, studyDesc;
  {
    DcmFileFormat ff;
    if (ff.loadFile(file.toStdString().c_str()).bad()) {
      QMessageBox::warning(this, "Error", "Failed to read DICOM file");
      return;
    }
    DcmDataset *ds = ff.getDataset();
    OFString v;
    if (ds->findAndGetOFString(DCM_PatientName, v).good())
      name = QString::fromLatin1(v.c_str());
    if (ds->findAndGetOFString(DCM_PatientID, v).good())
      id = QString::fromLatin1(v.c_str());
    if (ds->findAndGetOFString(DCM_Modality, v).good())
      modality = QString::fromLatin1(v.c_str());
    if (ds->findAndGetOFString(DCM_StudyDate, v).good())
      studyDate = QString::fromLatin1(v.c_str());
    if (ds->findAndGetOFString(DCM_StudyDescription, v).good())
      studyDesc = QString::fromLatin1(v.c_str());
  }
  if (studyDate.isEmpty())
    studyDate = QDate::currentDate().toString("yyyyMMdd");
  studyDesc = sanitizeName(studyDesc);

  FileStructureManager fsm(m_db);
  const std::string patientDir =
      fsm.ensurePatientFolderFor(name.toStdString(), id.toStdString());
  if (patientDir.empty()) {
    QMessageBox::warning(this, "Error", "Failed to prepare patient folder");
    return;
  }

  QString target;
  if (modality.compare("CT", Qt::CaseInsensitive) == 0 ||
      modality.compare("MR", Qt::CaseInsensitive) == 0 ||
      modality.compare("MRI", Qt::CaseInsensitive) == 0 ||
      modality.compare("PT", Qt::CaseInsensitive) == 0 ||
      modality.compare("PET", Qt::CaseInsensitive) == 0) {
    QString mod = modality.toUpper();
    if (mod == "MR")
      mod = "MRI";
    if (mod == "PT")
      mod = "PET";
    target = QString::fromStdString(patientDir) + "/Images/" + mod + "/" +
             studyDate + "_" + studyDesc;
  } else if (modality.startsWith("RT", Qt::CaseInsensitive)) {
    QString cat = "Analysis";
    if (modality.compare("RTSTRUCT", Qt::CaseInsensitive) == 0)
      cat = "Structures";
    else if (modality.compare("RTDOSE", Qt::CaseInsensitive) == 0)
      cat = "Doses";
    else if (modality.compare("RTPLAN", Qt::CaseInsensitive) == 0)
      cat = "Plans";
    target = QString::fromStdString(patientDir) + "/RT_Data/" + cat + "/" +
             studyDate + "_" + studyDesc;
  } else {
    target = QString::fromStdString(patientDir) + "/Images/Others/" +
             studyDate + "_" + studyDesc;
  }

  std::error_code ec;
  std::filesystem::create_directories(target.toStdString(), ec);
  const QString dest = target + "/" + QFileInfo(file).fileName();
  std::filesystem::copy_file(file.toStdString(), dest.toStdString(),
                             std::filesystem::copy_options::overwrite_existing,
                             ec);

  m_scanner.scanPath(target.toStdString());
  m_meta.writeGlobalIndex();
  refresh();
}

void DataWindow::onImportDicomFolder() {
  const QString dir = QFileDialog::getExistingDirectory(
      this, "Import DICOM Folder", QDir::homePath());
  if (dir.isEmpty())
    return;
  QDirIterator it(dir, QDir::Files, QDirIterator::Subdirectories);
  QStringList allFiles;
  while (it.hasNext())
    allFiles << it.next();
  QStringList files;
  for (const QString &f : allFiles) {
    DcmFileFormat ff;
    if (ff.loadFile(f.toStdString().c_str()).good())
      files << f;
  }
  if (files.isEmpty()) {
    QMessageBox::warning(this, "Error", "No DICOM files in folder");
    return;
  }

  QString name, id;
  {
    DcmFileFormat ff;
    if (ff.loadFile(files.first().toStdString().c_str()).bad()) {
      QMessageBox::warning(this, "Error", "Failed to read DICOM");
      return;
    }
    DcmDataset *ds = ff.getDataset();
    OFString v;
    if (ds->findAndGetOFString(DCM_PatientName, v).good())
      name = QString::fromLatin1(v.c_str());
    if (ds->findAndGetOFString(DCM_PatientID, v).good())
      id = QString::fromLatin1(v.c_str());
  }

  FileStructureManager fsm(m_db);
  const std::string patientDir =
      fsm.ensurePatientFolderFor(name.toStdString(), id.toStdString());

  QSet<QString> targets;
  for (const QString &f : files) {
    QString modality, studyDate, studyDesc;
    {
      DcmFileFormat ff;
      if (ff.loadFile(f.toStdString().c_str()).bad())
        continue;
      DcmDataset *ds = ff.getDataset();
      OFString v;
      if (ds->findAndGetOFString(DCM_Modality, v).good())
        modality = QString::fromLatin1(v.c_str());
      if (ds->findAndGetOFString(DCM_StudyDate, v).good())
        studyDate = QString::fromLatin1(v.c_str());
      if (ds->findAndGetOFString(DCM_StudyDescription, v).good())
        studyDesc = QString::fromLatin1(v.c_str());
    }
    if (studyDate.isEmpty())
      studyDate = QDate::currentDate().toString("yyyyMMdd");
    if (studyDesc.isEmpty())
      studyDesc = QFileInfo(f).dir().dirName();
    studyDesc = sanitizeName(studyDesc);

    QString mod = modality.toUpper();
    if (mod == "MR")
      mod = "MRI";
    if (mod == "PT")
      mod = "PET";
    QString target;
    if (mod == "CT" || mod == "MRI" || mod == "PET") {
      target = QString::fromStdString(patientDir) + "/Images/" + mod + "/" +
               studyDate + "_" + studyDesc;
    } else if (mod.startsWith("RT")) {
      QString cat = "Analysis";
      if (mod == "RTSTRUCT")
        cat = "Structures";
      else if (mod == "RTDOSE")
        cat = "Doses";
      else if (mod == "RTPLAN")
        cat = "Plans";
      target = QString::fromStdString(patientDir) + "/RT_Data/" + cat + "/" +
               studyDate + "_" + studyDesc;
    } else {
      target = QString::fromStdString(patientDir) + "/Images/Others/" +
               studyDate + "_" + studyDesc;
    }

    std::error_code ec;
    std::filesystem::create_directories(target.toStdString(), ec);
    const QString dest = target + "/" + QFileInfo(f).fileName();
    std::filesystem::copy_file(
        f.toStdString(), dest.toStdString(),
        std::filesystem::copy_options::overwrite_existing, ec);
    targets.insert(target);
  }

  for (const QString &t : targets) {
    m_scanner.scanPath(t.toStdString());
  }
  m_meta.writeGlobalIndex();
  refresh();
}

void DataWindow::onImportImageFiles() {
  const QStringList files = QFileDialog::getOpenFileNames(
      this, "Import Images", QDir::homePath(),
      "All Files (*.*);;Images (*.png *.jpg *.jpeg *.bmp)");
  if (files.isEmpty())
    return;

  bool ok1 = false, ok2 = false;
  const QString name = QInputDialog::getText(this, "Target Patient",
                                             "Patient Name:", QLineEdit::Normal,
                                             QString(), &ok1);
  if (!ok1 || name.isEmpty())
    return;
  const QString id =
      QInputDialog::getText(this, "Target Patient",
                            "Patient ID:", QLineEdit::Normal, QString(), &ok2);
  if (!ok2 || id.isEmpty())
    return;
  const QString label = sanitizeName(QInputDialog::getText(
      this, "Study Label", "Label:", QLineEdit::Normal, "Imported"));

  FileStructureManager fsm(m_db);
  const std::string patientDir =
      fsm.ensurePatientFolderFor(name.toStdString(), id.toStdString());
  const QString date = QDate::currentDate().toString("yyyyMMdd");
  const QString target = QString::fromStdString(patientDir) +
                         "/Images/Others/" + date + "_" + label;
  std::error_code ec;
  std::filesystem::create_directories(target.toStdString(), ec);
  for (const QString &f : files) {
    const QString dest = target + "/" + QFileInfo(f).fileName();
    std::filesystem::copy_file(
        f.toStdString(), dest.toStdString(),
        std::filesystem::copy_options::overwrite_existing, ec);
  }
  m_scanner.scanPath(target.toStdString());
  m_meta.writeGlobalIndex();
  refresh();
}

void DataWindow::onDeleteSelected() {
  // If an image study is selected, delete that; else if a patient is selected,
  // delete patient
  auto studySel =
      m_imageList ? m_imageList->selectedItems() : QList<QTreeWidgetItem *>();
  if (!studySel.isEmpty()) {
    auto *it = studySel.first();
    QString fsPath = it->data(3, Qt::UserRole).toString();
    const QString dbPath = it->data(3, Qt::UserRole + 3).toString();
    if (fsPath.isEmpty())
      fsPath = dbPath;
    if (fsPath.isEmpty())
      return;
    const QString sqlPath = !dbPath.isEmpty() ? dbPath : fsPath;
    const auto ans = QMessageBox::question(
        this, "Delete Study",
        QString("Delete study folder and DB entry?\n%1").arg(fsPath));
    if (ans != QMessageBox::Yes)
      return;
    std::error_code ec;
    std::filesystem::remove_all(fsPath.toStdString(), ec);
    std::stringstream s1;
    s1 << "DELETE FROM studies WHERE path='" << sqlEscape(sqlPath.toStdString())
       << "';";
    m_db.exec(s1.str());
    refresh();
    return;
  }
  auto patSel = m_patientList ? m_patientList->selectedItems()
                              : QList<QTreeWidgetItem *>();
  if (!patSel.isEmpty()) {
    auto *it = patSel.first();
    const QString patientKey = it->data(0, Qt::UserRole).toString();
    if (patientKey.isEmpty())
      return;
    const auto ans = QMessageBox::question(
        this, "Delete Patient",
        QString("Delete patient '%1' and all data? This cannot be undone.")
            .arg(it->text(0)));
    if (ans != QMessageBox::Yes)
      return;
    const QString patientDir =
        QString::fromStdString(m_db.dataRoot()) + "/Patients/" + patientKey;
    std::error_code ec;
    std::filesystem::remove_all(patientDir.toStdString(), ec);
    std::stringstream ss;
    ss << "DELETE FROM patients WHERE patient_key='"
       << sqlEscape(patientKey.toStdString()) << "';";
    m_db.exec(ss.str());
    refresh();
  }
}
