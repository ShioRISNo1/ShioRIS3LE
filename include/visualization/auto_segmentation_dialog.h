//=============================================================================
// ファイル: include/visualization/auto_segmentation_dialog.h
// 修正内容: AIセグメンテーションダイアログの完全実装
//=============================================================================

#ifndef AUTO_SEGMENTATION_DIALOG_H
#define AUTO_SEGMENTATION_DIALOG_H

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QProgressBar>
#include <QCheckBox>
#include <QLineEdit>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QGroupBox>
#include <QTextEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QScrollArea>
#include <QSplitter>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QHeaderView>
#include <QApplication>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>

#ifdef USE_ONNXRUNTIME
#include "ai/onnx_segmenter.h"
#endif

/**
 * @brief AIセグメンテーション実行のための統合ダイアログ
 * 
 * 機能:
 * - ONNXモデルの選択・ロード
 * - セグメンテーション実行
 * - 進捗表示
 * - 結果プレビュー
 * - 結果の調整・編集
 * - エクスポート機能
 */
class AutoSegmentationDialog : public QDialog {
    Q_OBJECT

public:
    explicit AutoSegmentationDialog(QWidget *parent = nullptr);
    ~AutoSegmentationDialog() = default;

    /**
     * @brief セグメンテーション開始（外部から呼び出し）
     * @param volume 入力ボリューム（3Dデータ）
     */
    void startSegmentation(const cv::Mat &volume);

    /**
     * @brief 既存のセグメンテーションモデルを共有利用する
     * @param segmenter 外部でロード済みのOnnxSegmenter
     */
    void setSharedSegmenter(OnnxSegmenter *segmenter);

    /**
     * @brief 現在のセグメンテーション結果を取得
     * @return セグメンテーション結果（3Dマスク）
     */
    cv::Mat getSegmentationResult() const { return m_segmentationResult; }

    /**
     * @brief セグメンテーション結果が利用可能かチェック
     * @return 結果がある場合true
     */
    bool hasValidResult() const { return !m_segmentationResult.empty(); }

public slots:
    /**
     * @brief 進捗更新スロット
     * @param progress 進捗率（0.0-1.0）
     */
    void onProgressUpdate(float progress);

    /**
     * @brief セグメンテーション完了スロット
     * @param result セグメンテーション結果
     */
    void onSegmentationCompleted(const cv::Mat &result);

    /**
     * @brief エラー発生時のスロット
     * @param errorMessage エラーメッセージ
     */
    void onError(const QString &errorMessage);

signals:
    /**
     * @brief セグメンテーション開始シグナル
     * @param volume 入力ボリューム
     */
    void segmentationStarted(const cv::Mat &volume);

    /**
     * @brief セグメンテーション完了シグナル
     * @param result セグメンテーション結果
     */
    void segmentationFinished(const cv::Mat &result);

    /**
     * @brief 結果を適用シグナル
     * @param result 適用するセグメンテーション結果
     */
    void applyResult(const cv::Mat &result);

private slots:
    /**
     * @brief モデルファイル選択
     */
    void selectModelFile();

    /**
     * @brief セグメンテーション開始
     */
    void startSegmentationProcess();

    /**
     * @brief セグメンテーション停止
     */
    void stopSegmentation();

    /**
     * @brief 結果をメインビューに適用
     */
    void applySegmentationResult();

    /**
     * @brief 結果をファイルにエクスポート
     */
    void exportResult();

    /**
     * @brief プレビュー更新
     */
    void updatePreview();

    /**
     * @brief ラベル表示切り替え
     * @param item 切り替えられたアイテム
     * @param column カラム
     */
    void onLabelVisibilityChanged(QTreeWidgetItem *item, int column);

    /**
     * @brief しきい値調整
     */
    void onThresholdChanged();

private:
    OnnxSegmenter* activeSegmenter() const;
    bool hasLoadedModel() const;

    /**
     * @brief UI初期化
     */
    void setupUI();

    /**
     * @brief モデル設定セクション作成
     */
    QWidget* createModelSection();

    /**
     * @brief 実行制御セクション作成
     */
    QWidget* createExecutionSection();

    /**
     * @brief 結果表示セクション作成
     */
    QWidget* createResultSection();

    /**
     * @brief プレビューセクション作成
     */
    QWidget* createPreviewSection();

    /**
     * @brief 結果調整セクション作成
     */
    QWidget* createAdjustmentSection();

    /**
     * @brief アクションボタンセクション作成
     */
    QWidget* createActionSection();

    /**
     * @brief セグメンテーションワーカースレッド開始
     */
    void startWorkerThread();

    /**
     * @brief セグメンテーションワーカースレッド停止
     */
    void stopWorkerThread();

    /**
     * @brief 結果統計情報更新
     */
    void updateResultStatistics();

    /**
     * @brief 臓器ラベルツリー初期化
     */
    void initializeOrganLabels();

    /**
     * @brief UIの有効/無効状態更新
     */
    void updateUIState();

    /**
     * @brief プレビュー画像生成
     */
    QPixmap generatePreviewImage();

    OnnxSegmenter *m_sharedSegmenter{nullptr};
    // UIコンポーネント
    QVBoxLayout *m_mainLayout;
    QSplitter *m_mainSplitter;
    
    // モデル設定
    QGroupBox *m_modelGroupBox;
    QLineEdit *m_modelPathEdit;
    QPushButton *m_selectModelButton;
    QLabel *m_modelStatusLabel;
    
    // 実行制御
    QGroupBox *m_executionGroupBox;
    QComboBox *m_qualityModeComboBox;
    QLabel *m_qualityModeLabel;
    QPushButton *m_startButton;
    QPushButton *m_stopButton;
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;
    QTextEdit *m_logTextEdit;
    
    // 結果表示
    QGroupBox *m_resultGroupBox;
    QTreeWidget *m_organTree;
    QLabel *m_statisticsLabel;
    
    // プレビュー
    QGroupBox *m_previewGroupBox;
    QLabel *m_previewLabel;
    QScrollArea *m_previewScrollArea;
    QSpinBox *m_sliceSpinBox;
    QLabel *m_sliceLabel;
    
    // 結果調整
    QGroupBox *m_adjustmentGroupBox;
    QSpinBox *m_thresholdSpinBox;
    QCheckBox *m_smoothingCheckBox;
    QCheckBox *m_fillHolesCheckBox;
    
    // アクション
    QHBoxLayout *m_actionLayout;
    QPushButton *m_applyButton;
    QPushButton *m_exportButton;
    QPushButton *m_cancelButton;

    // データとロジック
#ifdef USE_ONNXRUNTIME
    std::unique_ptr<OnnxSegmenter> m_segmenter;
#endif
    cv::Mat m_inputVolume;
    cv::Mat m_segmentationResult;
    cv::Mat m_adjustedResult;
    
    // スレッド制御
    QThread *m_workerThread;
    QMutex m_processingMutex;
    QWaitCondition m_processingCondition;
    bool m_isProcessing;
    bool m_shouldStop;
    
    // UI状態
    bool m_modelLoaded;
    int m_currentSlice;
    float m_currentProgress;
    
    // 統計情報
    struct OrganStatistics {
        QString name;
        int voxelCount;
        double volumeCm3;
        double percentage;
        bool visible;
        QColor color;
    };
    
    std::vector<OrganStatistics> m_organStats;
};

#endif // AUTO_SEGMENTATION_DIALOG_H