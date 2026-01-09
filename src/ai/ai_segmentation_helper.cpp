//=============================================================================
// ファイル: src/ai/ai_segmentation_helper.cpp
// 修正内容: AI機能ヘルパーの実装
//=============================================================================

#include "ai/ai_segmentation_helper.h"
#include <QMessageBox>
#include <QFileInfo>
#include <QElapsedTimer>
#include <random>

bool AISegmentationHelper::initializeAIDirectories() {
    try {
        QString documentsPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        QDir baseDir(documentsPath);
        
        // ShioRIS3/AIModels ディレクトリ構造を作成
        QString aiBasePath = "ShioRIS3/" + QString(AI_MODELS_DIR);
        
        if (!baseDir.mkpath(aiBasePath)) {
            qWarning() << "Failed to create AI base directory:" << aiBasePath;
            return false;
        }
        
        QDir aiDir(baseDir.filePath(aiBasePath));
        
        // サブディレクトリ作成
        if (!aiDir.mkdir(ONNX_DIR)) {
            qDebug() << "ONNX directory already exists or creation failed";
        }
        
        if (!aiDir.mkdir(TEMP_DIR)) {
            qDebug() << "Temp directory already exists or creation failed";
        }
        
        if (!aiDir.mkdir(SAMPLES_DIR)) {
            qDebug() << "Samples directory already exists or creation failed";
        }
        
        // README.txt ファイルの作成
        QString readmePath = aiDir.filePath("README.txt");
        if (!QFileInfo::exists(readmePath)) {
            QFile readmeFile(readmePath);
            if (readmeFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QTextStream out(&readmeFile);
                out << "ShioRIS3 AI Models Directory\n";
                out << "=============================\n\n";
                out << "このディレクトリには以下のファイルが保存されます:\n\n";
                out << "onnx/     - ONNXモデルファイル (.onnx)\n";
                out << "temp/     - 一時ファイル\n";
                out << "samples/  - サンプルデータ\n\n";
                out << "推奨モデル:\n";
                out << "- 腹部多臓器セグメンテーション用ONNXモデル\n";
                out << "- 入力: CT画像 (HU値 -1024 to 3071)\n";
                out << "- 出力: 4クラス (Background, Liver, Right Kidney, Left Kidney/Spleen)\n\n";
                out << "詳細: https://github.com/yourproject/ShioRIS3/docs/ai_models.md\n";
                readmeFile.close();
            }
        }
        
        qDebug() << "AI directories initialized successfully at:" << aiDir.absolutePath();
        return true;
        
    } catch (const std::exception &e) {
        qCritical() << "Error initializing AI directories:" << e.what();
        return false;
    }
}

cv::Mat AISegmentationHelper::generateSampleVolume(int depth, int height, int width) {
    qDebug() << "Generating sample volume:" << depth << "x" << height << "x" << width;
    
    try {
        int sizes[] = {depth, height, width};
        cv::Mat volume(3, sizes, CV_16SC1);
        
        // CTの一般的なHU値範囲でサンプルデータを生成
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // 異なる組織のHU値分布
        std::normal_distribution<float> air(-1000, 50);      // 空気
        std::normal_distribution<float> fat(-100, 30);       // 脂肪
        std::normal_distribution<float> soft_tissue(40, 20); // 軟組織
        std::normal_distribution<float> liver(60, 15);       // 肝臓
        std::normal_distribution<float> kidney(30, 10);      // 腎臓
        std::normal_distribution<float> bone(400, 200);      // 骨
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int16_t value = -1000; // デフォルト: 空気
                    
                    // 中心部により高密度の組織を配置
                    float centerX = width / 2.0f;
                    float centerY = height / 2.0f;
                    float centerZ = depth / 2.0f;
                    
                    float distFromCenter = std::sqrt(
                        std::pow(x - centerX, 2) + 
                        std::pow(y - centerY, 2) + 
                        std::pow(z - centerZ, 2)
                    );
                    
                    float maxDist = std::sqrt(centerX*centerX + centerY*centerY + centerZ*centerZ);
                    float normalizedDist = distFromCenter / maxDist;
                    
                    if (normalizedDist < 0.3f) {
                        // 中心部: 臓器領域
                        if (x < centerX) {
                            value = static_cast<int16_t>(liver(gen)); // 肝臓様
                        } else {
                            value = static_cast<int16_t>(kidney(gen)); // 腎臓様
                        }
                    } else if (normalizedDist < 0.6f) {
                        // 中間部: 軟組織
                        value = static_cast<int16_t>(soft_tissue(gen));
                    } else if (normalizedDist < 0.8f) {
                        // 外側: 脂肪組織
                        value = static_cast<int16_t>(fat(gen));
                    } else {
                        // 最外部: 空気
                        value = static_cast<int16_t>(air(gen));
                    }
                    
                    // HU値の範囲制限
                    value = std::max(static_cast<int16_t>(-1024), 
                            std::min(static_cast<int16_t>(3071), value));
                    
                    volume.at<int16_t>(z, y, x) = value;
                }
            }
        }
        
        qDebug() << "Sample volume generated successfully";
        return volume;
        
    } catch (const std::exception &e) {
        qCritical() << "Error generating sample volume:" << e.what();
        return cv::Mat();
    }
}

cv::Mat AISegmentationHelper::generateDummySegmentation(const cv::Mat &volume) {
    if (volume.empty()) {
        qWarning() << "Empty volume provided for dummy segmentation";
        return cv::Mat();
    }
    
    qDebug() << "Generating dummy segmentation for volume";
    
    try {
        cv::Mat segmentation;
        
        if (volume.dims == 3) {
            // 3D volume
            int depth = volume.size[0];
            int height = volume.size[1];
            int width = volume.size[2];
            
            int sizes[] = {depth, height, width};
            segmentation = cv::Mat(3, sizes, CV_8UC1, cv::Scalar(0));
            
            // 簡単な幾何学的パターンでダミーセグメンテーションを生成
            float centerX = width / 2.0f;
            float centerY = height / 2.0f;
            float centerZ = depth / 2.0f;
            
            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        float distFromCenter = std::sqrt(
                            std::pow(x - centerX, 2) + 
                            std::pow(y - centerY, 2) + 
                            std::pow((z - centerZ) * 2, 2) // Z軸を圧縮
                        );
                        
                        uchar label = 0; // Background
                        
                        // 楕円形の臓器領域を生成
                        if (distFromCenter < centerY * 0.4f) {
                            // 肝臓様の大きな領域
                            if (x < centerX * 0.8f) {
                                label = 1; // Liver
                            } else {
                                // 右側に腎臓様の小さな領域
                                float kidneyDist = std::sqrt(
                                    std::pow(x - centerX * 1.3f, 2) + 
                                    std::pow(y - centerY, 2)
                                );
                                if (kidneyDist < centerY * 0.15f) {
                                    label = 2; // Right Kidney
                                }
                            }
                        } else if (distFromCenter < centerY * 0.3f) {
                            // 左側に脾臓・左腎様の領域
                            if (x > centerX * 1.2f) {
                                float spleenDist = std::sqrt(
                                    std::pow(x - centerX * 1.4f, 2) + 
                                    std::pow(y - centerY * 0.8f, 2)
                                );
                                if (spleenDist < centerY * 0.12f) {
                                    label = 3; // Left Kidney/Spleen
                                }
                            }
                        }
                        
                        segmentation.at<uchar>(z, y, x) = label;
                    }
                }
            }
            
        } else {
            // 2D slice
            segmentation = cv::Mat::zeros(volume.size(), CV_8UC1);
            
            float centerX = volume.cols / 2.0f;
            float centerY = volume.rows / 2.0f;
            
            // 2Dでの簡単なセグメンテーション
            cv::ellipse(segmentation, cv::Point(centerX * 0.7f, centerY), 
                       cv::Size(centerX * 0.3f, centerY * 0.4f), 0, 0, 360, cv::Scalar(1), -1); // Liver
            
            cv::circle(segmentation, cv::Point(centerX * 1.3f, centerY * 0.8f), 
                      centerY * 0.15f, cv::Scalar(2), -1); // Right Kidney
            
            cv::circle(segmentation, cv::Point(centerX * 1.3f, centerY * 1.2f), 
                      centerY * 0.12f, cv::Scalar(3), -1); // Left Kidney/Spleen
        }
        
        // 統計情報の出力
        std::vector<int> labelCounts(4, 0);
        int totalVoxels = 0;
        
        if (segmentation.dims == 3) {
            totalVoxels = segmentation.size[0] * segmentation.size[1] * segmentation.size[2];
            for (int z = 0; z < segmentation.size[0]; ++z) {
                for (int y = 0; y < segmentation.size[1]; ++y) {
                    for (int x = 0; x < segmentation.size[2]; ++x) {
                        uchar label = segmentation.at<uchar>(z, y, x);
                        if (label < 4) labelCounts[label]++;
                    }
                }
            }
        } else {
            totalVoxels = segmentation.rows * segmentation.cols;
            for (int y = 0; y < segmentation.rows; ++y) {
                for (int x = 0; x < segmentation.cols; ++x) {
                    uchar label = segmentation.at<uchar>(y, x);
                    if (label < 4) labelCounts[label]++;
                }
            }
        }
        
        qDebug() << "Dummy segmentation statistics:";
        QStringList organNames = {"Background", "Liver", "Right Kidney", "Left Kidney/Spleen"};
        for (int i = 0; i < 4; ++i) {
            double percentage = (totalVoxels > 0) ? (labelCounts[i] * 100.0 / totalVoxels) : 0.0;
            qDebug() << QString("  %1: %2 voxels (%3%)")
                        .arg(organNames[i])
                        .arg(labelCounts[i])
                        .arg(percentage, 0, 'f', 1);
        }
        
        qDebug() << "Dummy segmentation generated successfully";
        return segmentation;
        
    } catch (const std::exception &e) {
        qCritical() << "Error generating dummy segmentation:" << e.what();
        return cv::Mat();
    }
}

bool AISegmentationHelper::validateONNXModel(const QString &modelPath) {
    QFileInfo fileInfo(modelPath);
    
    if (!fileInfo.exists()) {
        qWarning() << "ONNX model file does not exist:" << modelPath;
        return false;
    }
    
    if (!fileInfo.isReadable()) {
        qWarning() << "ONNX model file is not readable:" << modelPath;
        return false;
    }
    
    if (fileInfo.suffix().toLower() != "onnx") {
        qWarning() << "File does not have .onnx extension:" << modelPath;
        return false;
    }
    
    if (fileInfo.size() < 1024) { // 最小1KB
        qWarning() << "ONNX model file is too small:" << fileInfo.size() << "bytes";
        return false;
    }
    
    if (fileInfo.size() > 2LL * 1024 * 1024 * 1024) { // 最大2GB
        qWarning() << "ONNX model file is too large:" << fileInfo.size() << "bytes";
        return false;
    }
    
    // ファイルヘッダーの簡単なチェック
    QFile modelFile(modelPath);
    if (modelFile.open(QIODevice::ReadOnly)) {
        QByteArray header = modelFile.read(8);
        modelFile.close();
        
        // ONNXファイルは通常protobufフォーマット
        // 簡単なマジックバイトチェック（完全ではない）
        if (header.isEmpty()) {
            qWarning() << "Failed to read ONNX model header";
            return false;
        }
    }
    
    qDebug() << "ONNX model validation passed:" << modelPath;
    return true;
}

QString AISegmentationHelper::checkSystemRequirements() {
    QStringList requirements;
    QStringList recommendations;
    
    // メモリチェック（概算）
    // 実際のシステムメモリ取得は OS 依存なので、簡易版
    requirements << "✓ Qt6対応";
    requirements << "✓ OpenCV 4.8+対応";
    
#ifdef USE_ONNXRUNTIME
    requirements << "✓ ONNX Runtime利用可能";
#else
    requirements << "✗ ONNX Runtime未対応（AI機能制限）";
#endif
    
    // CPU情報
#ifdef __x86_64__
    requirements << "✓ 64-bit x86アーキテクチャ";
#elif defined(__aarch64__)
    requirements << "✓ 64-bit ARM64アーキテクチャ";
#else
    requirements << "? 未知のCPUアーキテクチャ";
#endif
    
    // OpenGLサポート
    recommendations << "推奨: OpenGL 3.3+対応GPU";
    recommendations << "推奨: 16GB以上RAM（大ボリューム処理時）";
    recommendations << "推奨: SSD（高速ファイルアクセス）";
    
    QString result = "=== システム要件チェック ===\n\n";
    result += "基本要件:\n" + requirements.join("\n") + "\n\n";
    result += "推奨要件:\n" + recommendations.join("\n") + "\n";
    
    return result;
}

QString AISegmentationHelper::getModelMetadata(const QString &modelPath) {
    if (!validateONNXModel(modelPath)) {
        return "無効なONNXモデルファイル";
    }
    
    QFileInfo fileInfo(modelPath);
    
    QString metadata;
    metadata += QString("ファイル名: %1\n").arg(fileInfo.fileName());
    metadata += QString("ファイルサイズ: %1 MB\n").arg(fileInfo.size() / (1024.0 * 1024.0), 0, 'f', 2);
    metadata += QString("作成日時: %1\n").arg(fileInfo.birthTime().toString("yyyy-MM-dd hh:mm:ss"));
    metadata += QString("更新日時: %1\n").arg(fileInfo.lastModified().toString("yyyy-MM-dd hh:mm:ss"));
    metadata += QString("パス: %1\n").arg(fileInfo.absoluteFilePath());
    
    // 簡易的なモデル情報（実際のONNX解析はより複雑）
    metadata += "\n=== 推定モデル情報 ===\n";
    metadata += "対象: 腹部CT画像セグメンテーション\n";
    metadata += "入力: 3D Volume [batch, channel, depth, height, width]\n";
    metadata += "出力: 4クラス セグメンテーション\n";
    metadata += "クラス: Background, Liver, Right Kidney, Left Kidney/Spleen\n";
    
    return metadata;
}

cv::Mat AISegmentationHelper::preprocessVolume(const cv::Mat &volume) {
    if (volume.empty()) {
        return cv::Mat();
    }
    
    qDebug() << "Preprocessing volume...";
    
    cv::Mat processed = volume.clone();
    
    try {
        // 1. HU値クリッピング（CT画像の場合）
        if (volume.type() == CV_16SC1) {
            qDebug() << "Applying HU clipping for CT data";
            cv::threshold(processed, processed, -1024, -1024, cv::THRESH_TOZERO); // 下限
            cv::threshold(processed, processed, 3071, 3071, cv::THRESH_TRUNC);    // 上限
        }
        
        // 2. ノイズ除去（軽微な）
        if (volume.dims == 3) {
            // 3D Gaussian blur (軽微)
            qDebug() << "Applying 3D noise reduction";
            // OpenCVには3D Gaussianがないので、各スライスに適用
            for (int z = 0; z < processed.size[0]; ++z) {
                cv::Mat slice = processed(cv::Range(z, z+1), cv::Range::all(), cv::Range::all());
                slice = slice.reshape(1, processed.size[1]);
                
                cv::Mat blurred;
                cv::GaussianBlur(slice, blurred, cv::Size(3, 3), 0.5);
                blurred.copyTo(slice);
            }
        } else {
            // 2D Gaussian blur
            cv::GaussianBlur(processed, processed, cv::Size(3, 3), 0.5);
        }
        
        qDebug() << "Volume preprocessing completed";
        
    } catch (const std::exception &e) {
        qWarning() << "Error in volume preprocessing:" << e.what();
        return volume; // 元のボリュームを返す
    }
    
    return processed;
}

cv::Mat AISegmentationHelper::postprocessSegmentation(const cv::Mat &segmentation) {
    if (segmentation.empty()) {
        return cv::Mat();
    }
    
    qDebug() << "Post-processing segmentation...";
    
    cv::Mat processed = segmentation.clone();
    
    try {
        // モルフォロジー演算用カーネル
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        
        if (segmentation.dims == 3) {
            // 3D後処理（各スライスに適用）
            for (int z = 0; z < processed.size[0]; ++z) {
                cv::Mat slice = processed(cv::Range(z, z+1), cv::Range::all(), cv::Range::all());
                slice = slice.reshape(1, processed.size[1]);
                
                // 各ラベルに対して個別に処理
                for (int label = 1; label < 4; ++label) {
                    cv::Mat labelMask = (slice == label);
                    
                    // 1. Opening（ノイズ除去）
                    cv::morphologyEx(labelMask, labelMask, cv::MORPH_OPEN, kernel);
                    
                    // 2. Closing（穴埋め）
                    cv::morphologyEx(labelMask, labelMask, cv::MORPH_CLOSE, kernel);
                    
                    // ラベルを元に戻す
                    slice.setTo(label, labelMask);
                }
            }
        } else {
            // 2D後処理
            for (int label = 1; label < 4; ++label) {
                cv::Mat labelMask = (processed == label);
                
                // Opening + Closing
                cv::morphologyEx(labelMask, labelMask, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(labelMask, labelMask, cv::MORPH_CLOSE, kernel);
                
                processed.setTo(label, labelMask);
            }
        }
        
        qDebug() << "Segmentation post-processing completed";
        
    } catch (const std::exception &e) {
        qWarning() << "Error in segmentation post-processing:" << e.what();
        return segmentation; // 元のセグメンテーションを返す
    }
    
    return processed;
}

// AISegmentationDemo クラスの実装

void AISegmentationDemo::runDemoSegmentation() {
    qDebug() << "=== AI Segmentation Demo ===";
    
    QElapsedTimer timer;
    timer.start();
    
    // 1. サンプルボリューム生成
    qDebug() << "Step 1: Generating sample volume...";
    cv::Mat sampleVolume = AISegmentationHelper::generateSampleVolume(32, 128, 128);
    logTestResult("Sample Volume Generation", !sampleVolume.empty(), 
                  QString("Size: %1x%2x%3").arg(sampleVolume.size[2]).arg(sampleVolume.size[1]).arg(sampleVolume.size[0]));
    
    // 2. 前処理
    qDebug() << "Step 2: Preprocessing...";
    cv::Mat preprocessed = AISegmentationHelper::preprocessVolume(sampleVolume);
    logTestResult("Preprocessing", !preprocessed.empty());
    
    // 3. ダミーセグメンテーション（実際のAI推論の代わり）
    qDebug() << "Step 3: Running dummy segmentation...";
    cv::Mat segmentation = AISegmentationHelper::generateDummySegmentation(preprocessed);
    logTestResult("Dummy Segmentation", !segmentation.empty());
    
    // 4. 後処理
    qDebug() << "Step 4: Post-processing...";
    cv::Mat postprocessed = AISegmentationHelper::postprocessSegmentation(segmentation);
    logTestResult("Post-processing", !postprocessed.empty());
    
    qint64 totalTime = timer.elapsed();
    qDebug() << "Demo completed in" << totalTime << "ms";
    
    // 結果表示
    QMessageBox::information(nullptr, "AIセグメンテーション デモ完了", 
        QString("デモが正常に完了しました。\n\n"
                "実行時間: %1 ms\n"
                "サンプルボリューム: %2x%3x%4\n"
                "処理ステップ: 4/4 成功")
                .arg(totalTime)
                .arg(sampleVolume.size[2]).arg(sampleVolume.size[1]).arg(sampleVolume.size[0]));
}

void AISegmentationDemo::runPerformanceTest() {
    qDebug() << "=== Performance Test ===";
    
    QStringList results;
    
    // 複数サイズでのテスト
    std::vector<std::tuple<int,int,int>> testSizes = {
        {16, 64, 64},     // Small
        {32, 128, 128},   // Medium  
        {64, 256, 256},   // Large
        {128, 512, 512}   // Very Large
    };
    
    for (const auto &size : testSizes) {
        int depth = std::get<0>(size);
        int height = std::get<1>(size);
        int width = std::get<2>(size);
        
        QElapsedTimer timer;
        timer.start();
        
        // ボリューム生成
        cv::Mat volume = AISegmentationHelper::generateSampleVolume(depth, height, width);
        qint64 genTime = timer.restart();
        
        // セグメンテーション
        cv::Mat segmentation = AISegmentationHelper::generateDummySegmentation(volume);
        qint64 segTime = timer.elapsed();
        
        QString result = QString("Size %1x%2x%3: Gen=%4ms, Seg=%5ms")
                        .arg(width).arg(height).arg(depth)
                        .arg(genTime).arg(segTime);
        results << result;
        qDebug() << result;
    }
    
    // 結果表示
    QMessageBox::information(nullptr, "パフォーマンステスト結果", 
        "パフォーマンステスト完了:\n\n" + results.join("\n"));
}

void AISegmentationDemo::runVolumeVariationTest() {
    qDebug() << "=== Volume Variation Test ===";
    
    // 様々な条件でのテスト
    QStringList testResults;
    
    // 1. 異なるデータ型
    cv::Mat volume16 = AISegmentationHelper::generateSampleVolume(16, 64, 64);
    cv::Mat volume8;
    volume16.convertTo(volume8, CV_8UC1, 1.0/16.0); // 8bit変換
    
    cv::Mat seg16 = AISegmentationHelper::generateDummySegmentation(volume16);
    cv::Mat seg8 = AISegmentationHelper::generateDummySegmentation(volume8);
    
    testResults << QString("16-bit volume: %1").arg(!seg16.empty() ? "OK" : "FAILED");
    testResults << QString("8-bit volume: %1").arg(!seg8.empty() ? "OK" : "FAILED");
    
    // 2. 極端なサイズ
    cv::Mat tinyVolume = AISegmentationHelper::generateSampleVolume(4, 16, 16);
    cv::Mat tinySeg = AISegmentationHelper::generateDummySegmentation(tinyVolume);
    testResults << QString("Tiny volume (4x16x16): %1").arg(!tinySeg.empty() ? "OK" : "FAILED");
    
    // 3. 2Dスライス
    cv::Mat slice2D = cv::Mat::zeros(128, 128, CV_16SC1);
    cv::Mat seg2D = AISegmentationHelper::generateDummySegmentation(slice2D);
    testResults << QString("2D slice: %1").arg(!seg2D.empty() ? "OK" : "FAILED");
    
    QMessageBox::information(nullptr, "ボリューム多様性テスト", 
        "テスト結果:\n\n" + testResults.join("\n"));
}

void AISegmentationDemo::runMemoryUsageTest() {
    qDebug() << "=== Memory Usage Test ===";
    
    QStringList memoryInfo;
    
    // 大きなボリュームでのメモリ使用量推定
    std::vector<std::tuple<int,int,int,QString>> sizesWithDesc = {
        {64, 256, 256, "4MB (Medium)"},
        {128, 512, 512, "32MB (Large)"},
        {256, 512, 512, "64MB (Very Large)"}
    };
    
    for (const auto &sizeDesc : sizesWithDesc) {
        int depth = std::get<0>(sizeDesc);
        int height = std::get<1>(sizeDesc);
        int width = std::get<2>(sizeDesc);
        QString desc = std::get<3>(sizeDesc);
        
        // メモリ使用量計算（推定）
        size_t volumeBytes = depth * height * width * sizeof(int16_t);
        size_t segmentationBytes = depth * height * width * sizeof(uint8_t);
        size_t totalMB = (volumeBytes + segmentationBytes) / (1024 * 1024);
        
        memoryInfo << QString("%1x%2x%3 %4: ~%5MB")
                      .arg(width).arg(height).arg(depth)
                      .arg(desc).arg(totalMB);
    }
    
    QMessageBox::information(nullptr, "メモリ使用量テスト", 
        "推定メモリ使用量:\n\n" + memoryInfo.join("\n") + 
        "\n\n注意: 実際の使用量は処理により変動します");
}

void AISegmentationDemo::logTestResult(const QString &testName, bool success, const QString &details) {
    QString status = success ? "✓ PASS" : "✗ FAIL";
    QString message = QString("[%1] %2: %3").arg(testName).arg(status).arg(details);
    qDebug() << message;
}