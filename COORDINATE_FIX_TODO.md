# 座標オフセット修正 - 実装メモ

## 問題
RT-CT自動クロッピング後、セグメンテーション結果の位置がずれている。

## 原因
クロップしたオフセット情報（minX, minY, minZ）が失われているため、
resample3DToOriginalSize()で元の座標系に戻すときに正しい位置に配置されない。

## 実装状況

### ✅ 完了
1. メンバ変数追加（include/ai/onnx_segmenter.h）
   - m_cropOffsetX, m_cropOffsetY, m_cropOffsetZ
   - m_croppedWidth, m_croppedHeight, m_croppedDepth

2. オフセット保存（src/ai/onnx_segmenter.cpp: ~1685行）
   - resampleVolumeFor3D()でクロップオフセットを保存

### ⏳ 未完了
3. resample3DToOriginalSize()の修正（src/ai/onnx_segmenter.cpp: ~1954行）

   必要な処理：
   ```cpp
   // Step 1: モデル出力をクロップサイズにリサンプル
   cv::Mat croppedResult = resampleToSize(segmentation3D, m_croppedDepth, m_croppedHeight, m_croppedWidth);

   // Step 2: 元のサイズの空白ボリュームを作成（背景=0）
   cv::Mat result(targetDepth, targetHeight, targetWidth, CV_8UC1, Scalar(0));

   // Step 3: オフセットを加えてクロップ領域を配置
   for (z, y, x in croppedResult):
       result[z + m_cropOffsetZ][y + m_cropOffsetY][x + m_cropOffsetX] = croppedResult[z][y][x];
   ```

## 次のステップ
1. まず、クロッピングが動作しているか確認（ログに "Detecting body boundaries" が表示されるか）
2. 動作していない場合はリビルドを確認
3. 動作している場合、resample3DToOriginalSize()を修正
