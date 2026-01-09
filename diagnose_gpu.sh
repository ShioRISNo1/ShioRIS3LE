#!/bin/bash
# GPU診断スクリプト

echo "=== GPU環境診断 ==="
echo ""

echo "1. nvidia-smi確認:"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo ""

echo "2. CUDA Toolkitバージョン確認:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "nvcc not found"
fi
echo ""

echo "3. ONNX Runtime関連ファイル確認:"
echo "Main library:"
ls -lh /usr/local/onnxruntime/lib/libonnxruntime.so 2>/dev/null || echo "Not found"
echo ""
echo "CUDA Provider:"
ls -lh /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so 2>/dev/null || echo "Not found"
echo ""
echo "TensorRT Provider:"
ls -lh /usr/local/onnxruntime/lib/libonnxruntime_providers_tensorrt.so 2>/dev/null || echo "Not found"
echo ""

echo "4. CMakeビルド設定確認:"
if [ -f build/CMakeCache.txt ]; then
    echo "ONNXRUNTIME_USE_CUDA:"
    grep "ONNXRUNTIME_USE_CUDA" build/CMakeCache.txt || echo "Not set"
    echo ""
    echo "ONNXRUNTIME_CUDA_LIB:"
    grep "ONNXRUNTIME_CUDA_LIB" build/CMakeCache.txt || echo "Not set"
fi
echo ""

echo "5. コンパイル済みバイナリのシンボル確認:"
if [ -f build/ShioRIS3 ]; then
    echo "ONNXRUNTIME_USE_CUDA マクロ定義確認:"
    strings build/ShioRIS3 | grep -i "cuda.*execution.*provider" | head -3
    echo ""
    echo "リンクされているONNX Runtime ライブラリ:"
    ldd build/ShioRIS3 | grep onnx
fi

echo ""
echo "=== 診断完了 ==="
