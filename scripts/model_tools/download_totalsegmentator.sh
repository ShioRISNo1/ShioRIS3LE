#!/bin/bash
# TotalSegmentator モデルダウンロード・セットアップスクリプト

set -e

echo "========================================="
echo "TotalSegmentator Setup Script"
echo "========================================="
echo ""

# Python環境確認
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "Please install Python 3.8 or later"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# 仮想環境の作成（推奨）
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv totalseg_env
    source totalseg_env/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# TotalSegmentatorインストール
echo ""
echo "Installing TotalSegmentator..."
pip install --upgrade pip
pip install TotalSegmentator

echo ""
echo "✓ TotalSegmentator installed successfully"

# モデルの事前ダウンロード
echo ""
echo "Downloading segmentation models..."
echo "This may take several minutes..."
totalseg_download_weights -t total

echo ""
echo "========================================="
echo "✓ Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Run: python3 export_totalseg_to_onnx.py"
echo "2. Load the generated ONNX model in ShioRIS3"
echo ""
