```
#!/bin/bash

###############################################################################
# Kaggle Dataset Downloader Script - Custom Version
# -----------------------------------------------------------------------------
# Downloads and unzips the flower-pollinator dataset from Kaggle.
#
# Dataset: sadiqkaggle/flower-pollinator-dataset
# Output:  ./data/
#
# Usage:
#   1. Install the Kaggle CLI:
#        pip install kaggle
#
#   2. Get your Kaggle API key:
#        - Go to: https://www.kaggle.com/account
#        - Click: "Create New API Token"
#        - Save the downloaded kaggle.json file
#
#   3. Place `kaggle.json` in:
#        - ~/.kaggle/kaggle.json  (recommended)
#        - OR current directory (this script will move it for you)
#
#   4. Run:
#        ./download.sh
###############################################################################

set -e

DATASET="sadiqkaggle/flower-pollinator-dataset"
OUT_DIR="data"

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Error: Kaggle CLI not found. Please install it:"
    echo "   pip install kaggle"
    exit 1
fi

# Prepare kaggle.json if in current folder
if [ -f "./kaggle.json" ]; then
    echo "🔐 Found kaggle.json in current directory. Configuring..."
    mkdir -p ~/.kaggle
    cp ./kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✅ Moved kaggle.json to ~/.kaggle/"
elif [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ kaggle.json not found!"
    echo "➡️  Please place it in either:"
    echo "   - ~/.kaggle/kaggle.json"
    echo "   - OR this script directory (we will move it automatically)"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

echo "📦 Downloading dataset: $DATASET"
kaggle datasets download -d "$DATASET" -p "$OUT_DIR" --unzip

echo "✅ Download complete! Files are saved in: $OUT_DIR"

```
