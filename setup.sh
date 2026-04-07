#!/bin/bash
set -e

echo "========================================="
echo " Indian Stock Market AI Agent - Setup"
echo "========================================="
echo ""

cd "$(dirname "$0")"

# --- Step 1: Install Python dependencies ---
echo "[1/4] Installing Python dependencies..."
python3 -m pip install --user -r requirements.txt
echo "  Done."
echo ""

# --- Step 2: Check / Install Ollama ---
echo "[2/4] Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "  Ollama is already installed."
else
    echo "  Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed."
fi
echo ""

# --- Step 3: Pull the LLM model ---
echo "[3/4] Pulling llama3 model (this may take a few minutes on first run)..."
ollama pull llama3
echo "  Model ready."
echo ""

# --- Step 4: Quick smoke test ---
echo "[4/4] Running smoke test..."
python3 -c "
from data.fetcher import get_stock_data
from analysis.technical import generate_analysis_summary

print('Fetching RELIANCE.NS data...')
df = get_stock_data('RELIANCE', period='3mo')
print(f'  Got {len(df)} rows of price data.')

summary = generate_analysis_summary(df, 'RELIANCE')
print()
print(f'  Price:          ₹{summary[\"price\"]}')
print(f'  RSI:            {summary[\"rsi\"]}')
print(f'  MACD:           {summary[\"macd\"]}')
print(f'  SMA Trend:      {summary[\"sma_trend\"]}')
print(f'  Bollinger:      {summary[\"bollinger\"]}')
print(f'  Support:        ₹{summary[\"support\"]}')
print(f'  Resistance:     ₹{summary[\"resistance\"]}')
print(f'  Overall Signal: {summary[\"overall_signal\"]}')
print()
print('Smoke test PASSED!')
"
echo ""
echo "========================================="
echo " Setup complete! You can now use:"
echo ""
echo "   python3 cli.py analyze RELIANCE"
echo "   python3 cli.py compare TCS INFY WIPRO"
echo "   python3 cli.py ask \"Is HDFC Bank bullish?\""
echo "   python3 cli.py chat-mode"
echo "   streamlit run app.py"
echo ""
echo "========================================="
