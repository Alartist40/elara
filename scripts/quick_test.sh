#!/bin/bash
# Quick validation that everything works

echo "=== Elara Quick Test ==="
source elara_env/bin/activate 2>/dev/null || echo "Warning: venv not activated"

# Test 1: Memory check
echo ""
echo "[1/5] Testing memory check..."
python3 -c "from elara_core.utils import check_memory; m = check_memory(); print(f'  ✓ Memory: {m[\"available_gb\"]:.1f}GB available')"

# Test 2: Tier 1 (if model exists)
echo ""
echo "[2/5] Testing Tier 1..."
if [ -f "${ELARA_MODEL_PATH:-models/qwen-1.5b-q4.gguf}" ]; then
    python3 -c "
from elara_core.tiers.tier1 import Tier1Engine
e = Tier1Engine()
print(f'  ✓ Tier 1 loaded: {e.get_stats()}' if e.model else '  ✗ Tier 1 failed to load')
"
else
    echo "  ⚠ No model found, skipping Tier 1 test"
fi

# Test 3: Tier 2
echo ""
echo "[3/5] Testing Tier 2..."
python3 -c "
from elara_core.tiers.tier2 import Tier2Engine
e = Tier2Engine(tier1_engine=None)
print(f'  ✓ Tier 2 loaded, {len(e.documents)} documents indexed')
"

# Test 4: STT (OLMoASR)
echo ""
echo "[4/5] Testing OLMoASR STT..."
python3 -c "
from elara_core.voice.olmo_stt import OLMoSTT
try:
    s = OLMoSTT('tiny')
    print(f'  ✓ OLMoASR ready (lazy load)')
except Exception as e:
    print(f'  ✗ OLMoASR error: {e}')
"

# Test 5: TTS (Piper)
echo ""
echo "[5/5] Testing Piper TTS..."
python3 -c "
from elara_core.voice.piper_tts import PiperTTS
try:
    t = PiperTTS()
    if t.piper_available:
        print(f'  ✓ Piper TTS available')
    else:
        print(f'  ⚠ Piper not installed (run: pip install piper-tts)')
except Exception as e:
    print(f'  ✗ Piper error: {e}')
"

echo ""
echo "=== Test Complete ==="
echo "Run full test: python3 -m elara_core.main --text 'Hello world'"
