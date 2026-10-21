# ML pipeline (Python)

Trains the production MNIST model offline and exports it in TF.js Layers format
for the live browser demo.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r ml/requirements.txt
python ml/train.py
```

- Dataset: MNIST via `keras.datasets` (60,000 train / 10,000 test, real held-out test set)
- Model: Conv(32) → Pool → Conv(64) → Pool → Dropout(0.3) → Dense(128) → Softmax(10)
- Early stopping on validation accuracy (patience 2)
- Output: `model/model.json` + weight shards, plus `ml/metrics.json` with the true test metrics

The demo (`app.js`) loads `./model/model.json` when present and falls back to
live in-browser training if the pretrained files are missing. The in-browser
trainer is intentionally small (fast, CPU-friendly); the Python pipeline is
where the published accuracy comes from.
