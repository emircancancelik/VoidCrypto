import pandas as pd
import numpy as np
import xgboost as xgb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_15m_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_15m_LABELED.csv")
data_4h_path  = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_4h_LABELED.csv")

model_paths = {
    '15m_long':  os.path.join(BASE_DIR, "agents", "void_model_15m_long.json"),
    '15m_short': os.path.join(BASE_DIR, "agents", "void_model_15m_short.json"),
    '4h_long':   os.path.join(BASE_DIR, "agents", "void_model_4h_long.json"),
    '4h_short':  os.path.join(BASE_DIR, "agents", "void_model_4h_short.json"),
}

# ── Yükle ────────────────────────────────────────────────────────────────────
models = {}
for name, path in model_paths.items():
    m = xgb.XGBClassifier()
    m.load_model(path)
    models[name] = m

df_15m = pd.read_csv(data_15m_path)
df_4h  = pd.read_csv(data_4h_path)

drop_cols = ['timestamp', 'datetime', 'target_long', 'target_short']

X_4h = df_4h.drop(columns=[c for c in drop_cols if c in df_4h.columns], errors='ignore')
df_4h['p_4h_l'] = models['4h_long'].predict_proba(X_4h)[:, 1]
df_4h['p_4h_s'] = models['4h_short'].predict_proba(X_4h)[:, 1]
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

X_15m = df_15m.drop(columns=[c for c in drop_cols if c in df_15m.columns], errors='ignore')
df_15m['p_15m_l'] = models['15m_long'].predict_proba(X_15m)[:, 1]
df_15m['p_15m_s'] = models['15m_short'].predict_proba(X_15m)[:, 1]
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])

rename_map = {}
if 'dist_ema200' in df_4h.columns: rename_map['dist_ema200'] = 'macro_ema200'
if 'atr_pct' in df_4h.columns:     rename_map['atr_pct'] = 'atr_4h'

macro_cols = ['datetime', 'p_4h_l', 'p_4h_s'] + list(rename_map.keys())
df_macro = df_4h[[c for c in macro_cols if c in df_4h.columns]].rename(columns=rename_map)

merged_df = pd.merge_asof(
    df_15m.sort_values('datetime'),
    df_macro.sort_values('datetime'),
    on='datetime', direction='backward'
)

if 'atr_pct' not in merged_df.columns and 'atr_4h' in merged_df.columns:
    merged_df['atr_pct'] = merged_df['atr_4h']

test_df = merged_df.iloc[int(len(merged_df) * 0.80):].copy().reset_index(drop=True)
N = len(test_df)

print(f"\n{'='*55}")
print(f"  TEST SETİ: {N} satır")
print(f"{'='*55}\n")

# ── Kolon bilgisi ─────────────────────────────────────────────────────────────
print("[ KOLON KONTROL ]")
for col in ['atr_pct', 'atr_4h', 'macro_ema200', 'dist_ema200', 'adx',
            'p_4h_l', 'p_4h_s', 'p_15m_l', 'p_15m_s']:
    present = col in test_df.columns
    if present:
        s = test_df[col]
        print(f"  ✓ {col:<18} min={s.min():.4f}  max={s.max():.4f}  "
              f"mean={s.mean():.4f}  NaN={s.isna().sum()}")
    else:
        print(f"  ✗ {col:<18} — MEVCUT DEĞİL")

# ── Olasılık dağılımları ──────────────────────────────────────────────────────
print("\n[ OLASILIK DAĞILIMLARI — Percentile ]")
for col in ['p_4h_l', 'p_4h_s', 'p_15m_l', 'p_15m_s']:
    if col in test_df.columns:
        s = test_df[col]
        pcts = np.percentile(s, [10, 25, 50, 75, 90, 95, 99])
        print(f"  {col}: p10={pcts[0]:.3f} p25={pcts[1]:.3f} p50={pcts[2]:.3f} "
              f"p75={pcts[3]:.3f} p90={pcts[4]:.3f} p95={pcts[5]:.3f} p99={pcts[6]:.3f}")

# ── Filtre huni analizi ───────────────────────────────────────────────────────
print("\n[ FİLTRE HUNİSİ — Her adımda kaç satır kalıyor ]")

MIN_VOLATILITY = 0.25
MAX_VOLATILITY = 5.0
CONF_4H_MIN    = 0.55
SCORE_THRESHOLD = 65

# Adım 1: Volatilite
f1 = test_df['atr_pct'].between(MIN_VOLATILITY, MAX_VOLATILITY) if 'atr_pct' in test_df.columns \
     else pd.Series([True] * N)
print(f"  [1] Volatilite filtresi ({MIN_VOLATILITY}–{MAX_VOLATILITY}): "
      f"{f1.sum()} / {N}  (%{f1.mean()*100:.1f})")

# Adım 2: 4H yön
f2_long  = test_df['p_4h_l'] > CONF_4H_MIN
f2_short = test_df['p_4h_s'] > CONF_4H_MIN
f2 = f1 & (f2_long | f2_short)
print(f"  [2] + 4H bias (>{CONF_4H_MIN}):            "
      f"{f2.sum()} / {N}  (%{f2.mean()*100:.1f})")
print(f"       └─ Long  sinyali: {(f1 & f2_long).sum()}")
print(f"       └─ Short sinyali: {(f1 & f2_short).sum()}")

# Adım 3: Score (basit hesap — macro_ema200 yoksa sıfır varsay)
ema = test_df.get('macro_ema200', pd.Series(np.zeros(N)))

scores_long = pd.Series(np.zeros(N))
scores_short = pd.Series(np.zeros(N))

edge_l = test_df['p_15m_l'] - test_df['p_15m_s']
edge_s = test_df['p_15m_s'] - test_df['p_15m_l']
atr_n  = ((test_df['atr_pct'].clip(MIN_VOLATILITY, MAX_VOLATILITY) - MIN_VOLATILITY)
          / (MAX_VOLATILITY - MIN_VOLATILITY)) if 'atr_pct' in test_df.columns else pd.Series(np.zeros(N))

scores_long  = (edge_l.clip(lower=0) * 40
                + test_df['p_4h_l'] * 30
                + (ema.abs().clip(upper=10) / 10) * 20
                + atr_n * 10)

scores_short = (edge_s.clip(lower=0) * 40
                + test_df['p_4h_s'] * 30
                + (ema.abs().clip(upper=10) / 10) * 20
                + atr_n * 10)

# Score'u yön bazlı seç
actual_score = pd.Series(np.where(f2_long, scores_long, scores_short))

f3 = f2 & (actual_score >= SCORE_THRESHOLD)
print(f"  [3] + Score >= {SCORE_THRESHOLD}:              "
      f"{f3.sum()} / {N}  (%{f3.mean()*100:.1f})")

# Score dağılımı (filtre 2'yi geçenler için)
if f2.sum() > 0:
    sc_filtered = actual_score[f2]
    pcts = np.percentile(sc_filtered, [25, 50, 75, 90, 95, 99]) if len(sc_filtered) > 0 else [0]*6
    print(f"       └─ Score dağılımı (4H geçenler): "
          f"p25={pcts[0]:.1f} p50={pcts[1]:.1f} p75={pcts[2]:.1f} "
          f"p90={pcts[3]:.1f} p95={pcts[4]:.1f} p99={pcts[5]:.1f}")

# ── Önerilen eşikler ──────────────────────────────────────────────────────────
print("\n[ ÖNERİLEN EŞİK AYARLARI ]")
for thresh_4h in [0.50, 0.52, 0.55, 0.58, 0.60]:
    for thresh_sc in [50, 55, 60, 65, 70]:
        mask_4h = f1 & ((test_df['p_4h_l'] > thresh_4h) | (test_df['p_4h_s'] > thresh_4h))
        sc_dir = pd.Series(np.where(test_df['p_4h_l'] > test_df['p_4h_s'],
                                    scores_long, scores_short))
        mask_sc = mask_4h & (sc_dir >= thresh_sc)
        cnt = mask_sc.sum()
        if 10 <= cnt <= 500:  # makul aralık
            print(f"  CONF_4H={thresh_4h}  SCORE>={thresh_sc:2d}  → {cnt} işlem")

print("\n  (Çıktıyı kopyalayıp paylaş — parametreleri buna göre ayarlayalım)\n")