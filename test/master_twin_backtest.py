import pandas as pd
import numpy as np
import xgboost as xgb
import os
from dataclasses import dataclass
from typing import List, Optional
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_15m_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_15m_LABELED.csv")
data_4h_path  = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_4h_LABELED.csv")

model_paths = {
    '15m_long':  os.path.join(BASE_DIR, "agents", "void_model_15m_long.json"),
    '15m_short': os.path.join(BASE_DIR, "agents", "void_model_15m_short.json"),
    '4h_long':   os.path.join(BASE_DIR, "agents", "void_model_4h_long.json"),
    '4h_short':  os.path.join(BASE_DIR, "agents", "void_model_4h_short.json"),
}

for path in [data_15m_path, data_4h_path] + list(model_paths.values()):
    if not os.path.exists(path):
        print(f"[KRİTİK HATA] Dosya bulunamadi -> {path}")
        exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. PARAMETRELER
# ══════════════════════════════════════════════════════════════════════════════
INITIAL_CAPITAL   = 1000.0
COMMISSION_RATE   = 0.001

# "LONG_ONLY" | "SHORT_ONLY" | "BOTH"
TRADING_MODE = "LONG_ONLY"

# Position Sizing
USE_KELLY         = True
TRADE_PERCENT     = 0.10
KELLY_FRACTION    = 0.25
MAX_TRADE_PERCENT = 0.20
MIN_TRADE_PERCENT = 0.03
KELLY_WINDOW      = 50

# Risk — 4H ATR bazlı
# atr_4h mean=1.38% → sl = 1.38/100 * 1.0 = %1.38 (BTC icin makul)
ATR_MULT = 1.0
RR_RATIO = 2.0

# Volatilite (15m atr_pct)
MIN_VOLATILITY = 0.20
MAX_VOLATILITY = 2.0

# 4H esigi (0.55 → 0.57)
CONF_4H_MIN = 0.57

# Macro trend filtresi
EMA_LONG_MIN         = 0.0   # LONG icin macro_ema200 > 0
EMA_SHORT_MAX        = 0.0   # SHORT icin macro_ema200 < 0
USE_EMA_TREND_FILTER = True

# Score
SCORE_THRESHOLD        = 60
EDGE_NORM_MAX          = 0.15
P4H_NORM_MIN           = 0.57
P4H_NORM_MAX           = 0.80
EMA_NORM_MAX           = 15.0
CONF_15M_OPPOSE_THRESH = 0.56
SCORE_PENALTY          = 10

# Calibration
CALIBRATE_15M    = True
CALIB_TRAIN_FRAC = 0.15

# Koruma
COOLDOWN_PERIOD      = 3
MAX_CONSECUTIVE_LOSS = 4
DRAWDOWN_HALT_PCT    = 0.15

TEST_SPLIT = 0.80

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATACLASS & UTILS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    datetime:      str
    direction:     str
    score:         float
    pos_size:      float
    tp_pct:        float
    sl_pct:        float
    rr_actual:     float
    result:        str
    pnl:           float
    capital_after: float
    conf_4h:       float
    conf_15m_dir:  float
    conf_15m_opp:  float
    atr_15m:       float
    atr_4h_used:   float
    macro_ema:     float
    trade_pct:     float


def norm(v, lo, hi):
    if hi <= lo: return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def compute_score(row, direction):
    if direction == "LONG":
        p_4h  = row['p_4h_l']
        p_15d = row['p_15m_l']
        p_15o = row['p_15m_s']
    else:
        p_4h  = row['p_4h_s']
        p_15d = row['p_15m_s']
        p_15o = row['p_15m_l']

    edge    = p_15d - p_15o
    s_edge  = norm(edge, 0.0, EDGE_NORM_MAX) * 40.0
    s_4h    = norm(p_4h, P4H_NORM_MIN, P4H_NORM_MAX) * 30.0
    s_ema   = norm(abs(row.get('macro_ema200', 0.0)), 0.0, EMA_NORM_MAX) * 20.0
    s_vol   = norm(row['atr_pct'], MIN_VOLATILITY, MAX_VOLATILITY) * 10.0
    penalty = SCORE_PENALTY if (p_15o > CONF_15M_OPPOSE_THRESH and p_15o > p_15d) else 0.0
    return round(s_edge + s_4h + s_ema + s_vol - penalty, 2)


def compute_kelly(win_rate, rr):
    if win_rate <= 0 or win_rate >= 1:
        return MIN_TRADE_PERCENT
    q = 1.0 - win_rate
    k = (win_rate * rr - q) / rr * KELLY_FRACTION
    return float(np.clip(k, MIN_TRADE_PERCENT, MAX_TRADE_PERCENT))


def compute_sharpe(dollar_pnls, total_candles, periods_per_day=96):
    if len(dollar_pnls) < 2 or total_candles <= 0:
        return 0.0
        
    arr = np.array(dollar_pnls)
    std = arr.std()
    
    if std == 0:
        return 0.0
        
    mean_pnl = arr.mean()
    total_days = total_candles / periods_per_day
    
    if total_days <= 0:
        return 0.0
        
    # Yılda ortalama kaç işlem yapıyoruz?
    trades_per_year = len(dollar_pnls) / (total_days / 365.25)
    
    # Gerçek yıllıklandırılmış Sharpe
    sharpe_ratio = float((mean_pnl / std) * np.sqrt(trades_per_year))
    
    return sharpe_ratio


def compute_max_drawdown(equity):
    arr = np.array(equity)
    peak = np.maximum.accumulate(arr)
    return float(((arr - peak) / peak).min())


def calibrate_model(model, X_calib, y_calib):
    if _HAS_FROZEN:
        frozen_model = FrozenEstimator(model)
        calib = CalibratedClassifierCV(
            estimator=frozen_model, 
            method='isotonic'
        )
    else:
        calib = CalibratedClassifierCV(
            estimator=model, 
            method='isotonic', 
            cv='prefit'
        )
        
    calib.fit(X_calib, y_calib)
    return calib

# ══════════════════════════════════════════════════════════════════════════════
# 4. YUKLEME
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  VOID MASTER ORCHESTRATOR V5.2")
print(f"  Mod: {TRADING_MODE}  |  RR: {RR_RATIO}  |  ATR Bazi: 4H")
print("=" * 62)

print("\n[1/4] Modeller yukleniyor...")
raw_models = {}
for name, path in model_paths.items():
    m = xgb.XGBClassifier()
    m.load_model(path)
    raw_models[name] = m
    print(f"      OK {name}")

print("\n[2/4] Veriler okunuyor...")
df_15m = pd.read_csv(data_15m_path)
df_4h  = pd.read_csv(data_4h_path)

drop_cols = ['timestamp', 'datetime', 'target_long', 'target_short']

X_4h = df_4h.drop(columns=[c for c in drop_cols if c in df_4h.columns], errors='ignore')
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

X_15m_full = df_15m.drop(columns=[c for c in drop_cols if c in df_15m.columns], errors='ignore')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])

# ── Calibration ───────────────────────────────────────────────────────────────
models = dict(raw_models)

if CALIBRATE_15M:
    print("\n[3/4] 15m modelleri kalibre ediliyor (isotonic)...")
    train_end   = int(len(df_15m) * TEST_SPLIT)
    calib_start = int(train_end * (1 - CALIB_TRAIN_FRAC))
    X_calib     = X_15m_full.iloc[calib_start:train_end]

    for target, model_key in [('target_long', '15m_long'), ('target_short', '15m_short')]:
        if target in df_15m.columns:
            y_calib = df_15m[target].iloc[calib_start:train_end].values
            models[model_key] = calibrate_model(raw_models[model_key], X_calib, y_calib)
            print(f"      OK {model_key} kalibre edildi")
        else:
            print(f"      WARN {target} kolonu yok, atlandi")
else:
    print("\n[3/4] Calibration atlandi.")

# ── Tahminler ─────────────────────────────────────────────────────────────────
print("\n[4/4] Tahminler ve merge...")

df_4h['p_4h_l'] = raw_models['4h_long'].predict_proba(X_4h)[:, 1]
df_4h['p_4h_s'] = raw_models['4h_short'].predict_proba(X_4h)[:, 1]

df_15m['p_15m_l'] = models['15m_long'].predict_proba(X_15m_full)[:, 1]
df_15m['p_15m_s'] = models['15m_short'].predict_proba(X_15m_full)[:, 1]

rename_map = {}
if 'dist_ema200' in df_4h.columns: rename_map['dist_ema200'] = 'macro_ema200'
if 'atr_pct'     in df_4h.columns: rename_map['atr_pct']     = 'atr_4h'

macro_cols = ['datetime', 'p_4h_l', 'p_4h_s'] + list(rename_map.keys())
df_macro = df_4h[[c for c in macro_cols if c in df_4h.columns]].rename(columns=rename_map)

merged_df = pd.merge_asof(
    df_15m.sort_values('datetime'),
    df_macro.sort_values('datetime'),
    on='datetime', direction='backward'
)

if 'atr_pct' not in merged_df.columns and 'atr_4h' in merged_df.columns:
    merged_df['atr_pct'] = merged_df['atr_4h']

if 'atr_4h' not in merged_df.columns:
    raise KeyError("atr_4h kolonu yok. 4H CSV'sinde 'atr_pct' olmali.")

print(f"      OK {len(merged_df)} satir birlestirildi")
print(f"      15m_long  p50={np.percentile(merged_df['p_15m_l'],50):.4f}  "
      f"p99={np.percentile(merged_df['p_15m_l'],99):.4f}")
print(f"      15m_short p50={np.percentile(merged_df['p_15m_s'],50):.4f}  "
      f"p99={np.percentile(merged_df['p_15m_s'],99):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
test_df = merged_df.iloc[int(len(merged_df) * TEST_SPLIT):].copy().reset_index(drop=True)

capital          = INITIAL_CAPITAL
peak_capital     = INITIAL_CAPITAL
l_w = l_l = s_w = s_l = 0
total_comm       = 0.0
cooldown_timer   = 0
consecutive_loss = 0
halted           = False

trade_log:    List[Trade] = []
equity_curve: List[float] = [INITIAL_CAPITAL]
dollar_pnls:  List[float] = []
win_history:  List[int]   = []

print(f"\n{'─'*62}")
print(f"  Backtest | {len(test_df)} mum | Score>={SCORE_THRESHOLD} | "
      f"4H>={CONF_4H_MIN} | ATR*{ATR_MULT} | RR={RR_RATIO}")
print(f"{'─'*62}\n")

for idx, row in test_df.iterrows():

    if halted:
        break

    dd = (capital - peak_capital) / peak_capital
    if dd <= -DRAWDOWN_HALT_PCT:
        print(f"  DURDURULDU: Drawdown %{dd*100:.1f}")
        halted = True
        break

    if cooldown_timer > 0:
        cooldown_timer -= 1
        continue

    atr_15m = row['atr_pct']
    if not (MIN_VOLATILITY <= atr_15m <= MAX_VOLATILITY):
        continue

    # 4H Bias
    direction: Optional[str] = None
    if TRADING_MODE in ("LONG_ONLY", "BOTH"):
        if row['p_4h_l'] > CONF_4H_MIN and row['p_4h_l'] > row['p_4h_s']:
            direction = "LONG"
    if direction is None and TRADING_MODE in ("SHORT_ONLY", "BOTH"):
        if row['p_4h_s'] > CONF_4H_MIN and row['p_4h_s'] > row['p_4h_l']:
            direction = "SHORT"
    if direction is None:
        continue

    # EMA trend filtresi
    ema = row.get('macro_ema200', 0.0)
    if USE_EMA_TREND_FILTER:
        if direction == "LONG"  and ema < EMA_LONG_MIN:  continue
        if direction == "SHORT" and ema > EMA_SHORT_MAX: continue

    # Score
    score = compute_score(row, direction)
    if score < SCORE_THRESHOLD:
        continue

    # Position sizing
    if USE_KELLY and len(win_history) >= 10:
        wr = sum(win_history[-KELLY_WINDOW:]) / min(len(win_history), KELLY_WINDOW)
        trade_pct = compute_kelly(wr, RR_RATIO)
    else:
        trade_pct = TRADE_PERCENT

    pos_size = capital * trade_pct

    # TP/SL — 4H ATR (gürültüden korunmak için)
    atr_4h_val = row['atr_4h']
    atr_4h_dec = atr_4h_val / 100.0
    sl_pct     = atr_4h_dec * ATR_MULT
    tp_pct     = sl_pct * RR_RATIO
    rr_actual  = tp_pct / sl_pct if sl_pct > 0 else 0.0

    # Simülasyon
    entry_fee = pos_size * COMMISSION_RATE

    if direction == "LONG":
        won = (row['target_long'] == 1)
        exit_ratio = (1 + tp_pct) if won else (1 - sl_pct)
    else:
        won = (row['target_short'] == 1)
        exit_ratio = (1 - tp_pct) if won else (1 + sl_pct)

    exit_fee = (pos_size * exit_ratio) * COMMISSION_RATE
    gross    =  pos_size * tp_pct if won else -(pos_size * sl_pct)
    net_pnl  = gross - entry_fee - exit_fee

    capital     += net_pnl
    total_comm  += entry_fee + exit_fee
    peak_capital = max(peak_capital, capital)

    win_history.append(1 if won else 0)
    if direction == "LONG":
        l_w += (1 if won else 0)
        l_l += (0 if won else 1)
    else:
        s_w += (1 if won else 0)
        s_l += (0 if won else 1)

    consecutive_loss = 0 if won else consecutive_loss + 1
    if consecutive_loss >= MAX_CONSECUTIVE_LOSS:
        cooldown_timer = COOLDOWN_PERIOD * 4
        consecutive_loss = 0
        print(f"  WARN ardisik {MAX_CONSECUTIVE_LOSS} kayip -> uzun cooldown @ idx={idx}")
    else:
        cooldown_timer = COOLDOWN_PERIOD

    equity_curve.append(capital)
    dollar_pnls.append(net_pnl)

    trade_log.append(Trade(
        datetime      = str(row.get('datetime', idx)),
        direction     = direction,
        score         = score,
        pos_size      = round(pos_size, 2),
        tp_pct        = round(tp_pct * 100, 4),
        sl_pct        = round(sl_pct * 100, 4),
        rr_actual     = round(rr_actual, 3),
        result        = "WIN" if won else "LOSS",
        pnl           = round(net_pnl, 4),
        capital_after = round(capital, 2),
        conf_4h       = round(row['p_4h_l'] if direction == "LONG" else row['p_4h_s'], 4),
        conf_15m_dir  = round(row['p_15m_l'] if direction == "LONG" else row['p_15m_s'], 4),
        conf_15m_opp  = round(row['p_15m_s'] if direction == "LONG" else row['p_15m_l'], 4),
        atr_15m       = round(atr_15m, 4),
        atr_4h_used   = round(atr_4h_val, 4),
        macro_ema     = round(ema, 4),
        trade_pct     = round(trade_pct, 4),
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 6. METRİKLER
# ══════════════════════════════════════════════════════════════════════════════
total_tr = l_w + l_l + s_w + s_l
wins     = l_w + s_w
losses   = l_l + s_l
win_rate = (wins / total_tr * 100) if total_tr > 0 else 0.0
roi      = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
max_dd   = compute_max_drawdown(equity_curve)
sharpe = compute_sharpe(dollar_pnls, total_candles=len(test_df), periods_per_day=96)
calmar   = (roi / 100) / abs(max_dd) if max_dd != 0 else 0.0

win_pnls  = [t.pnl for t in trade_log if t.result == "WIN"]
loss_pnls = [t.pnl for t in trade_log if t.result == "LOSS"]
avg_win   = np.mean(win_pnls)  if win_pnls  else 0.0
avg_loss  = np.mean(loss_pnls) if loss_pnls else 0.0
pf        = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf')
best      = max(trade_log, key=lambda t: t.pnl).pnl  if trade_log else 0.0
worst     = min(trade_log, key=lambda t: t.pnl).pnl  if trade_log else 0.0

scores_all = [t.score for t in trade_log]
sc_pcts    = np.percentile(scores_all, [25, 50, 75, 90]) if scores_all else [0,0,0,0]

# Breakeven win rate (beklenti sifir olmasi icin gereken minimum)
# Expected Value = wr * avg_win + (1-wr) * avg_loss = 0
# wr_break = |avg_loss| / (avg_win + |avg_loss|)
if win_pnls and loss_pnls and avg_win > 0:
    wr_break = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100
else:
    wr_break = 50.0

print("╔" + "═"*60 + "╗")
print("║       VOID MASTER ORCHESTRATOR V5.2 — SONUCLAR         ║")
print("╠" + "═"*60 + "╣")
print(f"║  Baslangic Kasasi   : ${INITIAL_CAPITAL:>10.2f}                        ║")
print(f"║  Bitis Kasasi       : ${capital:>10.2f}                        ║")
print(f"║  ROI                : %{roi:>+10.2f}                        ║")
print("╠" + "═"*60 + "╣")
print(f"║  Toplam Islem       : {total_tr:>6}                                ║")
print(f"║  Kazanma Orani      : %{win_rate:>6.2f}  (breakeven: %{wr_break:.1f})          ║")
print(f"║  Long   (W / L)     : {l_w:>4} / {l_l:<4}                             ║")
print(f"║  Short  (W / L)     : {s_w:>4} / {s_l:<4}                             ║")
print("╠" + "═"*60 + "╣")
print(f"║  Sharpe Ratio       : {sharpe:>8.3f}                              ║")
print(f"║  Max Drawdown       : %{max_dd*100:>+8.2f}                             ║")
print(f"║  Calmar Ratio       : {calmar:>8.3f}                              ║")
print(f"║  Profit Factor      : {pf:>8.3f}                              ║")
print("╠" + "═"*60 + "╣")
print(f"║  Ort. Kazanc/Islem  : ${avg_win:>8.4f}                           ║")
print(f"║  Ort. Kayip/Islem   : ${avg_loss:>8.4f}                           ║")
print(f"║  En Iyi Islem       : ${best:>+8.4f}                           ║")
print(f"║  En Kotu Islem      : ${worst:>+8.4f}                           ║")
print(f"║  Odenen Komisyon    : ${total_comm:>10.2f}                        ║")
print("╠" + "═"*60 + "╣")
print(f"║  Score (p25/p50/p75/p90): "
      f"{sc_pcts[0]:.1f} / {sc_pcts[1]:.1f} / {sc_pcts[2]:.1f} / {sc_pcts[3]:.1f}           ║")
print("╚" + "═"*60 + "╝")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════
LOG_PATH = os.path.join(BASE_DIR, "backtest_trade_log_v5_2.csv")
if trade_log:
    pd.DataFrame([t.__dict__ for t in trade_log]).to_csv(LOG_PATH, index=False)
    print(f"\n  Trade log -> {LOG_PATH}")

print("  Backtest tamamlandi.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SONRAKI ADIM: SHORT MODEL RETRAINING REHBERI
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 62)
print("  SHORT MODEL RETRAINING ICIN YAPILMASI GEREKENLER:")
print()
print("  1. Label kontrolu:")
print("     df['target_short'] dagiliminina bak:")
print("     df['target_short'].value_counts() / len(df)")
print("     Eger 0/1 orani >4:1 ise imbalance var.")
print()
print("  2. Imbalance cozumu:")
print("     xgb.XGBClassifier(scale_pos_weight=negative/positive)")
print("     veya smote ile oversample et.")
print()
print("  3. Short-specific feature ekle:")
print("     - RSI overbought (>70) durumu")
print("     - Funding rate negatif (bearish bias)")
print("     - Higher high / lower high pattern")
print()
print("  4. Calibration sonrasi p99 > 0.65 olursa short'u BOTH moda al.")
print("─" * 62)
