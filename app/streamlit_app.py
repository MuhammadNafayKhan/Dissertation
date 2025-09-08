import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from tensorflow import keras

# Locating stored artifacts



ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
REGISTRY_PATH = ARTIFACT_DIR / "registry.json"

# Safety limit: max daily price movement during forecasting (20%)
MAX_DAILY_MOVE_PCT = 0.20


def load_registry() -> Dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return {"models": []}


def list_symbols(registry: Dict) -> List[str]:
    symbols = sorted({m.get("symbol", "BTC-USD") for m in registry.get("models", [])})
    return symbols or ["BTC-USD"]


def list_models_for_symbol(registry: Dict, symbol: str) -> List[Dict]:
    return [m for m in registry.get("models", []) if m.get("symbol") == symbol]


def _resolve_model_path(entry_path: str) -> Path:
    p = Path(entry_path)
    if p.exists():
        return p
    alt = ARTIFACT_DIR / Path(entry_path).name
    if alt.exists():
        return alt
    if REGISTRY_PATH.exists():
        alt2 = REGISTRY_PATH.parent / Path(entry_path).name
        if alt2.exists():
            return alt2
    alt3 = Path("artifacts") / Path(entry_path).name
    return alt3


def load_model(entry: Dict):
    path_str = entry.get("path")
    path = _resolve_model_path(path_str) if path_str else None
    if entry.get("type") == "sklearn":
        return joblib.load(path)
    elif entry.get("type") == "keras":
        custom_objects = {}
        try:
            from tcn import TCN  # type: ignore
            custom_objects.update({"TCN": TCN, "Custom>TCN": TCN})
        except Exception:
            pass
        return keras.models.load_model(path, custom_objects=custom_objects, compile=False, safe_mode=False)
    else:
        return None


# Helper functions for display formatting

def format_currency(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


def format_percent(x: float) -> str:
    try:
        return f"{x:,.2f}%"
    except Exception:
        return str(x)


def plot_series(history: pd.DataFrame, preds: Optional[pd.Series] = None, band: Optional[Tuple[pd.Series, pd.Series]] = None, title: str = "Price"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history['close'], mode='lines', name='Close'))
    if band is not None:
        lower, upper = band
        fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode='lines', name='Upper', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode='lines', name='Lower', fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(width=0), showlegend=False))
    if preds is not None:
        fig.add_trace(go.Scatter(x=preds.index, y=preds.values, mode='lines+markers', name='Forecast', line=dict(color='orange')))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price (USD)')
    st.plotly_chart(fig, use_container_width=True)

# Technical indicators and feature engineering (same as notebook)

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret'] = out['close'].pct_change()
    out['log_ret'] = np.log(out['close']).diff()
    out['sma_10'] = out['close'].rolling(10).mean()
    out['sma_20'] = out['close'].rolling(20).mean()
    out['ema_10'] = out['close'].ewm(span=10, adjust=False).mean()
    out['rsi_14'] = compute_rsi(out['close'], 14)
    out['bb_mid'] = out['close'].rolling(20).mean()
    out['bb_std'] = out['close'].rolling(20).std()
    out['bb_up'] = out['bb_mid'] + 2 * out['bb_std']
    out['bb_low'] = out['bb_mid'] - 2 * out['bb_std']
    out['volatility_20'] = out['log_ret'].rolling(20).std() * np.sqrt(20)
    out = out.dropna()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna()
    return out


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ['close']]


def make_window_tabular(df_feat: pd.DataFrame, features: List[str], lookback: int) -> np.ndarray:
    window = df_feat[features].tail(lookback).values
    return window.reshape(1, -1)


def make_window_seq(df_feat: pd.DataFrame, features: List[str], lookback: int) -> np.ndarray:
    window = df_feat[features].tail(lookback).values
    return window.reshape(1, lookback, len(features))


def fetch_data(symbol: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False, group_by="column", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
    return df[['open','high','low','close','volume']].dropna()


# Model selection and forecasting logic

def choose_best_model(models: List[Dict]) -> Dict:
    order = [
        ("keras", "LSTM"),
        ("sklearn", "GBR"),
        ("sklearn", "SVR"),
        ("keras", "TCN"),
    ]
    for t, name in order:
        for m in models:
            if m.get("type") == t and m.get("name") == name:
                return m
    return models[0] if models else {}


def business_days_horizon(start: pd.Timestamp, target: pd.Timestamp) -> int:
    start = pd.Timestamp(start).normalize()
    target = pd.Timestamp(target).normalize()
    if target <= start:
        return 0
    bdays = pd.bdate_range(start + pd.Timedelta(days=1), target)
    return len(bdays)


def next_business_dates(last_date: pd.Timestamp, steps: int) -> List[pd.Timestamp]:
    dates = []
    curr = pd.Timestamp(last_date).normalize()
    while len(dates) < steps:
        curr += pd.Timedelta(days=1)
        if curr.dayofweek < 5:
            dates.append(curr)
    return dates


def recursive_forecast(df_feat: pd.DataFrame, features: List[str], entry: Dict, steps: int):
    lookback = int(entry.get("lookback", 60))
    try:
        mdl = load_model(entry)
    except Exception as e:
        st.warning(f"Model '{entry.get('name')}' failed to load. Details: {e}")
        return None

    work = df_feat.copy()
    preds = []
    guardrails_triggered = False

    last_close = float(work['close'].iloc[-1])

    for _ in range(steps):
        if entry.get("type") == "sklearn":
            X = make_window_tabular(work, features, lookback)
            yhat = float(mdl.predict(X)[0])
        elif entry.get("type") == "keras":
            model = mdl if not isinstance(mdl, tuple) else mdl[0]
            scaler_x = None if not isinstance(mdl, tuple) else mdl[1]
            scaler_y = None if not isinstance(mdl, tuple) else mdl[2]
            X = make_window_seq(work, features, lookback)
            if scaler_x is not None:
                X = scaler_x.transform(X.reshape(1, -1)).reshape(1, lookback, len(features))
            yhat_scaled = float(model.predict(X, verbose=0).ravel()[0])
            if scaler_y is not None:
                yhat = float(scaler_y.inverse_transform(np.array(yhat_scaled).reshape(-1,1)).ravel()[0])
            else:
                yhat = yhat_scaled
        else:
            yhat = last_close

        # Keeping predictions reasonable
        upper_bound = last_close * (1.0 + MAX_DAILY_MOVE_PCT)
        lower_bound = max(0.0, last_close * (1.0 - MAX_DAILY_MOVE_PCT))
        if yhat > upper_bound:
            yhat = upper_bound
            guardrails_triggered = True
        elif yhat < lower_bound:
            yhat = lower_bound
            guardrails_triggered = True

        preds.append(yhat)
        last_close = yhat

        next_idx = work.index[-1] + pd.Timedelta(days=1)
        while next_idx.dayofweek >= 5:
            next_idx += pd.Timedelta(days=1)
        new_row = {'open': yhat, 'high': yhat, 'low': yhat, 'close': yhat, 'volume': work['volume'].iloc[-1]}
        work = pd.concat([work, pd.DataFrame([new_row], index=[next_idx])])
        work = add_technical_indicators(work)

    pred_index = next_business_dates(df_feat.index[-1], steps)
    series = pd.Series(preds, index=pred_index)
    if guardrails_triggered:
        st.info(f"Applied safety clamp of ±{int(MAX_DAILY_MOVE_PCT*100)}% per day to stabilize forecast.")
    return series


def forecast_band(df_feat: pd.DataFrame, pred_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    vol = float(df_feat['log_ret'].rolling(20).std().dropna().iloc[-1] if 'log_ret' in df_feat.columns else 0.02)
    last = float(df_feat['close'].iloc[-1])
    lowers, uppers = [], []
    for i, (_, val) in enumerate(pred_series.items(), 1):
        width = last * vol * np.sqrt(i)
        lowers.append(val - width)
        uppers.append(val + width)
    return pd.Series(lowers, index=pred_series.index), pd.Series(uppers, index=pred_series.index)


# User interface styling and main app

def apply_css():
    st.markdown("""
    <style>
    [data-testid="block-container"] { padding-left: 2rem; padding-right: 2rem; padding-top: 1rem; }
    [data-testid="stMetric"] { background-color: #2e2e2e; text-align: center; padding: 14px 0; border-radius: 10px; border: 1px solid #3a3a3a; }
    .metric-label { font-weight: 600; }
    .explain { background-color: #222; padding: 14px; border-radius: 8px; border: 1px solid #333; }
    .range-box { background-color: #1e1e1e; padding: 10px; border-radius: 8px; display:inline-block; margin-top: 6px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Crypto Forecasting", layout="wide", page_icon="📈", initial_sidebar_state="expanded")
    apply_css()

    registry = load_registry()
    symbols = list_symbols(registry)
    if not registry.get("models"):
        st.warning("No models found. Ensure artifacts and registry.json are under 'artifacts/' or 'app/artifacts/'.")
        st.stop()

    with st.sidebar:
        st.title("📈 Crypto Forecasting")
        symbol = st.selectbox("Symbol", symbols)
        target_date = st.date_input("Forecast date", pd.Timestamp.today().date() + pd.Timedelta(days=7))
        models_for_symbol = list_models_for_symbol(registry, symbol)
        model_labels = ["Auto (best)"] + [f"{m.get('name')} ({m.get('type')})" for m in models_for_symbol]
        model_choice = st.selectbox("Model", model_labels, index=0)
        manual = model_choice != "Auto (best)"
        st.caption("Pick a specific model or let us choose the best one.")

    if manual:
        idx = model_labels.index(model_choice) - 1
        model_entry = models_for_symbol[idx]
    else:
        model_entry = choose_best_model(models_for_symbol)

    raw = fetch_data(symbol, period="3y", interval="1d")
    df_feat = add_technical_indicators(raw)
    features = model_entry.get("features") or select_feature_columns(df_feat)

    last_price = float(df_feat['close'].iloc[-1])
    last_ret = float(df_feat['ret'].iloc[-1]) if 'ret' in df_feat.columns else 0.0
    vol20 = float(df_feat['volatility_20'].iloc[-1]) if 'volatility_20' in df_feat.columns else 0.0

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.metric(label="Last Price", value=format_currency(last_price))
    with c2:
        st.metric(label="Last Return", value=format_percent(last_ret*100))
    with c3:
        st.metric(label="Volatility 20", value=format_percent(vol20*100))
    with c4:
        st.metric(label="Model", value=f"{model_entry.get('name')} ({model_entry.get('type')})")

    horizon = business_days_horizon(df_feat.index[-1], pd.Timestamp(target_date))
    if horizon <= 0:
        st.info("Pick a future date to see our forecast.")
        plot_series(df_feat.tail(300), title=f"{symbol} close price")
        return
    if horizon > 90:
        st.warning("Horizon capped at 90 business days for stability.")
        horizon = 90

    with st.spinner("Working on your forecast..."):
        preds = recursive_forecast(df_feat, features, model_entry, steps=horizon)
        if preds is None:
            if manual:
                st.error("Selected model failed to load. Please choose a different model.")
                return
            remaining = [m for m in models_for_symbol if m is not model_entry]
            for cand in remaining:
                st.info(f"Trying fallback model: {cand.get('name')} ({cand.get('type')})")
                preds = recursive_forecast(df_feat, features, cand, steps=horizon)
                if preds is not None:
                    model_entry = cand
                    break
        if preds is None:
            st.error("All models failed to load. Ensure dependencies (e.g., keras-tcn) are installed.")
            return
        lower, upper = forecast_band(df_feat, preds)

    final_price = float(preds.iloc[-1])
    delta = final_price - last_price
    delta_pct = (delta / last_price) * 100.0 if last_price != 0 else 0.0

    extreme = (final_price > last_price * 5) or (final_price < last_price / 5)
    if extreme:
        st.warning("This forecast looks way off from current price. Maybe try a different model.")

    c5, c6, c7 = st.columns([1,1,1])
    with c5:
        st.metric(label="Forecast (target)", value=format_currency(final_price), delta=format_currency(delta))
    with c6:
        st.metric(label="Change %", value=format_percent(delta_pct), delta=format_percent(delta_pct))
    with c7:
        st.metric(label="Horizon (bdays)", value=f"{horizon}")

    lo_last, hi_last = float(lower.iloc[-1]), float(upper.iloc[-1])
    st.markdown(f"<div class='range-box'>Expected range on {pd.Timestamp(target_date).date()}: <b>{format_currency(lo_last)}</b> to <b>{format_currency(hi_last)}</b></div>", unsafe_allow_html=True)

    plot_series(df_feat.tail(300), preds=preds, band=(lower, upper), title=f"{symbol} forecast to {pd.Timestamp(target_date).date()}")

    st.markdown(
        f"""
        <div class='explain'>
        <b>How to read this:</b><br/>
        - We forecast the {symbol} closing price for {pd.Timestamp(target_date).date()} over {horizon} business days ahead.<br/>
        - The orange line shows the step-by-step forecast path; the shaded band is an uncertainty interval derived from recent volatility (not a guarantee).<br/>
        - "Forecast (target)" is our point estimate for that date; "Change %" compares it to the latest close.<br/>
        - Results depend on the selected trained model and the latest market data used as input.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


