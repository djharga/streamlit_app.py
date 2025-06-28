import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os
import logging

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# الإعدادات الافتراضية
DEFAULT_PAIRS = ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "EURUSD=X"]
DEFAULT_INDICATORS = {
    "RSI": {"period": 14, "active": True},
    "MACD": {"active": True},
    "BB": {"period": 20, "active": True},
    "CTI": {"period": 14, "active": True},
    "VPIN": {"period": 20, "active": True},
    "AMV": {"period": 14, "active": True},
    "TSD": {"period": 10, "active": True},
    "QMS": {"period": 5, "active": True},
    "NVI": {"period": 255, "active": True},
    "PFE": {"period": 14, "active": True},
}
MODEL_PARAMS = {
    "lookback_period": 5,
    "test_size": 0.2,
    "threshold": 0.8,
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
}
MODEL_PATH = "trained_model.joblib"

# جلب البيانات مع كاش
@st.cache_data(show_spinner=False)
def get_data(symbol, period="1y"):
    end_date = datetime.now()
    period_map = {
        "1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730
    }
    days = period_map.get(period, 365)
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.warning(f"لا توجد بيانات متاحة لـ {symbol}")
        raise ValueError("لا توجد بيانات.")
    data = data.ffill().bfill()
    return data

# حساب المؤشرات
def calculate_indicators(data, indicators_config):
    df = pd.DataFrame(index=data.index)
    for name, settings in indicators_config.items():
        if not settings.get("active", True):
            continue
        try:
            if name == "RSI":
                df["RSI"] = data.ta.rsi(length=settings.get("period", 14))
            elif name == "MACD":
                macd = data.ta.macd()
                if macd is not None and not macd.empty:
                    df["MACD"] = macd["MACD_12_26_9"] - macd["MACDs_12_26_9"]
            elif name == "BB":
                bb = data.ta.bbands(length=settings.get("period", 20))
                if bb is not None and not bb.empty:
                    df["BB"] = bb[f"BBP_{settings.get('period', 20)}_2.0"]
            elif name == "CTI":
                direction = np.where(data["Close"].diff() > 0, 1, -1)
                magnitude = np.log(data["Close"].diff().abs() / data["Close"].shift(1) + 1)
                volatility_adj = data["Close"].pct_change().rolling(settings.get("period", 14)).std()
                cti = (direction * magnitude * (1 + volatility_adj)).ewm(span=settings.get("period", 14)).mean()
                df["CTI"] = cti * 100
            elif name == "VPIN":
                buy_vol = np.where(data["Close"] > data["Open"], data["Volume"], 0)
                sell_vol = np.where(data["Close"] < data["Open"], data["Volume"], 0)
                vol_diff = pd.Series(sell_vol).rolling(settings.get("period", 20)).sum() - pd.Series(buy_vol).rolling(settings.get("period", 20)).sum()
                total_vol = data["Volume"].rolling(settings.get("period", 20)).sum().replace(0, 1)
                df["VPIN"] = (vol_diff / total_vol) * 100
            elif name == "AMV":
                hl_range = data["High"] - data["Low"]
                tr = data.ta.atr(length=1)
                amv = hl_range.rolling(settings.get("period", 14)).std() / (tr.rolling(settings.get("period", 14)).mean() + 1e-10)
                df["AMV"] = amv * 100
            elif name == "TSD":
                p = settings.get("period", 10)
                ma1 = data["Close"].ewm(span=p).mean()
                ma2 = data["Close"].ewm(span=p*2).mean()
                ma3 = data["Close"].ewm(span=p*4).mean()
                tsd = (ma1 - ma2).abs() + (ma2 - ma3).abs() + (ma1 - ma3).abs()
                df["TSD"] = tsd / data["Close"] * 100
            elif name == "QMS":
                p = settings.get("period", 5)
                log_ret = np.log(data["Close"]/data["Close"].shift(1))
                df["QMS"] = log_ret.rolling(p).apply(lambda x: np.sqrt(np.sum(x**2)), raw=False) * 100
            elif name == "NVI":
                p = settings.get("period", 255)
                price_change = data["Close"].pct_change()
                nvi = pd.Series(1, index=data.index)
                for i in range(1, len(data)):
                    if data["Volume"].iloc[i] < data["Volume"].iloc[i-1]:
                        nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change.iloc[i])
                    else:
                        nvi.iloc[i] = nvi.iloc[i-1]
                df["NVI"] = nvi.rolling(p).mean()
            elif name == "PFE":
                p = settings.get("period", 14)
                pfe = (data["Close"] - data["Close"].shift(p)) / \
                      (np.sqrt((data["Close"].diff()**2 + 1e-10).rolling(p).sum()))
                df["PFE"] = pfe * 100
        except Exception as e:
            logger.warning(f"مشكلة في حساب المؤشر {name}: {e}")
            df[name] = np.nan
    return df

# معالجة الميزات وهدف النموذج
def prepare_features(data, indicators, lookback=5):
    X = indicators.copy()
    X["Price_Momentum"] = data["Close"].pct_change(5).rolling(10).mean()
    pct_change = data["Close"].pct_change().fillna(0)
    X["Volume_Force"] = (data["Volume"] * pct_change).rolling(5).sum()
    volatility = data["Close"].pct_change().rolling(20).std()
    trend = data["Close"].pct_change(20)
    product = (volatility * abs(trend)).fillna(0)
    if len(product.unique()) < 3:
        X["Market_Regime"] = 0
    else:
        X["Market_Regime"] = pd.qcut(product, q=3, labels=[-1, 0, 1], duplicates="drop")
    future_returns = data["Close"].shift(-lookback).pct_change(lookback)
    y = (future_returns > 0).astype(int)
    combined = pd.concat([X, y.rename("target")], axis=1).dropna()
    return combined.drop("target", axis=1), combined["target"]

# تدريب وحفظ/تحميل النموذج
def train_ai_model(X, y, params=MODEL_PARAMS):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], shuffle=False
    )
    model = make_pipeline(
        RobustScaler(),
        lgb.LGBMClassifier(
            **params["hyperparameters"],
            random_state=42,
            verbosity=-1
        )
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "دقة": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }
    joblib.dump(model, MODEL_PATH)
    return model, metrics

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            logger.warning(f"تعذر تحميل النموذج: {e}")
            return None
    return None

# توليد إشارة التداول
def predict_signal(model, X, threshold=0.8):
    if model is None or X.empty:
        return 0, 0.0
    try:
        proba = model.predict_proba(X)[-1, 1]
        confidence = abs(proba - 0.5) * 2
        if proba > threshold:
            return 1, confidence
        elif proba < (1 - threshold):
            return -1, confidence
        else:
            return 0, confidence
    except Exception as e:
        logger.warning(f"مشكلة في التنبؤ: {e}")
        return 0, 0.0

# واجهة Streamlit
st.set_page_config(page_title="لوحة تحكم التداول الذكية", layout="wide")
st.title("📈 لوحة تحكم التداول الذكية بالذكاء الاصطناعي")

with st.sidebar:
    st.header("الإعدادات")
    symbol = st.selectbox("اختر الأصل:", DEFAULT_PAIRS)
    period = st.selectbox("الفترة الزمنية:", ["1w", "1m", "3m", "6m", "1y", "2y"])
    st.markdown("---")
    st.subheader("المؤشرات الفنية")
    indicators_config = {}
    for name, settings in DEFAULT_INDICATORS.items():
        active = st.checkbox(f"{name}", value=settings.get("active", True))
        period_value = settings.get("period", None)
        if period_value is not None:
            val = st.number_input(f"فترة {name}", min_value=2, max_value=255, value=period_value)
            indicators_config[name] = {"active": active, "period": val}
        else:
            indicators_config[name] = {"active": active}
    st.markdown("---")
    retrain = st.button("إعادة تدريب النموذج")

tab1, tab2 = st.tabs(["الرسم البياني والمؤشرات", "ذكاء اصطناعي"])

with tab1:
    st.subheader(f"البيانات والمؤشرات لـ {symbol}")
    data_load_state = st.info("جاري تحميل البيانات...")
    try:
        data = get_data(symbol, period)
        data_load_state.success("تم تحميل البيانات بنجاح!")
        st.line_chart(data["Close"], use_container_width=True)
        indicators = calculate_indicators(data, indicators_config)
        st.dataframe(indicators.tail(20))
        for ind in indicators.columns:
            st.line_chart(indicators[ind].dropna(), use_container_width=True)
    except Exception as e:
        st.error(f"حدث خطأ أثناء جلب البيانات: {e}")

with tab2:
    st.subheader("ذكاء اصطناعي: توقع وإشارة")
    model = load_model()
    ai_status = ""
    if retrain or (model is None and "data" in locals()):
        if "data" in locals():
            X, y = prepare_features(data, indicators)
            try:
                model, metrics = train_ai_model(X, y)
                ai_status = "تم تدريب النموذج!"
                st.success(f"تم تدريب النموذج. المقاييس: {metrics}")
            except Exception as e:
                st.error(f"فشل التدريب: {e}")
        else:
            st.info("الرجاء تحميل البيانات أولاً.")
    elif model is not None and "data" in locals():
        X, y = prepare_features(data, indicators)
        signal, confidence = predict_signal(model, X, threshold=MODEL_PARAMS["threshold"])
        signal_map = {1: "شراء", -1: "بيع", 0: "حياد"}
        st.metric("إشارة التداول الحالية", signal_map[signal])
        st.metric("درجة الثقة", f"{confidence:.1%}")
        # تنبيه بصري فقط عند وجود إشارة واضحة
        if confidence >= 0.5 and signal != 0:
            st.warning(f"🚨 إشارة قوية: {signal_map[signal]} (ثقة: {confidence:.1%})")
        st.write("---")
        st.write("أداء النموذج (قد لا يكون دقيقًا إذا تغيرت الإعدادات):")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=MODEL_PARAMS["test_size"], shuffle=False
            )
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            st.write({
                "دقة": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC_AUC": roc_auc_score(y_test, y_proba)
            })
        except Exception as e:
            st.warning(f"لا يمكن حساب الأداء الحالي: {e}")
    else:
        st.info("اضغط 'إعادة تدريب النموذج' بعد تحميل البيانات والمؤشرات.")

st.markdown(
    """
    ---
    <div style="text-align:center; color:gray; font-size:small;">
    تم تطوير السكريبت ليعمل بسهولة على Streamlit Cloud بدون أي إعدادات بريد إلكتروني أو كلمات مرور.  
    جميع التنبيهات تظهر داخل الواجهة فقط.<br>
    <br>
    <b>الدعم الفني: djharga</b>
    </div>
    """,
    unsafe_allow_html=True
)