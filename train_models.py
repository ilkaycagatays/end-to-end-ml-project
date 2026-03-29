"""
=============================================================
LOJİSTİK OPTİMİZASYON PROJESİ - MAKİNE ÖĞRENMESİ MODELLERİ
=============================================================
Bu dosya üç farklı ML modeli eğitir:

  MODEL 1: Gecikme Tahmini (Classification)
    → Bir teslimat gecikmeli mi olacak? (Evet/Hayır)
    → Algoritma: Random Forest Classifier
    → Kullanım: Proaktif müdahale için erken uyarı sistemi

  MODEL 2: Gecikme Süresi Tahmini (Regression)
    → Gecikme kaç gün sürecek?
    → Algoritma: Gradient Boosting Regressor
    → Kullanım: Müşteri bilgilendirme sistemi

  MODEL 3: İş Yükü Tahmini / Talep Öngörüsü (Time Series)
    → Önümüzdeki 30 gün sipariş hacmi ne olacak?
    → Algoritma: Exponential Smoothing + Trend decomposition
    → Kullanım: Personel planlama, depo kapasitesi yönetimi

Neden bu modeller?
  → Junior DA için portföyde ML görmek kritik
  → Her model farklı bir problem türü (sınıflandırma, regresyon, zaman serisi)
  → Sonuçlar doğrudan iş kararlarına bağlanıyor
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import joblib
from datetime import datetime, timedelta

# Scikit-learn modelleri ve araçları
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.pipeline        import Pipeline
from sklearn.inspection      import permutation_importance

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# ÇIKTI DİZİNLERİ
# ──────────────────────────────────────────────────────────
os.makedirs("models",          exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed",  exist_ok=True)

print("\n" + "="*60)
print("  LOJİSTİK OPTİMİZASYON - ML MODELLERİ EĞİTİMİ")
print("="*60)


# ──────────────────────────────────────────────────────────
# VERİ YÜKLEME VE BİRLEŞTİRME
# ──────────────────────────────────────────────────────────

orders   = pd.read_csv("data/raw/orders.csv",    parse_dates=["order_date"])
delivery = pd.read_csv("data/raw/deliveries.csv",
                       parse_dates=["pickup_date",
                                    "promised_delivery_date",
                                    "actual_delivery_date"])
warehouse= pd.read_csv("data/raw/warehouse_performance.csv", parse_dates=["date"])

# Birleştir
df = pd.merge(orders, delivery, on="order_id",
              suffixes=("_ord", "_del"), how="inner")

# Sütun isim çakışmalarını çöz (suffix eklenmiş olanları kullan)
df = df.rename(columns={
    "destination_city_del": "destination_city",
    "origin_city_del":      "origin_city",
    "priority_ord":         "priority",
    "weight_kg_ord":        "weight_kg",
    "warehouse_id_ord":     "warehouse_id",
})

# Gereksiz duplicate sütunları düşür
df = df.drop(columns=[
    "destination_city_ord", "origin_city_ord",
    "priority_del", "weight_kg_del", "warehouse_id_del"
], errors="ignore")

print(f"✅ Veri yüklendi: {df.shape[0]:,} satır × {df.shape[1]} sütun")


# ──────────────────────────────────────────────────────────
# ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
# ──────────────────────────────────────────────────────────
# Özellik mühendisliği: Ham veriden ML modeline beslenecek
# özellikleri (feature) türetme sürecidir.
# Bu adım genellikle model performansını en çok etkileyen adımdır.

print("\n🔧 Özellik mühendisliği yapılıyor...")

# ── Zaman Özellikleri ──
df["order_month"]      = df["order_date"].dt.month
df["order_weekday"]    = df["order_date"].dt.dayofweek
df["order_quarter"]    = df["order_date"].dt.quarter
df["order_dayofyear"]  = df["order_date"].dt.dayofyear
df["is_weekend_order"] = (df["order_weekday"] >= 5).astype(int)
df["is_peak_season"]   = df["order_month"].isin([11, 12]).astype(int)
df["is_winter"]        = df["order_month"].isin([12, 1, 2]).astype(int)
df["is_summer"]        = df["order_month"].isin([6, 7, 8]).astype(int)

# ── Şehir Bazlı Tarihsel Gecikme Oranı ──
# Her şehir için geçmiş gecikme oranını özellik olarak ekle
# Bu "target encoding" olarak bilinir - kategorik değişkeni sayısala çevirir
city_delay_history = (
    df.groupby("destination_city")["is_delayed"]
    .mean()
    .reset_index()
    .rename(columns={"is_delayed": "city_historical_delay_rate"})
)
df = pd.merge(df, city_delay_history, on="destination_city", how="left")

# ── Depo Verimliliği ──
# O günkü depo verimliliğini siparişe ekle
# Her sipariş için ilgili deponu ve tarihi eşleştir
warehouse_daily = warehouse[["date", "warehouse_id",
                              "efficiency_score", "capacity_used_pct",
                              "error_rate_pct"]].copy()
warehouse_daily = warehouse_daily.rename(columns={"date": "order_date"})

df = pd.merge(
    df,
    warehouse_daily,
    on=["order_date", "warehouse_id"],
    how="left"
)

# Boş kalan verimlilik değerlerini median ile doldur
df["efficiency_score"]   = df["efficiency_score"].fillna(df["efficiency_score"].median())
df["capacity_used_pct"]  = df["capacity_used_pct"].fillna(df["capacity_used_pct"].median())
df["error_rate_pct"]     = df["error_rate_pct"].fillna(df["error_rate_pct"].median())

# ── Türetilmiş Özellikler ──
# Sipariş değeri / ağırlık = değer yoğunluğu
# Yüksek değerli+hafif → elektronik, Express tercih
df["value_per_kg"]     = df["order_value_tl"] / (df["weight_kg"] + 0.01)
df["cost_ratio"]       = df["delivery_cost_tl"] / (df["order_value_tl"] + 1)

# Priority ve fragile kombinasyonu (etkileşim özelliği)
# Kategori bazlı gecikme riski
high_risk_categories = ["Mobilya", "Elektronik", "Ev & Yaşam"]
df["is_high_risk_category"] = df["product_category"].isin(high_risk_categories).astype(int)

# ── Kategorik Kodlama ──
# ML modelleri sayısal veri ister, kategorik değişkenleri sayıya çevir
le = LabelEncoder()
categorical_cols = ["destination_city", "origin_city", "product_category",
                    "priority", "warehouse_id"]
for col in categorical_cols:
    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

# Priority sayısal sırası (Economy=0, Standard=1, Express=2)
priority_map = {"Economy": 0, "Standard": 1, "Express": 2}
df["priority_numeric"] = df["priority"].map(priority_map)

# is_fragile bool → int
df["is_fragile"] = df["is_fragile"].astype(int)

print(f"   ✅ {len(df.columns)} sütunlu zenginleştirilmiş veri seti hazır")

# İşlenmiş veriyi kaydet (notebook'larda tekrar kullanmak için)
df.to_csv("data/processed/combined_features.csv", index=False, encoding="utf-8-sig")
print("   💾 data/processed/combined_features.csv kaydedildi")


# ──────────────────────────────────────────────────────────
# MODEL 1: GECİKME TAHMİNİ (SINIFLANDIRMA)
# ──────────────────────────────────────────────────────────

print("\n" + "─"*50)
print("  MODEL 1: Gecikme Tahmini (Random Forest)")
print("─"*50)

# ── Feature Seçimi ──
# Hangi özellikleri modele besleyeceğiz?
# Kural: Hedef değişkeni (is_delayed) ve teslimat SONRASI bilinen
# değişkenleri (delay_days, actual_delivery_date) KESİNLİKLE EKLEME
# → Bu "data leakage" (veri sızıntısı) oluşturur ve sahte yüksek skor verir

FEATURES_CLF = [
    # Sipariş özellikleri (sipariş anında bilinir)
    "weight_kg",
    "order_value_tl",
    "quantity",
    "is_fragile",
    "priority_numeric",
    "value_per_kg",
    "cost_ratio",

    # Zaman özellikleri
    "order_month",
    "order_weekday",
    "order_quarter",
    "is_weekend_order",
    "is_peak_season",
    "is_winter",

    # Coğrafi özellikler
    "destination_city_encoded",
    "city_historical_delay_rate",
    "is_high_risk_category",

    # Depo özellikleri
    "efficiency_score",
    "capacity_used_pct",
    "error_rate_pct",
    "warehouse_id_encoded",
]

TARGET_CLF = "is_delayed"  # 0 = Zamanında, 1 = Gecikmeli

# Eksik değerleri kontrol et
feature_df = df[FEATURES_CLF + [TARGET_CLF, "delay_days"]].dropna()
print(f"   Eğitim seti boyutu: {len(feature_df):,} satır")

X = feature_df[FEATURES_CLF]
y = feature_df[TARGET_CLF]

# ── Train/Test Bölme ──
# %80 eğitim, %20 test
# stratify=y → dengesiz sınıflarda her bölmede aynı oran korunur
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Eğitim: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Gecikme oranı (train): %{y_train.mean()*100:.1f}")

# ── Random Forest Modeli ──
# Random Forest: Birden fazla karar ağacının (decision tree) oylamasıyla tahmin yapar
# Avantajları:
#   - Aşırı öğrenmeye (overfitting) karşı dayanıklı
#   - Özellik önemliliği (feature importance) verir
#   - Hiperparametre ayarına az duyarlı
#   - Hem sınıflandırma hem regresyon için kullanılır

rf_model = RandomForestClassifier(
    n_estimators=200,        # 200 karar ağacı
    max_depth=10,            # Her ağacın max derinliği (overfitting önler)
    min_samples_leaf=20,     # Yaprak düğümde min örnek (genelleme için)
    class_weight="balanced", # Dengesiz sınıf durumunda ağırlık dengele
    random_state=42,
    n_jobs=-1                # Tüm CPU çekirdeğini kullan
)

# Cross-validation: Modeli 5 farklı bölümde eğitip test et
# Tek bir train/test bölmesine kıyasla daha güvenilir sonuç verir
print("\n   Cross-validation yapılıyor (5-fold)...")
cv_scores = cross_val_score(
    rf_model, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",      # ROC-AUC: dengesiz sınıflar için uygun metrik
    n_jobs=-1
)
print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Final model eğitimi (tüm train seti ile)
rf_model.fit(X_train, y_train)

# ── Test Seti Değerlendirme ──
y_pred_clf     = rf_model.predict(X_test)        # Sınıf tahmini (0/1)
y_prob_clf     = rf_model.predict_proba(X_test)[:, 1]  # Gecikme olasılığı (0-1)

roc_auc = roc_auc_score(y_test, y_prob_clf)
print(f"\n   Test ROC-AUC:  {roc_auc:.4f}")
print(f"   Test Accuracy: {(y_pred_clf == y_test).mean():.4f}")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred_clf,
      target_names=["Zamanında", "Gecikmeli"]))

# ── Modeli Kaydet ──
joblib.dump(rf_model, "models/delay_classifier.pkl")
print("   💾 models/delay_classifier.pkl kaydedildi")

# ── Görselleştirme: Confusion Matrix + ROC Curve + Feature Importance ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model 1: Gecikme Tahmini (Random Forest)", fontsize=15, fontweight="bold")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_clf)
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Blues",
    xticklabels=["Zamanında", "Gecikmeli"],
    yticklabels=["Zamanında", "Gecikmeli"],
    ax=axes[0],
    linewidths=0.5
)
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Gerçek Durum")
axes[0].set_xlabel("Tahmin Edilen")

# ROC Eğrisi
# ROC: Hassasiyet (True Positive Rate) vs Yanlış Alarm (False Positive Rate)
# AUC = 1.0 → Mükemmel model, 0.5 → Rastgele tahmin
fpr, tpr, _ = roc_curve(y_test, y_prob_clf)
axes[1].plot(fpr, tpr,
             color="#1B4F72", linewidth=2.5,
             label=f"ROC (AUC = {roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1],
             linestyle="--", color="#7F8C8D",
             label="Rastgele Tahmin (AUC=0.5)")
axes[1].fill_between(fpr, tpr, alpha=0.1, color="#1B4F72")
axes[1].set_xlabel("False Positive Rate (Yanlış Alarm Oranı)")
axes[1].set_ylabel("True Positive Rate (Doğru Yakalama Oranı)")
axes[1].set_title("ROC Eğrisi")
axes[1].legend()

# Feature Importance (Özellik Önemliliği)
importance_df = pd.DataFrame({
    "feature":    FEATURES_CLF,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=True).tail(15)  # Top 15

colors_imp = ["#1B4F72" if v > 0.05 else "#AED6F1" for v in importance_df["importance"]]
axes[2].barh(importance_df["feature"], importance_df["importance"],
             color=colors_imp, edgecolor="white")
axes[2].set_title("Özellik Önemliliği (Top 15)")
axes[2].set_xlabel("Önem Skoru")

plt.tight_layout()
plt.savefig("reports/figures/07_delay_classifier_results.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ reports/figures/07_delay_classifier_results.png kaydedildi")


# ──────────────────────────────────────────────────────────
# MODEL 2: GECİKME SÜRESİ TAHMİNİ (REGRESYON)
# ──────────────────────────────────────────────────────────

print("\n" + "─"*50)
print("  MODEL 2: Gecikme Süresi Tahmini (Gradient Boosting)")
print("─"*50)

# Sadece gecikmeli teslimatları al
delayed_df = feature_df[feature_df["is_delayed"] == 1].copy()
print(f"   Gecikmeli teslimat sayısı: {len(delayed_df):,}")

TARGET_REG = "delay_days"

X_reg = delayed_df[FEATURES_CLF]
y_reg = delayed_df[TARGET_REG]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# Gradient Boosting Regressor
# Her iterasyonda önceki modelin hatalarını düzeltecek yeni ağaç ekler
# Random Forest'tan farkı: paralel değil, sıralı öğrenir
# Genellikle daha yüksek doğruluk ama daha yavaş ve overfitting riski var

gb_model = GradientBoostingRegressor(
    n_estimators=200,    # 200 iterasyon
    max_depth=5,         # Her ağacın max derinliği
    learning_rate=0.05,  # Küçük öğrenme hızı → daha iyi genelleme
    subsample=0.8,       # Her iterasyonda %80 örnekle eğit (stochastic)
    min_samples_leaf=10,
    random_state=42
)

gb_model.fit(X_train_r, y_train_r)
y_pred_reg = gb_model.predict(X_test_r)

# Regresyon metrikleri
mae  = mean_absolute_error(y_test_r, y_pred_reg)   # Ort. Mutlak Hata
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))  # Kök Ort. Kare Hata
r2   = r2_score(y_test_r, y_pred_reg)               # R² (1 = mükemmel fit)

print(f"\n   MAE  (Ort. Mutlak Hata):  {mae:.3f} gün")
print(f"   RMSE (Kök Ort. Kare Hata): {rmse:.3f} gün")
print(f"   R²   (Açıklama Oranı):     {r2:.4f}")

joblib.dump(gb_model, "models/delay_regressor.pkl")
print("   💾 models/delay_regressor.pkl kaydedildi")

# ── Görselleştirme: Regresyon ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model 2: Gecikme Süresi Tahmini (Gradient Boosting)",
             fontsize=15, fontweight="bold")

# Gerçek vs Tahmin scatter
axes[0].scatter(y_test_r, y_pred_reg, alpha=0.3, color="#1B4F72", s=20)
# Mükemmel tahmin çizgisi (45 derece)
lims = [0, max(y_test_r.max(), y_pred_reg.max()) + 0.5]
axes[0].plot(lims, lims, "--", color="#E74C3C", linewidth=2, label="Mükemmel Tahmin")
axes[0].set_xlabel("Gerçek Gecikme (gün)")
axes[0].set_ylabel("Tahmin Edilen Gecikme (gün)")
axes[0].set_title(f"Gerçek vs Tahmin\n(R² = {r2:.3f})")
axes[0].legend()

# Hata dağılımı (residuals histogram)
residuals = y_test_r - y_pred_reg
axes[1].hist(residuals, bins=30, color="#1B4F72",
             edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="#E74C3C", linestyle="--", linewidth=2)
axes[1].set_xlabel("Hata (Gerçek - Tahmin)")
axes[1].set_ylabel("Frekans")
axes[1].set_title(f"Hata Dağılımı\n(MAE = {mae:.2f} gün)")

# Feature importance
gb_importance = pd.DataFrame({
    "feature":    FEATURES_CLF,
    "importance": gb_model.feature_importances_
}).sort_values("importance", ascending=True).tail(12)

axes[2].barh(gb_importance["feature"], gb_importance["importance"],
             color="#1B4F72", edgecolor="white")
axes[2].set_title("Özellik Önemliliği (Gradient Boosting)")
axes[2].set_xlabel("Önem Skoru")

plt.tight_layout()
plt.savefig("reports/figures/08_delay_regressor_results.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ reports/figures/08_delay_regressor_results.png kaydedildi")


# ──────────────────────────────────────────────────────────
# MODEL 3: İŞ YÜKÜ TAHMİNİ (ZAMAN SERİSİ)
# ──────────────────────────────────────────────────────────

print("\n" + "─"*50)
print("  MODEL 3: İş Yükü ve Talep Tahmini (Zaman Serisi)")
print("─"*50)

# Günlük sipariş hacmi hazırla
daily_orders = (
    df.groupby("order_date")
    .agg(
        order_count       = ("order_id",    "count"),
        total_weight      = ("weight_kg",   "sum"),
        total_value       = ("order_value_tl","sum"),
        delay_rate        = ("is_delayed",  "mean"),
    )
    .reset_index()
    .sort_values("order_date")
)

# Zaman serisi özellikleri ekle
daily_orders["day_of_week"]  = daily_orders["order_date"].dt.dayofweek
daily_orders["month"]        = daily_orders["order_date"].dt.month
daily_orders["day_of_year"]  = daily_orders["order_date"].dt.dayofyear
daily_orders["week_of_year"] = daily_orders["order_date"].dt.isocalendar().week.astype(int)
daily_orders["is_weekend"]   = (daily_orders["day_of_week"] >= 5).astype(int)
daily_orders["is_peak"]      = daily_orders["month"].isin([11, 12]).astype(int)

# ── Hareketli Ortalama ve Üstel Düzleştirme ──
# Trend ve mevsimselliği ayrıştır

# 7 günlük hareketli ortalama (haftalık döngüyü düzleştirir)
daily_orders["ma_7"]  = daily_orders["order_count"].rolling(7,  min_periods=1).mean()
# 30 günlük hareketli ortalama (aylık trendi yakalar)
daily_orders["ma_30"] = daily_orders["order_count"].rolling(30, min_periods=1).mean()

# Üstel Ağırlıklı Hareketli Ortalama (EWMA)
# Son gözlemlere daha fazla ağırlık verir (α = span'ın tersi)
daily_orders["ewma_7"]  = daily_orders["order_count"].ewm(span=7,  adjust=False).mean()
daily_orders["ewma_30"] = daily_orders["order_count"].ewm(span=30, adjust=False).mean()

# ── Mevsimsel Ayrıştırma ──
# Trend + Mevsimsellik + Hata bileşenlerini ayır
# Basit yaklaşım: gözlem / trend = mevsimsel faktör
daily_orders["seasonal_factor"] = (
    daily_orders["order_count"] / daily_orders["ma_30"].replace(0, np.nan)
).fillna(1.0)

# Gün ve aya göre ortalama mevsimsel faktör hesapla
day_seasonal = (
    daily_orders.groupby("day_of_week")["seasonal_factor"]
    .mean()
    .reset_index()
    .rename(columns={"seasonal_factor": "day_seasonal_factor"})
)
month_seasonal = (
    daily_orders.groupby("month")["seasonal_factor"]
    .mean()
    .reset_index()
    .rename(columns={"seasonal_factor": "month_seasonal_factor"})
)

# ── 30 Günlük Tahmin ──
# Son trendin devamını ve mevsimsel faktörleri kullanarak tahmin yap
last_date   = daily_orders["order_date"].max()
last_ma_30  = daily_orders["ma_30"].iloc[-1]
last_trend  = (daily_orders["ma_30"].iloc[-1] - daily_orders["ma_30"].iloc[-7]) / 7

forecast_dates = [last_date + timedelta(days=i+1) for i in range(30)]
forecast_df = pd.DataFrame({"date": forecast_dates})
forecast_df["day_of_week"]  = forecast_df["date"].dt.dayofweek
forecast_df["month"]        = forecast_df["date"].dt.month

# Gün ve ay faktörlerini ekle
forecast_df = pd.merge(forecast_df, day_seasonal,   on="day_of_week", how="left")
forecast_df = pd.merge(forecast_df, month_seasonal,  on="month",       how="left")

# Temel tahmin: Son trend + mevsimsel düzeltme
for i, row in forecast_df.iterrows():
    days_ahead = (row["date"] - last_date).days
    # Trend devam ettirilir ama sınırlandırılır (sonsuz büyüme olmaz)
    trend_component = last_ma_30 + last_trend * days_ahead * 0.5
    # Mevsimsel faktörlerle çarp
    forecast_df.at[i, "forecast"] = max(
        10,
        trend_component * row["day_seasonal_factor"] * row["month_seasonal_factor"]
    )

# Belirsizlik aralığı (±15% confidence interval)
forecast_df["lower_bound"] = forecast_df["forecast"] * 0.85
forecast_df["upper_bound"] = forecast_df["forecast"] * 1.15

# ── Personel Planlama Tavsiyesi ──
# Tahmin edilen sipariş hacmine göre gerekli sürücü sayısı
AVG_DELIVERIES_PER_DRIVER = 12  # Bir sürücü günde ortalama 12 teslimat yapar
BUFFER_FACTOR = 1.15             # %15 tampon kapasite

forecast_df["required_drivers"] = np.ceil(
    forecast_df["forecast"] / AVG_DELIVERIES_PER_DRIVER * BUFFER_FACTOR
).astype(int)

forecast_df.to_csv("data/processed/demand_forecast.csv", index=False, encoding="utf-8-sig")
print(f"   💾 data/processed/demand_forecast.csv kaydedildi")
print(f"   📅 Tahmin dönemi: {forecast_df['date'].min().date()} → {forecast_df['date'].max().date()}")
print(f"   📊 Ort. tahmin edilen günlük sipariş: {forecast_df['forecast'].mean():.0f}")
print(f"   👤 Ort. gerekli sürücü sayısı: {forecast_df['required_drivers'].mean():.0f}")

# ── Zaman Serisi Görselleştirmesi ──
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Model 3: İş Yükü Tahmini ve Personel Planlama",
             fontsize=15, fontweight="bold")

# Sol üst: Tarihsel sipariş hacmi + hareketli ortalamalar
ax = axes[0, 0]
# Son 3 ay veri (daha net görünüm için)
recent_data = daily_orders.tail(90)
ax.plot(recent_data["order_date"], recent_data["order_count"],
        alpha=0.4, color="#AED6F1", linewidth=1, label="Günlük Sipariş")
ax.plot(recent_data["order_date"], recent_data["ma_7"],
        color="#1B4F72", linewidth=2, label="7 Günlük MA")
ax.plot(recent_data["order_date"], recent_data["ma_30"],
        color="#E74C3C", linewidth=2.5, label="30 Günlük MA")
ax.set_title("Son 90 Gün: Sipariş Hacmi + Hareketli Ortalamalar")
ax.set_ylabel("Günlük Sipariş")
ax.legend()

# Sağ üst: 30 Günlük Tahmin
ax = axes[0, 1]
# Son 30 gün gerçek + 30 gün tahmin
last_30_actual = daily_orders.tail(30)
ax.plot(last_30_actual["order_date"], last_30_actual["order_count"],
        color="#1B4F72", linewidth=2, label="Gerçek Sipariş")
ax.plot(forecast_df["date"], forecast_df["forecast"],
        color="#E74C3C", linewidth=2.5, linestyle="--", label="Tahmin")
ax.fill_between(
    forecast_df["date"],
    forecast_df["lower_bound"],
    forecast_df["upper_bound"],
    alpha=0.2, color="#E74C3C", label="±%15 Güven Aralığı"
)
ax.axvline(last_date, color="#F39C12", linestyle="--",
           linewidth=1.5, alpha=0.8, label="Tahmin Başlangıcı")
ax.set_title("30 Günlük Talep Tahmini")
ax.set_ylabel("Günlük Sipariş")
ax.legend(fontsize=8)

# Sol alt: Haftanın günlerine göre mevsimsel faktör
ax = axes[1, 0]
weekday_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
day_sf = day_seasonal.copy()
day_sf["day_name"] = [weekday_names[d] for d in day_sf["day_of_week"]]
bar_colors = [
    "#E74C3C" if f > 1.05 else
    "#27AE60" if f < 0.95 else
    "#1B4F72"
    for f in day_sf["day_seasonal_factor"]
]
ax.bar(day_sf["day_name"], day_sf["day_seasonal_factor"],
       color=bar_colors, edgecolor="white", width=0.7)
ax.axhline(1.0, color="#7F8C8D", linestyle="--", linewidth=1.5, label="Taban (1.0)")
ax.set_title("Haftanın Günlerine Göre Mevsimsel Faktör")
ax.set_ylabel("Mevsimsel Faktör")
ax.set_xlabel("Gün (>1.0 = normalden yoğun)")
ax.legend()

# Sağ alt: Personel planlama tavsiyesi
ax = axes[1, 1]
# Gerekli sürücü sayısı bar grafiği
date_labels = [d.strftime("%d %b") for d in forecast_df["date"]]
driver_colors = [
    "#E74C3C" if d > forecast_df["required_drivers"].quantile(0.75) else
    "#F39C12" if d > forecast_df["required_drivers"].median() else
    "#27AE60"
    for d in forecast_df["required_drivers"]
]
ax.bar(range(len(forecast_df)), forecast_df["required_drivers"],
       color=driver_colors, edgecolor="white", width=0.8)
ax.set_xticks(range(0, len(forecast_df), 5))
ax.set_xticklabels([date_labels[i] for i in range(0, len(forecast_df), 5)],
                   rotation=30, fontsize=9)
ax.set_title("30 Gün Personel Planlama Tavsiyesi")
ax.set_ylabel("Gerekli Sürücü Sayısı")
ax.axhline(forecast_df["required_drivers"].mean(),
           color="#1B4F72", linestyle="--", linewidth=1.5,
           label=f"Ortalama: {forecast_df['required_drivers'].mean():.0f}")
ax.legend()

plt.tight_layout()
plt.savefig("reports/figures/09_demand_forecast.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ reports/figures/09_demand_forecast.png kaydedildi")


# ──────────────────────────────────────────────────────────
# MODEL PERFORMANS ÖZETİ
# ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  TÜM MODEL SONUÇLARI ÖZETİ")
print("="*60)
print(f"  Model 1 - Gecikme Sınıflandırıcı:")
print(f"    CV ROC-AUC:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"    Test ROC-AUC: {roc_auc:.4f}")
print(f"    Accuracy:     {(y_pred_clf == y_test).mean():.4f}")
print(f"\n  Model 2 - Gecikme Süresi Tahmini:")
print(f"    MAE:  {mae:.3f} gün")
print(f"    RMSE: {rmse:.3f} gün")
print(f"    R²:   {r2:.4f}")
print(f"\n  Model 3 - Talep Tahmini (30 Gün):")
print(f"    Ort. günlük sipariş tahmini: {forecast_df['forecast'].mean():.0f}")
print(f"    Pik günlük sipariş:          {forecast_df['forecast'].max():.0f}")
print(f"    Min günlük sipariş:          {forecast_df['forecast'].min():.0f}")
print("="*60)
print("\n✅ Tüm modeller eğitildi ve kaydedildi.")
