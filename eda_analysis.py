"""
=============================================================
LOJİSTİK OPTİMİZASYON PROJESİ - KEŞİFSEL VERİ ANALİZİ (EDA)
=============================================================
Bu dosya veriyi ilk kez analiz eder ve iş sorularını yanıtlar:
  ❓ Hangi bölgelerde gecikme daha fazla?
  ❓ Gecikmenin ana sebepleri neler?
  ❓ Hangi ürün kategorileri sorun çıkarıyor?
  ❓ Mevsimsel örüntüler var mı?
  ❓ Depo verimliliği nasıl?
  ❓ Sürücü performansları nasıl dağılıyor?

Her analiz sonunda ticari yorum (Business Insight) verilir.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# GÖRSEL AYARLAR
# ──────────────────────────────────────────────────────────

# Matplotlib tema: temiz, profesyonel görünüm
plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",  # Açık gri arka plan
    "axes.facecolor":   "#FFFFFF",  # Grafik alanı beyaz
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "grid.linestyle":   "--",
    "axes.spines.top":  False,      # Üst çerçeve kaldır
    "axes.spines.right":False,      # Sağ çerçeve kaldır
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
})

# Renk paleti (kurumsal mavi-kırmızı temalı)
COLORS = {
    "primary":  "#1B4F72",   # Koyu mavi
    "secondary":"#E74C3C",   # Kırmızı (uyarı)
    "success":  "#27AE60",   # Yeşil (iyi)
    "warning":  "#F39C12",   # Turuncu
    "neutral":  "#7F8C8D",   # Gri
    "light":    "#AED6F1",   # Açık mavi
}

# Çıktı dizini
os.makedirs("reports/figures", exist_ok=True)


# ──────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  VERİ YÜKLEME VE ÖN İŞLEME")
print("="*60)

# CSV dosyalarını yükle
orders   = pd.read_csv("data/raw/orders.csv",    parse_dates=["order_date"])
delivery = pd.read_csv("data/raw/deliveries.csv",
                       parse_dates=["pickup_date",
                                    "promised_delivery_date",
                                    "actual_delivery_date"])
warehouse= pd.read_csv("data/raw/warehouse_performance.csv", parse_dates=["date"])
drivers  = pd.read_csv("data/raw/driver_performance.csv")

# Siparişler ve teslimatları tek tabloda birleştir
# pd.merge → SQL'deki JOIN ile aynı işlev
# "inner" join → her iki tabloda da bulunan kayıtları al
combined = pd.merge(
    orders,
    delivery,
    on="order_id",      # Birleşme anahtarı
    suffixes=("_ord","_del"),  # Aynı isimli sütunlara suffix ekle
    how="inner"
)

# Zaman boyutları ekle (sonraki analizlerde kullanacağız)
combined["order_month"]   = combined["order_date"].dt.month
combined["order_quarter"] = combined["order_date"].dt.quarter
combined["order_year"]    = combined["order_date"].dt.year
combined["order_weekday"] = combined["order_date"].dt.dayofweek  # 0=Pzt, 6=Paz
combined["order_week"]    = combined["order_date"].dt.isocalendar().week.astype(int)

print(f"✅ Birleştirilmiş tablo: {combined.shape[0]:,} satır × {combined.shape[1]} sütun")
print(f"   Tarih aralığı: {combined['order_date'].min().date()} → {combined['order_date'].max().date()}")


# ──────────────────────────────────────────────────────────
# ANALİZ 1: BÖLGESEL GECİKME HARİTASI
# ──────────────────────────────────────────────────────────

print("\n📊 Analiz 1: Bölgesel gecikme oranları hesaplanıyor...")

# Her şehir için: toplam teslimat sayısı, gecikme sayısı, gecikme oranı, ort. gecikme
# groupby → şehirlere göre grupla
# agg → her grup için birden fazla hesaplama
city_delays = (
    combined
    .groupby("destination_city_del")
    .agg(
        total_deliveries = ("delivery_id", "count"),       # Kaç teslimat oldu
        delayed_count    = ("is_delayed",   "sum"),        # Kaç gecikme
        avg_delay_days   = ("delay_days",   "mean"),       # Ortalama gecikme günü
        avg_satisfaction = ("customer_satisfaction","mean"),# Ort. memnuniyet
        total_revenue    = ("order_value_tl","sum"),        # Toplam sipariş değeri
    )
    .reset_index()
)

# Gecikme oranı hesapla (yüzde olarak)
city_delays["delay_rate"] = city_delays["delayed_count"] / city_delays["total_deliveries"] * 100

# Gecikme oranına göre büyükten küçüğe sırala
city_delays = city_delays.sort_values("delay_rate", ascending=False)

# ── Görselleştirme ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Şehir Bazlı Gecikme Analizi", fontsize=16, fontweight="bold", y=1.02)

# Sol grafik: Gecikme oranı bar grafiği
# Renk: %25 üstü kırmızı, %15-25 turuncu, altı yeşil
bar_colors = [
    COLORS["secondary"] if r > 25 else
    COLORS["warning"]   if r > 15 else
    COLORS["success"]
    for r in city_delays["delay_rate"]
]

bars = axes[0].barh(
    city_delays["destination_city_del"],  # Y ekseni: şehir isimleri
    city_delays["delay_rate"],         # X ekseni: gecikme oranı
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
    height=0.7
)

# Her barın yanına değer yaz
for bar, val in zip(bars, city_delays["delay_rate"]):
    axes[0].text(
        val + 0.3,          # X pozisyon (barın biraz sağı)
        bar.get_y() + bar.get_height() / 2,  # Y merkezi
        f"{val:.1f}%",
        va="center", fontsize=9, fontweight="bold"
    )

# Kritik eşik çizgisi (%20)
axes[0].axvline(20, color=COLORS["secondary"], linestyle="--",
                linewidth=1.5, alpha=0.7, label="Kritik Eşik (%20)")
axes[0].legend(fontsize=9)
axes[0].set_title("Şehir Bazlı Gecikme Oranı (%)")
axes[0].set_xlabel("Gecikme Oranı (%)")
axes[0].invert_yaxis()  # En yüksek üstte

# Sağ grafik: Gecikme oranı vs Müşteri Memnuniyeti (scatter)
scatter = axes[1].scatter(
    city_delays["delay_rate"],       # X: gecikme oranı
    city_delays["avg_satisfaction"], # Y: memnuniyet
    s=city_delays["total_deliveries"] / 5,  # Nokta büyüklüğü = teslimat hacmi
    c=city_delays["delay_rate"],     # Renk = gecikme oranı
    cmap="RdYlGn_r",                 # Kırmızı (yüksek gecikme) → Yeşil
    alpha=0.7,
    edgecolors="white",
    linewidth=0.5
)

# Şehir isimlerini noktaların yanına yaz
for _, row in city_delays.iterrows():
    axes[1].annotate(
        row["destination_city_del"],
        (row["delay_rate"], row["avg_satisfaction"]),
        fontsize=8, ha="left", va="bottom",
        xytext=(3, 3), textcoords="offset points"
    )

plt.colorbar(scatter, ax=axes[1], label="Gecikme Oranı (%)")
axes[1].set_title("Gecikme Oranı vs Müşteri Memnuniyeti")
axes[1].set_xlabel("Gecikme Oranı (%)")
axes[1].set_ylabel("Ort. Müşteri Memnuniyeti (1-5)")

plt.tight_layout()
plt.savefig("reports/figures/01_city_delay_analysis.png",
            dpi=150, bbox_inches="tight")
plt.close()

print("   ✅ reports/figures/01_city_delay_analysis.png kaydedildi")
print(f"\n   📌 İÇ GÖRÜ: En yüksek gecikme → {city_delays.iloc[0]['destination_city_del']} "
      f"(%{city_delays.iloc[0]['delay_rate']:.1f})")
print(f"   📌 İÇ GÖRÜ: En düşük gecikme → {city_delays.iloc[-1]['destination_city_del']} "
      f"(%{city_delays.iloc[-1]['delay_rate']:.1f})")


# ──────────────────────────────────────────────────────────
# ANALİZ 2: GECİKMENİN ANA SEBEPLERİ
# ──────────────────────────────────────────────────────────

print("\n📊 Analiz 2: Gecikme sebepleri analiz ediliyor...")

# Sadece gecikmeli teslimatları al
delayed_only = combined[combined["is_delayed"] == True].copy()

# Sebep bazında gecikme sayısı
# value_counts() → her değerin kaç kez tekrarlandığını sayar
reason_counts = delayed_only["delay_reason"].value_counts().reset_index()
reason_counts.columns = ["delay_reason", "count"]

# Kümülatif oran hesapla (Pareto analizi için)
# Hangi sebepler toplam gecikmenin %80'ini oluşturuyor? (80/20 kuralı)
reason_counts["cumulative_pct"] = (
    reason_counts["count"].cumsum() / reason_counts["count"].sum() * 100
)
reason_counts["pct"] = reason_counts["count"] / reason_counts["count"].sum() * 100

# Ortalama gecikme süresi (sebep bazında)
avg_delay_by_reason = (
    delayed_only
    .groupby("delay_reason")["delay_days"]
    .mean()
    .reset_index()
    .rename(columns={"delay_days": "avg_delay_days"})
)

reason_counts = pd.merge(reason_counts, avg_delay_by_reason, on="delay_reason")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Gecikme Sebepleri Analizi (Pareto + Etki)", fontsize=16, fontweight="bold")

# Sol: Pareto grafiği
# Pareto → 80/20 kuralını görselleştirir (hangi az sebep çoğu sorunu yaratıyor)
ax1 = axes[0]
ax2 = ax1.twinx()  # İkinci Y ekseni (kümülatif yüzde için)

# Bar grafiği (sol Y ekseni)
bars = ax1.bar(
    range(len(reason_counts)),
    reason_counts["count"],
    color=COLORS["primary"],
    alpha=0.8,
    width=0.6
)

# Kümülatif çizgi (sağ Y ekseni)
ax2.plot(
    range(len(reason_counts)),
    reason_counts["cumulative_pct"],
    color=COLORS["secondary"],
    marker="o",
    linewidth=2,
    markersize=6,
    label="Kümülatif %"
)

# %80 eşik çizgisi
ax2.axhline(80, color=COLORS["warning"], linestyle="--",
            linewidth=1.5, alpha=0.8, label="%80 Eşiği")
ax2.legend(loc="center right", fontsize=9)

# X ekseni etiketleri
ax1.set_xticks(range(len(reason_counts)))
ax1.set_xticklabels(reason_counts["delay_reason"],
                     rotation=35, ha="right", fontsize=9)
ax1.set_ylabel("Gecikme Sayısı")
ax2.set_ylabel("Kümülatif Yüzde (%)")
ax1.set_title("Pareto Analizi: Gecikme Sebepleri")

# Sağ: Sebep başına ortalama gecikme süresi (gün)
reason_sorted = reason_counts.sort_values("avg_delay_days", ascending=True)
colors_by_severity = [
    COLORS["secondary"] if d > 4 else
    COLORS["warning"]   if d > 2 else
    COLORS["success"]
    for d in reason_sorted["avg_delay_days"]
]

axes[1].barh(
    reason_sorted["delay_reason"],
    reason_sorted["avg_delay_days"],
    color=colors_by_severity,
    edgecolor="white"
)
for i, (val, name) in enumerate(zip(reason_sorted["avg_delay_days"],
                                     reason_sorted["delay_reason"])):
    axes[1].text(val + 0.05, i, f"{val:.1f} gün", va="center", fontsize=9)

axes[1].set_title("Sebep Başına Ortalama Gecikme Süresi (gün)")
axes[1].set_xlabel("Ortalama Gecikme (gün)")

plt.tight_layout()
plt.savefig("reports/figures/02_delay_reasons_pareto.png",
            dpi=150, bbox_inches="tight")
plt.close()

# En kritik 3 sebep
top3 = reason_counts.head(3)
print("   📌 İÇ GÖRÜ: En kritik 3 gecikme sebebi:")
for _, row in top3.iterrows():
    print(f"      → {row['delay_reason']}: {row['count']:,} vaka (%{row['pct']:.1f}), "
          f"ort. {row['avg_delay_days']:.1f} gün")


# ──────────────────────────────────────────────────────────
# ANALİZ 3: MEVSİMSELLİK VE ZAMAN SERİSİ
# ──────────────────────────────────────────────────────────

print("\n📊 Analiz 3: Mevsimsel örüntüler inceleniyor...")

# Aylık sipariş hacmi ve gecikme oranı
monthly = (
    combined
    .groupby(["order_year", "order_month"])
    .agg(
        order_count  = ("order_id",   "count"),
        delay_rate   = ("is_delayed", "mean"),
        avg_cost     = ("delivery_cost_tl", "mean"),
    )
    .reset_index()
)

# Tarih sütunu oluştur (grafik ekseni için)
monthly["period"] = pd.to_datetime(
    monthly["order_year"].astype(str) + "-" + monthly["order_month"].astype(str),
    format="%Y-%m"
)
monthly = monthly.sort_values("period")

# Ay isimleri (Türkçe)
month_names = {
    1:"Oca", 2:"Şub", 3:"Mar", 4:"Nis", 5:"May", 6:"Haz",
    7:"Tem", 8:"Ağu", 9:"Eyl", 10:"Eki", 11:"Kas", 12:"Ara"
}

# Haftalık sipariş dağılımı (haftanın hangi günü daha yoğun?)
weekday_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
weekday_stats = (
    combined
    .groupby("order_weekday")
    .agg(
        order_count = ("order_id",   "count"),
        delay_rate  = ("is_delayed", "mean"),
    )
    .reset_index()
)
weekday_stats["day_name"] = weekday_stats["order_weekday"].map(
    dict(enumerate(weekday_names))
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Mevsimsel ve Zaman Bazlı Analiz", fontsize=16, fontweight="bold")

# Sol üst: Aylık sipariş hacmi
ax = axes[0, 0]
ax.fill_between(monthly["period"], monthly["order_count"],
                alpha=0.3, color=COLORS["primary"])
ax.plot(monthly["period"], monthly["order_count"],
        color=COLORS["primary"], linewidth=2.5, marker="o", markersize=4)

# Kasım-Aralık vurgulama
for _, row in monthly[monthly["order_month"].isin([11, 12])].iterrows():
    ax.axvspan(row["period"] - pd.Timedelta(days=15),
               row["period"] + pd.Timedelta(days=15),
               alpha=0.1, color=COLORS["secondary"])

ax.set_title("Aylık Sipariş Hacmi")
ax.set_ylabel("Sipariş Sayısı")
ax.set_xlabel("")

# Sağ üst: Aylık gecikme oranı trendi
ax = axes[0, 1]
ax.fill_between(monthly["period"], monthly["delay_rate"] * 100,
                alpha=0.3, color=COLORS["secondary"])
ax.plot(monthly["period"], monthly["delay_rate"] * 100,
        color=COLORS["secondary"], linewidth=2.5, marker="o", markersize=4)
ax.axhline(monthly["delay_rate"].mean() * 100,
           color=COLORS["neutral"], linestyle="--", linewidth=1,
           label=f"Ortalama: %{monthly['delay_rate'].mean()*100:.1f}")
ax.legend()
ax.set_title("Aylık Gecikme Oranı Trendi")
ax.set_ylabel("Gecikme Oranı (%)")

# Sol alt: Haftanın günlerine göre sipariş dağılımı
ax = axes[1, 0]
bar_colors_wd = [
    COLORS["secondary"] if d >= 5 else  # Hafta sonu kırmızı
    COLORS["primary"]
    for d in weekday_stats["order_weekday"]
]
ax.bar(weekday_stats["day_name"], weekday_stats["order_count"],
       color=bar_colors_wd, edgecolor="white", width=0.7)
ax.set_title("Haftanın Günlerine Göre Sipariş Dağılımı")
ax.set_ylabel("Sipariş Sayısı")
ax.set_xlabel("Gün")

# Sağ alt: Öncelik bazında gecikme oranı (kutu grafiği)
ax = axes[1, 1]
priority_order = ["Express", "Standard", "Economy"]
priority_colors = [COLORS["success"], COLORS["warning"], COLORS["secondary"]]

# Öncelik bazında gecikme verisi
priority_data = [
    combined[combined["priority_ord"] == p]["delay_days"].values
    for p in priority_order
]

bp = ax.boxplot(
    priority_data,
    labels=priority_order,
    patch_artist=True,   # Kutular renkli olsun
    medianprops={"color": "white", "linewidth": 2}
)

for patch, color in zip(bp["boxes"], priority_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title("Öncelik Seviyesine Göre Gecikme Dağılımı")
ax.set_ylabel("Gecikme (gün)")
ax.set_xlabel("Teslimat Önceliği")

plt.tight_layout()
plt.savefig("reports/figures/03_seasonality_analysis.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ reports/figures/03_seasonality_analysis.png kaydedildi")


# ──────────────────────────────────────────────────────────
# ANALİZ 4: DEPO VERİMLİLİK ANALİZİ
# ──────────────────────────────────────────────────────────

print("\n📊 Analiz 4: Depo verimliliği analiz ediliyor...")

# Depo bazlı özet istatistikler
warehouse_summary = (
    warehouse
    .groupby(["warehouse_id", "city"])
    .agg(
        avg_efficiency    = ("efficiency_score",      "mean"),
        avg_capacity_used = ("capacity_used_pct",     "mean"),
        avg_error_rate    = ("error_rate_pct",        "mean"),
        avg_processing    = ("avg_processing_min",    "mean"),
        avg_daily_orders  = ("daily_orders_processed","mean"),
    )
    .reset_index()
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Depo Performans Analizi", fontsize=16, fontweight="bold")

# Sol üst: Depo verimliliği heatmap (aya göre)
# Heatmap → iki boyutlu ısı haritası, örüntüleri kolayca gösterir
warehouse_monthly = (
    warehouse
    .groupby(["warehouse_id", warehouse["date"].dt.month])
    ["efficiency_score"]
    .mean()
    .unstack()  # pivot gibi: depo=satır, ay=sütun
)
warehouse_monthly.columns = [month_names[m] for m in warehouse_monthly.columns]

sns.heatmap(
    warehouse_monthly,
    ax=axes[0, 0],
    cmap="RdYlGn",        # Kırmızı (düşük) → Yeşil (yüksek)
    annot=True,            # Hücrelere değer yaz
    fmt=".2f",             # 2 ondalık
    linewidths=0.5,
    cbar_kws={"label": "Verimlilik Skoru"}
)
axes[0, 0].set_title("Depo × Ay Verimlilik Isı Haritası")
axes[0, 0].set_xlabel("Ay")
axes[0, 0].set_ylabel("Depo ID")

# Sağ üst: Kapasite kullanımı vs Hata oranı
sc = axes[0, 1].scatter(
    warehouse_summary["avg_capacity_used"],
    warehouse_summary["avg_error_rate"],
    s=warehouse_summary["avg_daily_orders"] * 0.5,  # Nokta = işlem hacmi
    c=warehouse_summary["avg_efficiency"],
    cmap="RdYlGn",
    alpha=0.8,
    edgecolors="white",
    linewidth=0.5
)
for _, row in warehouse_summary.iterrows():
    axes[0, 1].annotate(
        row["city"],
        (row["avg_capacity_used"], row["avg_error_rate"]),
        fontsize=8, xytext=(3, 3), textcoords="offset points"
    )

plt.colorbar(sc, ax=axes[0, 1], label="Verimlilik Skoru")
axes[0, 1].set_title("Kapasite Kullanımı vs Hata Oranı")
axes[0, 1].set_xlabel("Ort. Kapasite Kullanımı (%)")
axes[0, 1].set_ylabel("Ort. Hata Oranı (%)")

# Sol alt: Aylık kapasite kullanım trendi
warehouse_cap_monthly = (
    warehouse
    .groupby(warehouse["date"].dt.month)
    ["capacity_used_pct"]
    .mean()
)
months_tr = [month_names[m] for m in warehouse_cap_monthly.index]
bar_colors_cap = [
    COLORS["secondary"] if v > 85 else
    COLORS["warning"]   if v > 75 else
    COLORS["success"]
    for v in warehouse_cap_monthly.values
]

axes[1, 0].bar(months_tr, warehouse_cap_monthly.values,
               color=bar_colors_cap, edgecolor="white", width=0.7)
axes[1, 0].axhline(80, color=COLORS["secondary"], linestyle="--",
                    linewidth=1.5, label="Kritik Kapasite (%80)")
axes[1, 0].set_ylim(0, 100)
axes[1, 0].set_title("Aylık Ortalama Kapasite Kullanımı")
axes[1, 0].set_ylabel("Kapasite Kullanımı (%)")
axes[1, 0].legend()

# Sağ alt: Ortalama işlem süresi karşılaştırması
wh_sorted = warehouse_summary.sort_values("avg_processing", ascending=True)
bar_colors_proc = [
    COLORS["secondary"] if v > 10 else
    COLORS["warning"]   if v > 8  else
    COLORS["success"]
    for v in wh_sorted["avg_processing"]
]

axes[1, 1].barh(wh_sorted["city"], wh_sorted["avg_processing"],
                color=bar_colors_proc, edgecolor="white")
for i, v in enumerate(wh_sorted["avg_processing"]):
    axes[1, 1].text(v + 0.1, i, f"{v:.1f} dk", va="center", fontsize=9)
axes[1, 1].set_title("Depo Başına Ort. Sipariş İşlem Süresi")
axes[1, 1].set_xlabel("İşlem Süresi (dakika/sipariş)")

plt.tight_layout()
plt.savefig("reports/figures/04_warehouse_performance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ reports/figures/04_warehouse_performance.png kaydedildi")


# ──────────────────────────────────────────────────────────
# ANALİZ 5: MALİYET ANALİZİ
# ──────────────────────────────────────────────────────────

print("\n📊 Analiz 5: Maliyet analizi yapılıyor...")

# Ürün kategorisi bazında maliyet ve gecikme
category_analysis = (
    combined
    .groupby("product_category")
    .agg(
        total_orders    = ("order_id",           "count"),
        avg_cost        = ("delivery_cost_tl",   "mean"),
        total_revenue   = ("order_value_tl",     "sum"),
        delay_rate      = ("is_delayed",         "mean"),
        avg_weight      = ("weight_kg_ord",          "mean"),
    )
    .reset_index()
)
category_analysis["delay_rate_pct"] = category_analysis["delay_rate"] * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Ürün Kategorisi ve Maliyet Analizi", fontsize=16, fontweight="bold")

# Maliyet bazında sıralama
cat_sorted = category_analysis.sort_values("avg_cost", ascending=True)

# Ortalama teslimat maliyeti
axes[0].barh(
    cat_sorted["product_category"],
    cat_sorted["avg_cost"],
    color=COLORS["primary"],
    edgecolor="white"
)
axes[0].set_title("Kategori Başına Ort. Teslimat Maliyeti (TL)")
axes[0].set_xlabel("Ortalama Maliyet (TL)")

# Sipariş başına gelir vs maliyet
axes[1].scatter(
    category_analysis["avg_cost"],
    category_analysis["total_revenue"] / category_analysis["total_orders"],
    s=category_analysis["total_orders"] / 5,
    c=category_analysis["delay_rate_pct"],
    cmap="RdYlGn_r",
    alpha=0.8,
    edgecolors="white"
)
for _, row in category_analysis.iterrows():
    axes[1].annotate(
        row["product_category"],
        (row["avg_cost"],
         row["total_revenue"] / row["total_orders"]),
        fontsize=8, xytext=(3, 3), textcoords="offset points"
    )
axes[1].set_title("Ort. Maliyet vs Ort. Sipariş Değeri")
axes[1].set_xlabel("Ort. Teslimat Maliyeti (TL)")
axes[1].set_ylabel("Ort. Sipariş Değeri (TL)")

# Gecikme oranı vs maliyet
cat_delay_sorted = category_analysis.sort_values("delay_rate_pct", ascending=True)
delay_colors = [
    COLORS["secondary"] if r > 25 else
    COLORS["warning"]   if r > 20 else
    COLORS["success"]
    for r in cat_delay_sorted["delay_rate_pct"]
]
axes[2].barh(
    cat_delay_sorted["product_category"],
    cat_delay_sorted["delay_rate_pct"],
    color=delay_colors,
    edgecolor="white"
)
for i, v in enumerate(cat_delay_sorted["delay_rate_pct"]):
    axes[2].text(v + 0.2, i, f"%{v:.1f}", va="center", fontsize=9)
axes[2].set_title("Kategori Bazında Gecikme Oranı (%)")
axes[2].set_xlabel("Gecikme Oranı (%)")

plt.tight_layout()
plt.savefig("reports/figures/05_cost_analysis.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ reports/figures/05_cost_analysis.png kaydedildi")


# ──────────────────────────────────────────────────────────
# ANALİZ 6: SÜRÜCÜ PERFORMANSI
# ──────────────────────────────────────────────────────────

print("\n📊 Analiz 6: Sürücü performans analizi...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Sürücü Performans Analizi", fontsize=16, fontweight="bold")

# Performans kategorisi dağılımı (pasta grafiği)
perf_counts = drivers["performance_category"].value_counts()
perf_colors = [
    COLORS["success"] if "Yıldız"    in c else
    COLORS["primary"] if "İyi"       in c else
    COLORS["warning"] if "Ortalama"  in c else
    COLORS["secondary"]
    for c in perf_counts.index
]

axes[0].pie(
    perf_counts.values,
    labels=perf_counts.index,
    colors=perf_colors,
    autopct="%1.1f%%",  # Her dilime yüzde yaz
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
axes[0].set_title("Sürücü Performans Kategorisi Dağılımı")

# Gecikme oranı dağılımı (histogram)
axes[1].hist(
    drivers["delay_rate"] * 100,
    bins=20,
    color=COLORS["primary"],
    edgecolor="white",
    alpha=0.8
)
axes[1].axvline(
    drivers["delay_rate"].mean() * 100,
    color=COLORS["secondary"],
    linestyle="--",
    linewidth=2,
    label=f"Ortalama: %{drivers['delay_rate'].mean()*100:.1f}"
)
axes[1].set_title("Sürücü Gecikme Oranı Dağılımı")
axes[1].set_xlabel("Gecikme Oranı (%)")
axes[1].set_ylabel("Sürücü Sayısı")
axes[1].legend()

# Deneyim vs Gecikme oranı
exp_groups = drivers.groupby("experience_years").agg(
    avg_delay = ("delay_rate", "mean"),
    avg_sat   = ("avg_satisfaction", "mean"),
).reset_index()

axes[2].scatter(
    drivers["experience_years"],
    drivers["delay_rate"] * 100,
    alpha=0.4,
    color=COLORS["primary"],
    s=30,
    label="Bireysel sürücüler"
)
# Trend çizgisi (polinom fit)
z = np.polyfit(drivers["experience_years"], drivers["delay_rate"] * 100, 1)
p = np.poly1d(z)
x_line = np.linspace(1, 20, 100)
axes[2].plot(x_line, p(x_line),
             color=COLORS["secondary"], linewidth=2.5, linestyle="--",
             label="Trend")
axes[2].set_title("Deneyim Yılı vs Gecikme Oranı")
axes[2].set_xlabel("Deneyim Yılı")
axes[2].set_ylabel("Gecikme Oranı (%)")
axes[2].legend()

plt.tight_layout()
plt.savefig("reports/figures/06_driver_performance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ reports/figures/06_driver_performance.png kaydedildi")


# ──────────────────────────────────────────────────────────
# ÖZET RAPOR - KPI TABLOSU
# ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  EDA ÖZET KPI TABLOSU")
print("="*60)
print(f"  📦 Toplam sipariş:              {len(combined):>10,}")
print(f"  ⚠️  Gecikmeli teslimat sayısı:  {combined['is_delayed'].sum():>10,}")
print(f"  📊 Genel gecikme oranı:         {combined['is_delayed'].mean():>9.1%}")
print(f"  ⏰ Ort. gecikme süresi:         {combined[combined['is_delayed']]['delay_days'].mean():>8.1f} gün")
print(f"  ⭐ Ort. müşteri memnuniyeti:    {combined['customer_satisfaction'].mean():>8.2f}/5")
print(f"  💰 Ort. teslimat maliyeti:      {combined['delivery_cost_tl'].mean():>8.0f} TL")
print(f"  🏭 En verimli depo:             {warehouse_summary.loc[warehouse_summary['avg_efficiency'].idxmax(),'city']:>12}")
print(f"  🏭 En az verimli depo:          {warehouse_summary.loc[warehouse_summary['avg_efficiency'].idxmin(),'city']:>12}")
print(f"  👤 Yıldız Sürücü sayısı:        {(drivers['performance_category']=='Yıldız Sürücü').sum():>10,}")
print("="*60)
print("\n✅ Tüm EDA grafikleri reports/figures/ klasörüne kaydedildi.")
