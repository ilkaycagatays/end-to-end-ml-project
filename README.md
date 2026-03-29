# 🚚 Lojistik Operasyon Optimizasyonu — Veri Analizi Portföy Projesi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> **Junior Data Analyst Portföy Projesi** — Türkiye genelinde 15.000 lojistik teslimatının gecikme analizi, maliyet optimizasyonu ve iş yükü tahmini. Gerçek dünya ERP/WMS sistemlerini simüle eden sentetik veri ile uçtan uca veri analizi pipeline'ı.

---

## 📋 İçindekiler

- [Proje Özeti](#-proje-özeti)
- [İş Soruları ve Bulgular](#-i̇ş-soruları-ve-bulgular)
- [Veri Seti](#-veri-seti)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Nasıl Çalıştırılır](#-nasıl-çalıştırılır)
- [Modeller](#-makine-öğrenmesi-modelleri)
- [Temel Bulgular](#-temel-bulgular)
- [İletişim](#-i̇letişim)

---

## 🎯 Proje Özeti

Bu proje, bir lojistik şirketinin operasyonel verilerini analiz ederek şu soruları yanıtlar:

| Soru | Yaklaşım | Çıktı |
|------|----------|-------|
| Hangi bölgelerde gecikme daha fazla? | EDA + Görselleştirme | Şehir bazlı gecikme haritası |
| Gecikmenin ana sebepleri neler? | Pareto Analizi | %80 sorunu oluşturan top 3 sebep |
| İş yükü tahmin edilebilir mi? | Zaman Serisi | 30 günlük talep öngörüsü |
| Personel planlaması nasıl iyileşir? | Operasyonel Formül | Günlük sürücü ihtiyaç tavsiyesi |
| Bu teslimat gecikmeli olacak mı? | Random Forest | Erken Uyarı Sınıflandırıcısı |

**Kapsam:**
- 📅 2023–2024 dönemi (2 yıl)
- 🏙️ 17 Türkiye şehri
- 📦 15.000 sipariş kaydı
- 🚚 120 sürücü profili
- 🏭 8 depo, 730 günlük performans verisi

---

## ❓ İş Soruları ve Bulgular

### 1. Hangi bölgelerde gecikme daha fazla?

Doğu ve iç Anadolu şehirleri batı şehirlerine kıyasla 2–3 kat daha yüksek gecikme oranı gösteriyor. Bunun başlıca nedenleri: altyapı yoğunluğu, hava koşulları ve uzun mesafe.

```
Diyarbakır  → %41.7 gecikme  ⚠️ Kritik
Erzurum     → %37.2 gecikme  ⚠️ Kritik  
Trabzon     → %34.1 gecikme  ⚠️ Yüksek
Bursa       → %14.3 gecikme  ✅ Düşük
Kocaeli     → %15.1 gecikme  ✅ Düşük
```

### 2. Gecikmenin ana sebepleri neler?

Pareto analizi: **3 sebep toplam gecikmelerin %40'ını oluşturuyor.**

| Sebep | Vaka Sayısı | Pay | Ort. Süre |
|-------|-------------|-----|-----------|
| Hava Koşulları | 687 | %21.0 | 2.0 gün |
| Yol Çalışması | 310 | %9.5 | 2.0 gün |
| Depo Gecikme | 303 | %9.3 | 2.0 gün |

→ **Aksiyon:** Kış aylarında (Aralık–Şubat) hava koşulları monitörü + alternatif rota önerisi sistemi kurulması önerilir.

### 3. İş yükü tahmini yapılabilir mi?

Evet. Zaman serisi modeli:
- **Kasım–Aralık:** +%40 sipariş artışı (Black Friday etkisi)
- **Cuma günleri:** haftalık pik (+%15)
- **Pazar günleri:** minimum hacim (-%50)
- **30 günlük MAE:** ±%15 güven aralığı ile tahmin

### 4. Personel planlaması nasıl iyileşir?

Formül: `Gerekli Sürücü = ⌈(Tahmini Sipariş / 12) × 1.15⌉`

Pik dönemde (Kasım–Aralık) normal döneme göre **%30 ek personel** ihtiyacı öngörülüyor.

---

## 📁 Veri Seti

Proje gerçek ERP/WMS sistemlerini simüle eden sentetik veri kullanır.

### Tablolar

| Dosya | Satır | Sütun | Açıklama |
|-------|-------|-------|----------|
| `orders.csv` | 15.000 | 12 | Sipariş kayıtları |
| `deliveries.csv` | 15.000 | 18 | Teslimat detayları ve gecikme bilgisi |
| `warehouse_performance.csv` | 5.848 | 15 | Günlük depo metrikleri (8 depo × 730 gün) |
| `driver_performance.csv` | 120 | 14 | Sürücü performans profilleri |

### Temel Değişkenler

```
orders.csv
├── order_id          # Benzersiz sipariş ID
├── order_date        # Sipariş tarihi
├── destination_city  # Teslimat şehri (17 şehir)
├── product_category  # Ürün kategorisi (10 kategori)
├── weight_kg         # Ağırlık
├── order_value_tl    # Sipariş tutarı (TL)
└── priority          # Express / Standard / Economy

deliveries.csv
├── is_delayed           # HEDEF değişken (0/1)
├── delay_days           # Gecikme süresi (gün)
├── delay_reason         # Gecikme sebebi (10 kategori)
├── customer_satisfaction # Müşteri memnuniyeti (1–5)
└── delivery_cost_tl     # Teslimat maliyeti
```

### Gerçekçilik Özellikleri

- **Mevsimsellik:** Kasım–Aralık sipariş hacmi +%40
- **Nüfus ağırlığı:** İstanbul siparişlerin %28'ini alıyor
- **Coğrafi gecikme:** Doğu şehirleri yapısal olarak daha yüksek gecikme
- **Öncelik etkisi:** Economy teslimatlar 1.5x daha fazla gecikiyor

---

## 🗂️ Proje Yapısı

```
logistics-optimization/
│
├── data/
│   ├── raw/                    # Ham CSV dosyaları
│   │   ├── orders.csv
│   │   ├── deliveries.csv
│   │   ├── warehouse_performance.csv
│   │   └── driver_performance.csv
│   └── processed/              # İşlenmiş ve zenginleştirilmiş veri
│       ├── combined_features.csv
│       └── demand_forecast.csv
│
├── src/
│   ├── generate_data.py        # Sentetik veri üretici
│   ├── eda_analysis.py         # Keşifsel Veri Analizi (6 analiz, 6 grafik)
│   └── train_models.py         # 3 ML modeli eğitimi
│
├── models/
│   ├── delay_classifier.pkl    # Random Forest gecikme sınıflandırıcı
│   └── delay_regressor.pkl     # Gradient Boosting gecikme süresi tahmini
│
├── reports/
│   └── figures/                # Tüm grafik çıktıları (PNG)
│       ├── 01_city_delay_analysis.png
│       ├── 02_delay_reasons_pareto.png
│       ├── 03_seasonality_analysis.png
│       ├── 04_warehouse_performance.png
│       ├── 05_cost_analysis.png
│       ├── 06_driver_performance.png
│       ├── 07_delay_classifier_results.png
│       ├── 08_delay_regressor_results.png
│       └── 09_demand_forecast.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.10+
- pip

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/kullanici-adin/logistics-optimization.git
cd logistics-optimization

# 2. Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# ya da
venv\Scripts\activate           # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 🚀 Nasıl Çalıştırılır

Dosyalar sırayla çalıştırılmalıdır (her adım bir sonrakine bağımlı):

```bash
# ADIM 1 — Sentetik veri üret (data/raw/ klasörüne kaydeder)
python src/generate_data.py

# ADIM 2 — Keşifsel veri analizi (reports/figures/ klasörüne grafik kaydeder)
python src/eda_analysis.py

# ADIM 3 — ML modelleri eğit (models/ klasörüne .pkl kaydeder)
python src/train_models.py
```

**Beklenen çalışma süresi:**
- `generate_data.py` → ~30 saniye
- `eda_analysis.py`  → ~45 saniye
- `train_models.py`  → ~2–4 dakika (CPU hızına göre değişir)

---

## 🤖 Makine Öğrenmesi Modelleri

### Model 1 — Gecikme Sınıflandırıcısı

| Parametre | Değer |
|-----------|-------|
| Algoritma | Random Forest Classifier |
| Hedef | `is_delayed` (0/1) |
| CV ROC-AUC | 0.650 ± 0.009 |
| Test ROC-AUC | 0.665 |
| Test Accuracy | %68.6 |

**Kullanım senaryosu:** Sipariş alındığı anda %X gecikme olasılığı üretir → operasyon ekibi proaktif müdahale edebilir (alternatif rota, müşteri bilgilendirme).

**En önemli özellikler:**
1. Tarihsel şehir gecikme oranı
2. Depo kapasitesi ve verimliliği
3. Mevsim (Kasım–Aralık peak)
4. Teslimat önceliği

### Model 2 — Gecikme Süresi Tahmini

| Parametre | Değer |
|-----------|-------|
| Algoritma | Gradient Boosting Regressor |
| Hedef | `delay_days` |
| MAE | 1.29 gün |
| RMSE | 1.79 gün |

**Kullanım senaryosu:** Gecikme tespit edildiğinde müşteriye "teslimatınız 2 gün gecikmeli" gibi kesin bildirim yapılır.

### Model 3 — Talep Tahmini

| Parametre | Değer |
|-----------|-------|
| Yaklaşım | Hareketli Ortalama + EWMA + Mevsimsel Faktörler |
| Ufuk | 30 gün |
| Güven Aralığı | ±%15 |

**Kullanım senaryosu:** Önümüzdeki 30 gün için günlük sipariş hacmi ve gerekli sürücü sayısı tahmini → İnsan kaynakları planlaması.

---

## 📊 Temel Bulgular

```
┌─────────────────────────────────────────────────────────┐
│                   ÖZET KPI PANELİ                       │
├─────────────────────────────────────────────────────────┤
│  Toplam Sipariş            15.000                       │
│  Genel Gecikme Oranı       %21.8                        │
│  Ort. Gecikme Süresi       2.0 gün                      │
│  Ort. Müşteri Memnuniyeti  4.21 / 5.0                   │
│  En Sorunlu Şehir          Diyarbakır (%41.7)           │
│  En İyi Şehir              Bursa (%14.3)                │
│  En Yoğun Ay               Aralık (%40 sipariş artışı) │
│  En Verimli Depo           İstanbul (WH-01)             │
└─────────────────────────────────────────────────────────┘
```

### Operasyonel Öneriler

1. **Kış Hazırlığı:** Kasım başında Doğu Anadolu depolarına +%20 stok aktarımı
2. **Dinamik Sürücü Planlaması:** Cuma günleri için +%15 sürücü kapasitesi
3. **Erken Uyarı Sistemi:** Model 1 ile sipariş anında gecikme skoru üret
4. **Depo Optimizasyonu:** WH-07 (Konya) verimliliği için süreç iyileştirmesi

---

## 🛠️ Kullanılan Teknolojiler

| Kategori | Araç | Kullanım |
|----------|------|----------|
| Veri İşleme | `pandas`, `numpy` | ETL, feature engineering |
| Görselleştirme | `matplotlib`, `seaborn` | EDA grafikleri |
| ML Modelleme | `scikit-learn` | Sınıflandırma, regresyon |
| Model Kaydetme | `joblib` | Model serileştirme |

---

## 📬 İletişim

Proje hakkında sorularınız veya geri bildiriminiz için:

- **LinkedIn:** [linkedin.com/in/kullanici-adin](https://linkedin.com/in/kullanici-adin)
- **GitHub:** [github.com/kullanici-adin](https://github.com/kullanici-adin)

---

*Bu proje portföy amaçlıdır. Veriler tamamen sentetik olup gerçek bir şirkete ait değildir.*
