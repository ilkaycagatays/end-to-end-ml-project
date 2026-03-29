"""
=============================================================
LOJISTIK OPTİMİZASYON PROJESİ - VERİ ÜRETME MODÜLÜ
=============================================================
Bu dosya gerçek dünya lojistik verisini simüle eden
sentetik veri setleri üretir. Gerçek şirketlerde bu veriler
ERP/WMS sistemlerinden gelir (SAP, Oracle, vb.)

Üretilen veri setleri:
  1. orders.csv         → Sipariş kayıtları
  2. deliveries.csv     → Teslimat detayları
  3. warehouses.csv     → Depo performans metrikleri
  4. routes.csv         → Rota ve araç verileri
  5. drivers.csv        → Sürücü performans verileri
=============================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ──────────────────────────────────────────────────────────
# GENEL AYARLAR
# ──────────────────────────────────────────────────────────

# Tekrarlanabilirlik için seed sabitleme
# Aynı seed → her çalıştırmada aynı "rastgele" veri
np.random.seed(42)
random.seed(42)

# Simülasyon parametreleri
N_ORDERS = 15_000          # Toplam sipariş sayısı (15 bin)
N_DRIVERS = 120            # Sürücü sayısı
N_WAREHOUSES = 8           # Depo sayısı
START_DATE = datetime(2023, 1, 1)   # Veri başlangıç tarihi
END_DATE   = datetime(2024, 12, 31) # Veri bitiş tarihi

# Türkiye'nin büyük şehirleri ve koordinatları
CITIES = {
    "İstanbul":  {"lat": 41.0082, "lon": 28.9784, "population_weight": 0.28},
    "Ankara":    {"lat": 39.9334, "lon": 32.8597, "population_weight": 0.12},
    "İzmir":     {"lat": 38.4192, "lon": 27.1287, "population_weight": 0.09},
    "Bursa":     {"lat": 40.1885, "lon": 29.0610, "population_weight": 0.07},
    "Antalya":   {"lat": 36.8969, "lon": 30.7133, "population_weight": 0.06},
    "Adana":     {"lat": 37.0000, "lon": 35.3213, "population_weight": 0.05},
    "Konya":     {"lat": 37.8746, "lon": 32.4932, "population_weight": 0.04},
    "Gaziantep": {"lat": 37.0662, "lon": 37.3833, "population_weight": 0.04},
    "Kayseri":   {"lat": 38.7312, "lon": 35.4787, "population_weight": 0.03},
    "Trabzon":   {"lat": 41.0015, "lon": 39.7178, "population_weight": 0.03},
    "Samsun":    {"lat": 41.2867, "lon": 36.3300, "population_weight": 0.03},
    "Erzurum":   {"lat": 39.9086, "lon": 41.2769, "population_weight": 0.02},
    "Diyarbakır":{"lat": 37.9144, "lon": 40.2306, "population_weight": 0.02},
    "Eskişehir": {"lat": 39.7767, "lon": 30.5206, "population_weight": 0.02},
    "Mersin":    {"lat": 36.8121, "lon": 34.6415, "population_weight": 0.03},
    "Kocaeli":   {"lat": 40.8533, "lon": 29.8815, "population_weight": 0.05},
    "Manisa":    {"lat": 38.6191, "lon": 27.4289, "population_weight": 0.02},
}

# Ürün kategorileri ve ortalama ağırlıkları (kg)
PRODUCT_CATEGORIES = {
    "Elektronik":       {"avg_weight": 2.5,  "avg_value": 3500, "fragile": True},
    "Tekstil":          {"avg_weight": 1.2,  "avg_value": 450,  "fragile": False},
    "Gıda":             {"avg_weight": 8.0,  "avg_value": 250,  "fragile": False},
    "Mobilya":          {"avg_weight": 45.0, "avg_value": 2800, "fragile": True},
    "Kozmetik":         {"avg_weight": 0.8,  "avg_value": 320,  "fragile": False},
    "Kitap/Kırtasiye":  {"avg_weight": 1.5,  "avg_value": 120,  "fragile": False},
    "Spor Malzemesi":   {"avg_weight": 5.0,  "avg_value": 800,  "fragile": False},
    "Otomotiv Parça":   {"avg_weight": 12.0, "avg_value": 1500, "fragile": False},
    "Oyuncak":          {"avg_weight": 1.8,  "avg_value": 280,  "fragile": False},
    "Ev & Yaşam":       {"avg_weight": 6.5,  "avg_value": 650,  "fragile": True},
}

# Gecikme sebepleri ve olasılıkları
DELAY_REASONS = [
    "Trafik Yoğunluğu",
    "Hava Koşulları",
    "Araç Arızası",
    "Adres Bulunamadı",
    "Alıcı Mevcut Değil",
    "Depo Gecikme",
    "Yol Çalışması",
    "Gümrük Beklemesi",
    "Yüksek Sipariş Hacmi",
    "Sürücü Değişimi",
]

# ──────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> datetime:
    """İki tarih arasında rastgele bir tarih üretir."""
    delta = end - start                        # İki tarih arası fark (gün olarak)
    random_days = random.randint(0, delta.days)  # 0 ile delta gün arası rastgele sayı
    return start + timedelta(days=random_days)


def weighted_choice(choices: dict) -> str:
    """
    Ağırlıklı rastgele seçim yapar.
    choices: {"seçenek": ağırlık, ...} formatında dict
    Nüfus ağırlığına göre şehir seçmek için kullanılır.
    """
    items = list(choices.keys())
    weights = [choices[k] for k in items]
    return random.choices(items, weights=weights, k=1)[0]


def add_seasonality(date: datetime, base_value: float) -> float:
    """
    Mevsimsellik ve haftalık döngü ekler.
    Gerçek hayatta:
      - Kasım/Aralık → e-ticaret siparişleri patlar (Black Friday, Yılbaşı)
      - Hafta sonları → teslimat azalır
      - Sabah saatleri → depo yoğunluğu artar
    """
    # Ay bazlı mevsimsel faktör
    month_factors = {
        1: 0.75,  # Ocak: yılbaşı sonrası düşüş
        2: 0.70,  # Şubat: yıl içi en düşük
        3: 0.80,  # Mart: canlanma başlıyor
        4: 0.85,  # Nisan: bahar artışı
        5: 0.90,  # Mayıs: normal dönem
        6: 0.85,  # Haziran: yaz öncesi
        7: 0.80,  # Temmuz: tatil sezonu
        8: 0.78,  # Ağustos: yaz düşüşü
        9: 0.88,  # Eylül: okul başlangıcı
        10: 0.95, # Ekim: yoğunlaşma
        11: 1.30, # Kasım: Black Friday!
        12: 1.40, # Aralık: Yılbaşı siparişleri zirve!
    }

    # Haftanın günü faktörü (0=Pazartesi, 6=Pazar)
    weekday_factors = {
        0: 1.10,  # Pazartesi: haftanın başı yoğun
        1: 1.05,  # Salı
        2: 1.00,  # Çarşamba: normal
        3: 1.05,  # Perşembe
        4: 1.15,  # Cuma: hafta sonu öncesi sipariş patlaması
        5: 0.70,  # Cumartesi: azalma
        6: 0.50,  # Pazar: minimum
    }

    month_factor = month_factors.get(date.month, 1.0)
    weekday_factor = weekday_factors.get(date.weekday(), 1.0)

    # Küçük rastgele gürültü ekliyoruz (±10%)
    noise = np.random.normal(1.0, 0.10)

    return base_value * month_factor * weekday_factor * noise


# ──────────────────────────────────────────────────────────
# 1. SİPARİŞ VERİSİ OLUŞTURMA
# ──────────────────────────────────────────────────────────

def generate_orders() -> pd.DataFrame:
    """
    15.000 sipariş kaydı üretir.
    Her satır = bir müşteri siparişi

    Sütunlar:
      order_id        → Benzersiz sipariş kimliği
      order_date      → Sipariş tarihi
      customer_id     → Müşteri kimliği
      origin_city     → Gönderen şehir (depo şehri)
      destination_city→ Teslimat şehri
      product_category→ Ürün kategorisi
      quantity        → Adet
      weight_kg       → Toplam ağırlık
      order_value_tl  → Sipariş tutarı (TL)
      priority        → Öncelik seviyesi (Express/Standard/Economy)
      is_fragile      → Kırılgan ürün mü?
      warehouse_id    → Hangi depodan çıkıyor
    """
    print("📦 Sipariş verileri üretiliyor...")

    orders = []
    city_names = list(CITIES.keys())
    city_weights = [CITIES[c]["population_weight"] for c in city_names]

    # Ana depo şehirleri (gerçekte büyük lojistik merkezler)
    warehouse_cities = ["İstanbul", "Ankara", "İzmir", "Bursa",
                        "Antalya", "Adana", "Konya", "Kocaeli"]

    for i in range(N_ORDERS):
        # Rastgele sipariş tarihi
        order_date = random_date(START_DATE, END_DATE)

        # Hedef şehri nüfusa göre ağırlıklı seç
        # İstanbul en çok sipariş alıyor (nüfusa orantılı)
        destination = random.choices(city_names, weights=city_weights, k=1)[0]

        # Ürün kategorisi (eşit dağılım)
        category = random.choice(list(PRODUCT_CATEGORIES.keys()))
        cat_info = PRODUCT_CATEGORIES[category]

        # Adet: 1-10 arası, küçük değerlere daha yatkın (üstel dağılım)
        quantity = max(1, int(np.random.exponential(scale=2)))
        quantity = min(quantity, 50)  # max 50 adet

        # Ağırlık hesaplama (±20% varyasyon)
        base_weight = cat_info["avg_weight"] * quantity
        weight = max(0.1, np.random.normal(base_weight, base_weight * 0.2))

        # Sipariş değeri (ağırlık ve kategori bazlı)
        base_value = cat_info["avg_value"] * quantity
        order_value = max(10, np.random.normal(base_value, base_value * 0.25))

        # Öncelik (pahalı siparişler daha çok express tercih eder)
        if order_value > 5000:
            priority = random.choices(["Express", "Standard", "Economy"],
                                      weights=[0.5, 0.35, 0.15])[0]
        elif order_value > 1000:
            priority = random.choices(["Express", "Standard", "Economy"],
                                      weights=[0.25, 0.50, 0.25])[0]
        else:
            priority = random.choices(["Express", "Standard", "Economy"],
                                      weights=[0.10, 0.40, 0.50])[0]

        # Depo ataması
        warehouse_id = f"WH-{random.randint(1, N_WAREHOUSES):02d}"
        origin_city = warehouse_cities[(int(warehouse_id.split('-')[1]) - 1) % len(warehouse_cities)]

        orders.append({
            "order_id":          f"ORD-{i+1:06d}",
            "order_date":        order_date.strftime("%Y-%m-%d"),
            "customer_id":       f"CUS-{random.randint(1, 5000):05d}",
            "origin_city":       origin_city,
            "destination_city":  destination,
            "product_category":  category,
            "quantity":          quantity,
            "weight_kg":         round(weight, 2),
            "order_value_tl":    round(order_value, 2),
            "priority":          priority,
            "is_fragile":        cat_info["fragile"],
            "warehouse_id":      warehouse_id,
        })

    df = pd.DataFrame(orders)
    print(f"   ✅ {len(df):,} sipariş kaydı oluşturuldu.")
    return df


# ──────────────────────────────────────────────────────────
# 2. TESLİMAT VERİSİ OLUŞTURMA
# ──────────────────────────────────────────────────────────

def generate_deliveries(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her sipariş için teslimat kaydı üretir.
    Gecikme olasılıkları gerçekçi şekilde modellendi.

    Gecikmeyi etkileyen faktörler:
      - Şehir (doğu şehirleri → daha fazla gecikme)
      - Mevsim (kış → hava koşulları, Kasım-Aralık → yoğunluk)
      - Öncelik (Economy → daha fazla gecikme riski)
      - Mesafe (uzak şehirler → daha riskli)
    """
    print("🚚 Teslimat verileri üretiliyor...")

    # Şehirlere göre taban gecikme olasılığı
    # Doğu şehirleri altyapı nedeniyle daha fazla gecikme yaşar
    city_delay_prob = {
        "İstanbul":   0.18,  "Ankara":     0.14,  "İzmir":      0.13,
        "Bursa":      0.12,  "Antalya":    0.15,  "Adana":      0.18,
        "Konya":      0.20,  "Gaziantep":  0.22,  "Kayseri":    0.21,
        "Trabzon":    0.28,  "Samsun":     0.25,  "Erzurum":    0.35,
        "Diyarbakır": 0.32,  "Eskişehir":  0.15,  "Mersin":     0.17,
        "Kocaeli":    0.12,  "Manisa":     0.14,
    }

    deliveries = []

    for _, order in orders_df.iterrows():
        order_date = datetime.strptime(order["order_date"], "%Y-%m-%d")

        # ── Beklenen Teslimat Süresi ──
        # Önceliğe göre taahhüt edilen gün sayısı
        priority_days = {
            "Express":  1,  # 1 iş günü
            "Standard": 3,  # 3 iş günü
            "Economy":  7,  # 7 iş günü
        }
        promised_days = priority_days[order["priority"]]

        # ── Gecikme Olasılığı Hesaplama ──
        base_delay_prob = city_delay_prob.get(order["destination_city"], 0.20)

        # Mevsimsel etki (kış ayları → daha fazla gecikme)
        if order_date.month in [12, 1, 2]:  # Kış
            seasonal_multiplier = 1.4
        elif order_date.month in [11]:       # Kasım (Black Friday yoğunluğu)
            seasonal_multiplier = 1.6
        elif order_date.month in [7, 8]:     # Yaz tatili
            seasonal_multiplier = 0.85
        else:
            seasonal_multiplier = 1.0

        # Öncelik etkisi (Economy daha çok gecikir)
        priority_delay_mult = {"Express": 0.6, "Standard": 1.0, "Economy": 1.5}
        priority_mult = priority_delay_mult[order["priority"]]

        # Final gecikme olasılığı
        delay_prob = min(0.7, base_delay_prob * seasonal_multiplier * priority_mult)

        # ── Gecikme Var mı? ──
        is_delayed = np.random.random() < delay_prob

        if is_delayed:
            # Gecikme süresi: 1 ile 7 gün arası (üstel dağılım)
            delay_days = max(1, int(np.random.exponential(scale=2)))
            delay_days = min(delay_days, 10)  # max 10 gün gecikme

            delay_reason = random.choice(DELAY_REASONS)

            # Kış aylarında hava koşulları daha olası sebep
            if order_date.month in [12, 1, 2]:
                if random.random() < 0.4:
                    delay_reason = "Hava Koşulları"
        else:
            delay_days = 0
            delay_reason = None

        # ── Teslimat Tarihleri ──
        # Sipariş → Kargo Çıkışı → Teslimat
        pickup_days = random.randint(0, 1)          # 0-1 gün içinde kargoya verilir
        pickup_date = order_date + timedelta(days=pickup_days)

        # Taahhüt edilen teslimat tarihi
        promised_date = pickup_date + timedelta(days=promised_days)

        # Gerçek teslimat tarihi
        actual_delivery_date = promised_date + timedelta(days=delay_days)

        # ── Teslimat Ücreti Hesaplama ──
        # Ağırlık + mesafe + öncelik faktörü
        base_cost = order["weight_kg"] * 2.5  # kg başına 2.5 TL
        priority_cost_mult = {"Express": 2.5, "Standard": 1.0, "Economy": 0.7}
        distance_factor = 1.0

        # İstanbul'dan uzak şehirler daha pahalı
        if order["destination_city"] in ["Erzurum", "Diyarbakır", "Trabzon"]:
            distance_factor = 1.8
        elif order["destination_city"] in ["Gaziantep", "Adana", "Samsun"]:
            distance_factor = 1.4
        elif order["destination_city"] in ["İstanbul", "Kocaeli", "Bursa"]:
            distance_factor = 0.9

        delivery_cost = round(
            base_cost * priority_cost_mult[order["priority"]] * distance_factor, 2
        )

        # ── Müşteri Memnuniyeti Skoru ──
        # Gecikme yoksa yüksek, gecikme arttıkça düşüyor
        if not is_delayed:
            satisfaction = np.random.normal(4.5, 0.4)
        elif delay_days <= 2:
            satisfaction = np.random.normal(3.5, 0.5)
        elif delay_days <= 5:
            satisfaction = np.random.normal(2.5, 0.6)
        else:
            satisfaction = np.random.normal(1.8, 0.5)

        satisfaction = round(max(1.0, min(5.0, satisfaction)), 1)  # 1-5 arası sınırla

        # ── Teslimat Durumu ──
        # Tarihe göre otomatik durum ataması
        today = datetime(2025, 1, 1)  # Simülasyon bugün tarihi
        if actual_delivery_date <= today:
            if is_delayed:
                status = "Gecikmeli Teslim"
            else:
                status = "Zamanında Teslim"
        else:
            status = "Yolda"

        # Küçük bir kesim iptal edildi
        if random.random() < 0.02:  # %2 iptal oranı
            status = "İptal"

        deliveries.append({
            "delivery_id":           f"DEL-{len(deliveries)+1:06d}",
            "order_id":              order["order_id"],
            "warehouse_id":          order["warehouse_id"],
            "driver_id":             f"DRV-{random.randint(1, N_DRIVERS):03d}",
            "origin_city":           order["origin_city"],
            "destination_city":      order["destination_city"],
            "pickup_date":           pickup_date.strftime("%Y-%m-%d"),
            "promised_delivery_date":promised_date.strftime("%Y-%m-%d"),
            "actual_delivery_date":  actual_delivery_date.strftime("%Y-%m-%d"),
            "promised_days":         promised_days,
            "actual_days":           promised_days + delay_days,
            "delay_days":            delay_days,
            "is_delayed":            is_delayed,
            "delay_reason":          delay_reason,
            "delivery_cost_tl":      delivery_cost,
            "delivery_status":       status,
            "customer_satisfaction": satisfaction,
            "priority":              order["priority"],
            "weight_kg":             order["weight_kg"],
        })

    df = pd.DataFrame(deliveries)
    print(f"   ✅ {len(df):,} teslimat kaydı oluşturuldu.")
    print(f"   📊 Gecikme oranı: {df['is_delayed'].mean():.1%}")
    return df


# ──────────────────────────────────────────────────────────
# 3. DEPO PERFORMANS VERİSİ
# ──────────────────────────────────────────────────────────

def generate_warehouse_performance() -> pd.DataFrame:
    """
    Günlük depo performans metrikleri üretir.
    Gerçekte bu veriler WMS (Warehouse Management System) sistemlerinden gelir.

    Metrikler:
      - Kapasite kullanımı
      - İşlenen sipariş sayısı
      - Hata oranı (yanlış ürün, eksik adet vb.)
      - Personel sayısı
      - Ortalama işlem süresi
    """
    print("🏭 Depo performans verileri üretiliyor...")

    # Depo bilgileri
    warehouses = {
        "WH-01": {"city": "İstanbul",  "capacity": 50000, "base_efficiency": 0.87},
        "WH-02": {"city": "Ankara",    "capacity": 30000, "base_efficiency": 0.82},
        "WH-03": {"city": "İzmir",     "capacity": 25000, "base_efficiency": 0.85},
        "WH-04": {"city": "Bursa",     "capacity": 20000, "base_efficiency": 0.83},
        "WH-05": {"city": "Antalya",   "capacity": 18000, "base_efficiency": 0.79},
        "WH-06": {"city": "Adana",     "capacity": 15000, "base_efficiency": 0.76},
        "WH-07": {"city": "Konya",     "capacity": 12000, "base_efficiency": 0.74},
        "WH-08": {"city": "Kocaeli",   "capacity": 22000, "base_efficiency": 0.86},
    }

    records = []
    current_date = START_DATE

    while current_date <= END_DATE:
        for wh_id, wh_info in warehouses.items():

            # Mevsimsel sipariş hacmi
            base_orders = wh_info["capacity"] * 0.002  # Günlük taban sipariş
            daily_orders = int(add_seasonality(current_date, base_orders))
            daily_orders = max(10, daily_orders)

            # Kapasite kullanımı (Kasım-Aralık'ta taşıyor)
            cap_multiplier = 1.3 if current_date.month in [11, 12] else 1.0
            capacity_used = min(0.98,
                np.random.normal(0.65 * cap_multiplier, 0.08)
            )

            # Verimlilik (yorgunluk ve yoğunluk etkisi)
            # Kapasite dolunca verimlilik düşer
            efficiency_penalty = max(0, capacity_used - 0.85) * 0.5
            daily_efficiency = max(0.5,
                wh_info["base_efficiency"] - efficiency_penalty + np.random.normal(0, 0.03)
            )

            # Hata oranı (düşük verimlilik → yüksek hata)
            error_rate = max(0.001,
                (1 - daily_efficiency) * 0.08 + np.random.normal(0, 0.005)
            )

            # Personel sayısı (sipariş hacmine göre)
            base_staff = int(wh_info["capacity"] / 2000)
            # Yoğun dönemde ek personel alınır
            if current_date.month in [11, 12]:
                staff_count = int(base_staff * 1.3 + np.random.normal(0, 1))
            else:
                staff_count = int(base_staff + np.random.normal(0, 1))
            staff_count = max(5, staff_count)

            # Ortalama işlem süresi (dakika/sipariş)
            # Az personel veya yüksek hata → daha uzun süre
            base_processing_time = 8.0  # dakika
            processing_time = base_processing_time / daily_efficiency + np.random.normal(0, 0.5)
            processing_time = max(3.0, processing_time)

            # Araç sayısı (o gün kaç araç depoya geldi/gitti)
            vehicles_in = int(daily_orders / 20) + random.randint(0, 5)
            vehicles_out = int(daily_orders / 15) + random.randint(0, 5)

            records.append({
                "date":                  current_date.strftime("%Y-%m-%d"),
                "warehouse_id":          wh_id,
                "city":                  wh_info["city"],
                "capacity_limit":        wh_info["capacity"],
                "capacity_used_pct":     round(capacity_used * 100, 2),
                "daily_orders_processed":daily_orders,
                "efficiency_score":      round(daily_efficiency, 4),
                "error_rate_pct":        round(error_rate * 100, 4),
                "staff_count":           staff_count,
                "avg_processing_min":    round(processing_time, 2),
                "vehicles_in":           vehicles_in,
                "vehicles_out":          vehicles_out,
                "month":                 current_date.month,
                "weekday":               current_date.weekday(),
                "is_weekend":            current_date.weekday() >= 5,
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    print(f"   ✅ {len(df):,} depo kayıt satırı oluşturuldu.")
    return df


# ──────────────────────────────────────────────────────────
# 4. SÜRÜCÜ PERFORMANS VERİSİ
# ──────────────────────────────────────────────────────────

def generate_driver_performance(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her sürücünün aylık performans özetini çıkarır.
    Gerçekte bu veri sürücü takip sistemlerinden (GPS + mobil app) gelir.

    Metrikler:
      - Ortalama gecikme oranı
      - Teslimat sayısı
      - Müşteri memnuniyet skoru
      - Yakıt tüketimi
      - Sürüş skoru (fren, hız, güzergah)
    """
    print("👤 Sürücü performans verileri üretiliyor...")

    # Sürücü bazında teslimat istatistikleri hesapla
    driver_stats = (
        deliveries_df
        .groupby("driver_id")
        .agg(
            total_deliveries   = ("delivery_id", "count"),
            delayed_deliveries = ("is_delayed", "sum"),
            avg_satisfaction   = ("customer_satisfaction", "mean"),
            avg_delay_days     = ("delay_days", "mean"),
            total_cost         = ("delivery_cost_tl", "sum"),
        )
        .reset_index()
    )

    # Gecikme oranı sütunu ekle
    driver_stats["delay_rate"] = (
        driver_stats["delayed_deliveries"] / driver_stats["total_deliveries"]
    )

    # Ek sürücü metrikleri (GPS/telematik simülasyonu)
    driver_stats["avg_daily_km"] = np.random.normal(180, 40, len(driver_stats)).clip(50, 400)
    driver_stats["fuel_consumption_lt_100km"] = np.random.normal(12, 2, len(driver_stats)).clip(8, 20)
    driver_stats["driving_score"] = np.random.normal(75, 12, len(driver_stats)).clip(40, 100)
    driver_stats["experience_years"] = np.random.randint(1, 20, len(driver_stats))

    # Performans kategorisi
    # Gecikme oranı ve memnuniyet skoru birlikte değerlendiriliyor
    def classify_driver(row):
        if row["delay_rate"] < 0.10 and row["avg_satisfaction"] >= 4.2:
            return "Yıldız Sürücü"
        elif row["delay_rate"] < 0.20 and row["avg_satisfaction"] >= 3.5:
            return "İyi Performans"
        elif row["delay_rate"] < 0.30:
            return "Ortalama"
        else:
            return "Gelişim Gerekli"

    driver_stats["performance_category"] = driver_stats.apply(classify_driver, axis=1)

    # Sayısal değerleri yuvarla
    for col in ["avg_satisfaction", "avg_delay_days", "delay_rate",
                "avg_daily_km", "fuel_consumption_lt_100km", "driving_score"]:
        driver_stats[col] = driver_stats[col].round(3)

    print(f"   ✅ {len(driver_stats)} sürücü profili oluşturuldu.")
    return driver_stats


# ──────────────────────────────────────────────────────────
# 5. ANA ÇALIŞTIRICI
# ──────────────────────────────────────────────────────────

def main():
    """
    Tüm veri setlerini sırayla üretir ve CSV olarak kaydeder.
    Bağımlılık sırası: orders → deliveries → drivers
    """
    print("\n" + "="*60)
    print("  LOJİSTİK OPTİMİZASYON - VERİ ÜRETME SÜRECİ")
    print("="*60 + "\n")

    # Çıktı dizini oluştur
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Siparişler
    orders_df = generate_orders()
    orders_df.to_csv(f"{output_dir}/orders.csv", index=False, encoding="utf-8-sig")
    print(f"   💾 data/raw/orders.csv kaydedildi.\n")

    # 2. Teslimatlar (siparişlere bağlı)
    deliveries_df = generate_deliveries(orders_df)
    deliveries_df.to_csv(f"{output_dir}/deliveries.csv", index=False, encoding="utf-8-sig")
    print(f"   💾 data/raw/deliveries.csv kaydedildi.\n")

    # 3. Depo performansı
    warehouse_df = generate_warehouse_performance()
    warehouse_df.to_csv(f"{output_dir}/warehouse_performance.csv", index=False, encoding="utf-8-sig")
    print(f"   💾 data/raw/warehouse_performance.csv kaydedildi.\n")

    # 4. Sürücü performansı
    drivers_df = generate_driver_performance(deliveries_df)
    drivers_df.to_csv(f"{output_dir}/driver_performance.csv", index=False, encoding="utf-8-sig")
    print(f"   💾 data/raw/driver_performance.csv kaydedildi.\n")

    # Özet istatistikler
    print("="*60)
    print("  ÜRETİLEN VERİ ÖZETİ")
    print("="*60)
    print(f"  📦 Toplam sipariş:          {len(orders_df):>10,}")
    print(f"  🚚 Toplam teslimat:         {len(deliveries_df):>10,}")
    print(f"  🏭 Depo kayıt satırları:    {len(warehouse_df):>10,}")
    print(f"  👤 Sürücü profilleri:       {len(drivers_df):>10,}")
    print(f"  📅 Kapsanan dönem:          2023-01-01 → 2024-12-31")
    print(f"  ⚠️  Ortalama gecikme oranı: {deliveries_df['is_delayed'].mean():>9.1%}")
    print(f"  ⭐ Ort. müşteri memnuniyeti:{deliveries_df['customer_satisfaction'].mean():>9.2f}/5")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
