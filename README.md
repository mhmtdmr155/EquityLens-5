# EquityLens-5

## Proje Tanımı
EquityLens-5, üniversite Makine Öğrenmesi dersi kapsamında geliştirilmekte olan çok kaynaklı finansal analiz ve karar destek sistemidir. Proje; finansal piyasa verilerini, temel göstergeleri ve piyasa duyarlılık analizlerini bir araya getirerek yatırım kararlarını destekleyecek makine öğrenmesi modelleri ve analitik altyapı sunmayı amaçlamaktadır.

## Akademik ve Araştırma Amacı
Bu çalışmanın temel araştırma amacı; çok kaynaklı verilerden (sayısal zaman serileri, temel finansal oranlar ve metinsel finansal duyarlılık verileri) beslenen makine öğrenmesi ve doğal dil işleme modellerinin finansal analiz ve yön kestirimindeki doğruluğunu akademik standartlarda değerlendirmektir.

## Takip Edilen Şirketler
Sistem aşağıdaki beş şirketin verilerini kapsayacak şekilde tasarlanmaktadır:
- **AAPL** (Apple Inc.)
- **NVDA** (NVIDIA Corporation)
- **MU** (Micron Technology, Inc.)
- **DELL** (Dell Technologies Inc.)
- **SPCX** (Space Exploration Technologies Corp. - SpaceX)

## Planlanan Teknoloji Mimarisi
- **Backend:** Python 3.12, FastAPI
- **Veritabanı:** PostgreSQL
- **Frontend:** React, TypeScript, Vite
- **Veri Analitiği:** Pandas, NumPy
- **Makine Öğrenmesi:** scikit-learn, XGBoost
- **Finansal NLP:** FinBERT (Hugging Face / PyTorch)

## Proje Klasör Yapısı
- `backend/`: FastAPI uygulama mimarisi ve API uç noktaları.
- `frontend/`: React + TypeScript kullanıcı arayüzü bileşenleri.
- `ml/`: Özellik mühendisliği, model eğitimi, değerlendirme ve tahmin boru hatları.
- `scripts/`: Veri yönetimi, tohumlama ve yardımcı geliştirme araçları.
- `docs/`: Proje teknik şartnamesi ve akademik dokümantasyon.
- `tests/`: Entegrasyon, birim ve sistem testleri.
- `AGENTS.md`: Projenin kalıcı geliştirme ve GitHub çalışma protokolü kuralları.

## Mevcut Geliştirme Durumu
- **Aşama:** 00 – Ortam ve Proje Denetimi (EL5-002)
- **Durum:** Temel dizin iskeleti, yerel Git deposu ve izole Python sanal ortamı (`.venv`) yapılandırılmıştır. Henüz uygulama kodu yazımına veya paket yüklemelerine başlanmamıştır.

