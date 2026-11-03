# EquityLens-5 | Kalıcı Ajan ve Çalışma Protokolü

Bu belge, EquityLens-5 projesinde çalışan yapay zeka asistanları (Gemini / Antigravity) ve geliştiriciler için bağlayıcı, kalıcı geliştirme protokolünü tanımlar. Bütün geliştirme oturumlarında bu kurallar esas alınır.

---

## 1. Proje Kimliği ve Kapsam
* **Proje Adı:** EquityLens-5
* **Ana Plan Sürümü:** v1.2
* **Ders Bağlamı:** Üniversite Makine Öğrenmesi (Machine Learning) Dersi
* **Takip Edilen Şirketler:**
  1. **AAPL** (Apple Inc.)
  2. **NVDA** (NVIDIA Corporation)
  3. **MU** (Micron Technology, Inc.)
  4. **DELL** (Dell Technologies Inc.)
  5. **SPCX** (Space Exploration Technologies Corp. - SpaceX)
  *(Not: SPCX / SpaceX şirketi analiz kapsamından çıkarılamaz ve değiştirilemez).*

---

## 2. Geliştirme Aşamaları
* **00 – Ortam ve Proje Denetimi** *(Mevcut Aşama)*
* **01 – API Erişimi ve Veri Doğrulama**
* **02 – Proje Altyapısı ve Veritabanı**
* **03 – Veri Toplama**
* **04 – Finansal Göstergeler ve Özellikler (Feature Engineering)**
* **05 – İlk Makine Öğrenmesi Modelleri (Baseline Models)**
* **06 – Gelişmiş Veri ve ML Deneyleri (Ensemble & NLP)**
* **07 – Risk ve Karar Desteği**
* **08 – Web Uygulaması ve Entegrasyon (FastAPI + React)**
* **09 – Son Testler ve Akademik Teslim**

---

## 3. Görev ve Çalışma Disiplini
* **Raporlama Standardı:** Her görevde; görev kodu, yapılacak işlemler, teknik gerekçe, gerçek dosya değişiklikleri, test sonuçları, hatalar ve kabul durumu net olarak raporlanacaktır.
* **Katı Onay Kapısı:** Kullanıcının açık onayı olmadan bir sonraki ana veya alt göreve kendiliğinden geçilmez.
* **Model Tercihi:** Model düşünme seviyesi her zaman **HIGH** olarak yapılandırılır.
* **Doğrulama İlkesi:** *"Söylenen değil, doğrulanan iş kabul edilir."* Gerçek komut kanıtı ve exit code olmadan hiçbir işlem başarılı (PASS) gösterilemez.

---

## 4. Git ve GitHub Politikası
1. **Otomatik Commit/Push Yasağı:** Her prompt sonunda otomatik commit veya push **YAPILMAZ**. Gün içindeki tüm değişiklikler yerel çalışma alanında tutulur.
2. **Korumalı Dosyalar:** `.env`, `.env.*` (istisna: `.env.example`), API anahtarları, parolalar, `.venv/`, `node_modules/`, `*.log`, `.vscode/` ve büyük model ağırlıkları (`*.pt`, `*.bin`, `*.onnx` vb.) kesinlikle Git'e veya GitHub'a gönderilemez.
3. **Branch ve Push Kuralları:**
   * Ana geliştirme dalı **main** dalıdır.
   * `git push --force` kullanımı kesinlikle yasaktır.
   * Yalnızca kullanıcının kendi hesabındaki **PRIVATE** EquityLens-5 repository'si kullanılır.

---

## 5. Gün Sonu GitHub Protokolü (Kalıcı Kural)
Kullanıcı oturumu sonlandırmaya yönelik ifadeler kullandığında (örn. *"Bugünlük yeter"*, *"Bugün burada bırakalım"*, *"Bugünlük bu kadar"*, *"Bugünün çalışmasını bitirdik"*, *"Şimdilik burada duralım"*, *"Yarın devam ederiz"*):

1. **Otomatik Push Yapılmaz:**
2. **Kullanıcıya Standart Onay Sorusu Sorulur:**
   > *"Bugünkü bütün uygun değişiklikleri kontrol edip açıklayıcı commit mesajlarıyla GitHub'daki EquityLens-5 repository'sine göndermemi ister misiniz?"*
3. **Kullanıcının Açık "EVET" Yanıtı Beklenir.**
   * **Kullanıcı HAYIR derse:** Değişiklikler yerelde bırakılır, henüz gönderilmeyen dosyalar listelenir, push yapılmaz.
   * **Kullanıcı EVET derse:** Aşağıdaki protokol adımları işletilir:
     * `git status` ve `git diff` ile değişiklikler incelenir.
     * Gizli bilgi, API key ve büyük dosya taraması yapılır.
     * `.gitignore` kuralları test edilir.
     * Testler çalıştırılır; çalıştırılamayanlar raporda belirtilir.
     * Değişiklikler mantıklı gruplara ayrılarak anlamlı commit mesajları yazılır (örn. `feat: ...`, `fix: ...`, `chore: ...`, `docs: ...`, `test: ...`).
     * `origin` ve `main` doğrulanarak güvenli normal push (`git push origin main`) gerçekleştirilir.
     * Uzak GitHub deposu teyit edilir.

