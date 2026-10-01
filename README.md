#Bu çalışma, ders kapsamında ekip olarak geliştirilen bir projenin bireysel olarak yeniden yapılandırılmış halidir.
# Parfüm Markası Sınıflandırması

Fragrantica verisindeki en popüler 10 markanın parfümlerini; koku notaları (Top / Middle / Base), cinsiyet, puan ve yıl bilgisinden tahmin eden bir makine öğrenmesi çalışması. Marka tahmini pratik bir uygulamadan çok, koku notalarının markaya özgü bir imza taşıyıp taşımadığını ölçen bir vekil (proxy) görevdir.

## Veri

- Dosya: `fra_cleaned.csv` (24.063 parfüm, ayırıcı `;`, kodlama `latin-1`)
- Kaynak: Kaggle, [Fragrantica.com Fragrance Dataset](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset) (kullanıcı: olgagmiufana1). Veri Fragrantica.com'dan web scraping ile derlenmiştir; `fra_cleaned.csv`, veri setinin kısmen temizlenmiş sürümüdür.
- Lisans: Kaggle sayfasındaki lisans bilgisine bakınız (burada doğrulanmadı). Lisans ve kaynak sitenin kullanım koşulları belirsiz olduğu için veri dosyası bu repoda paylaşılmaz; kendi kopyanı Kaggle'dan indirip `data/` klasörüne koyman gerekir.
- Çalışmada en sık geçen 10 marka (Avon, Zara, O Boticário, Guerlain, Natura, Oriflame, Yves Saint Laurent, Dior, Givenchy, Giorgio Armani) kullanılır; `Year` eksik satırlar düşürüldükten sonra 3.083 örnek kalır.

## Yöntem

1. **Hazırlık:** metin boşlukları boş string, `Year` eksik satırlar düşürülür; `Rating Count` log dönüşümü, `Year` 1950 alt sınırıyla kırpılır.
2. **EDA:** sınıf dağılımı, korelasyon, tanımlayıcı istatistikler, çoklu doğrusallık kontrolü.
3. **Boyut indirgeme (keşifsel):** TF-IDF → TruncatedSVD, t-SNE (perplexity 15/30/50), UMAP, silhouette skoru.
4. **Bölme:** %80 eğitim / %20 test, stratified. Tüm öğrenilen dönüşümler yalnızca eğitim verisinden öğrenilir.
5. **Modelleme:** `ön işleme → RFE (50 özellik) → sınıflandırıcı` pipeline'ı; Logistic Regression, Random Forest, k-NN; 5 katmanlı stratified CV ile `GridSearchCV` (makro-F1). Şampiyon model **CV skoruna göre** seçilir, test seti yalnızca raporlama içindir.
6. **Ablation:** şampiyon modelin yalnızca koku notalarıyla performansı.
7. **Ek deneyler:** özel nota tokenizer'ı ve gradient boosting (`extra_experiments.py`).

## Sonuçlar

Örnek bir çalıştırmanın sonuçları (`random_state=42`, 617 test örneği):

| Model | CV makro-F1 | Test doğruluğu | Test makro-F1 |
|---|---|---|---|
| Random Forest | 0.408 | 0.491 | 0.433 |
| Logistic Regression | 0.341 | 0.379 | 0.351 |
| k-NN | 0.310 | 0.327 | 0.287 |
| *Baseline (çoğunluk sınıfı)* | – | 0.178 | – |

- 10 sınıflı problemde rastgele tahmin %10, çoğunluk sınıfı baseline'ı %17.8'dir; en iyi model bunun yaklaşık 2.7 katı doğruluk elde eder.
- Yalnızca koku notalarıyla doğruluk 0.407'ye düşer; puan, popülerlik ve yıl özellikleri belirgin katkı yapar.
- Küçük kütüphane sürümü farklarıyla sayılar hafifçe değişebilir.

### Denenen ama iyileşme sağlamayan fikirler

Aynı bölme ve CV protokolüyle iki fikir denendi; ikisi de anlamlı bir kazanç getirmedi:

| Metin temsili | Model | CV makro-F1 | Test doğruluğu |
|---|---|---|---|
| Kelime n-gram | Random Forest | 0.408 | 0.486 |
| Kelime n-gram | Gradient Boosting | 0.417 | 0.452 |
| Nota birimi (virgülle ayrılmış) | Random Forest | 0.406 | 0.465 |
| Nota birimi (virgülle ayrılmış) | Gradient Boosting | 0.402 | 0.476 |

Tüm varyantlar CV makro-F1'de 0.40–0.42 bandında kalıyor; farklar 617 örnekli test setinin belirsizliği (yaklaşık ±4 puan) içinde. Performans tavanı tokenizer veya model ailesinden çok verinin kendisinden (markaların benzer koku profilleri) kaynaklanıyor gibi görünüyor.

## Sınırlılıklar

- Test seti küçük ve tek bölmeye dayanıyor; güven aralığı geniştir.
- Yalnızca 10 marka kapsanır, sonuçlar genellenemez; sınıflar dengesizdir.
- Benzer koku profilli markalar (ör. Avon / Oriflame) sık karışır.
- Metin özellikleri varsayılan TF-IDF tokenizer'ıyla üretilir; çok kelimeli notalar parçalanır (özel tokenizer denendi, fark yaratmadı).

## Çalıştırma

```bash
pip install -r requirements.txt
# fra_cleaned.csv dosyasını data/ klasörüne koy
jupyter notebook parfum_marka_siniflandirma.ipynb
```

Google Colab'da çalıştırırsan dosyayı Drive'ın kök dizinine (`MyDrive/fra_cleaned.csv`) koyman yeterli. Notebook ortamı kendisi algılar.

Notebook'un tam çalıştırması CPU sayısına bağlı olarak birkaç dakika sürer (RFE + GridSearchCV en ağır adımlardır). Ek deneyler için `python extra_experiments.py` (tek çekirdekte yaklaşık 8–10 dakika).

## Yapı

```
├── parfum_marka_siniflandirma.ipynb
├── extra_experiments.py  # tokenizer ve gradient boosting deneyleri
├── data/                # fra_cleaned.csv (isteğe bağlı olarak repoya eklenmez)
├── requirements.txt
└── README.md
```
