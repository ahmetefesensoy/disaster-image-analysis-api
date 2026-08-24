# Disaster Image Analysis API

Python + FastAPI ile yazılmış, bir görüntü alıp `fire`, `flood`,
`collapsed_building`, `blocked_road` sınıflarından birini ve bir confidence
değeri döndüren bir görüntü analiz API'si. Sıfırdan eğitilmiş özel bir
model yerine, önceden eğitilmiş açık kaynak **CLIP** (OpenAI) modelini
zero-shot sınıflandırıcı olarak kullanır; yani görüntüyü gerçekten
işler, rastgele sonuç üretmez. Amaç API tasarımını, hata kontrolünü ve
kod düzenini göstermektir.

## Proje Yapısı

```
app/
├── main.py       # FastAPI app, /predict ve /health endpoint'leri, hata yönetimi
├── model.py      # ClipDisasterClassifier, CLIP tabanlı zero-shot sınıflandırıcı
├── schemas.py    # DisasterClass Enum ve PredictionResponse (Pydantic)
└── config.py     # Sabitler (dosya boyutu limiti, izinli content-type'lar)
tests/
└── test_api.py   # pytest ile 6 senaryo (happy path + hata durumları)
```

Sınıflandırıcı ayrı bir dosyada, `predict(image) -> (class, confidence)`
arayüzüyle izole edilmiş durumda. Model değiştirilecek olsa (örneğin
kendi eğittiğin bir CNN ile) sadece `model.py` değişir, `main.py`'a
dokunmaya gerek kalmaz.

### Model nasıl çalışıyor?

Kendi verimizle bir model eğitmek yerine, CLIP'in görüntü ile doğal dil
açıklamaları arasındaki benzerliğe dayanan zero-shot sınıflandırma
yeteneğini kullanıyoruz. Her sınıf için birkaç doğal dil açıklaması
tanımlı (`"a photo of a fire"`, `"a photo of a flooded street"` gibi,
bkz. `app/model.py` içindeki `CLASS_PROMPTS`). Gelen görüntünün CLIP
embedding'i, bu açıklamaların embedding'leriyle karşılaştırılır; en
yüksek benzerliğe sahip sınıf ve softmax ile normalize edilmiş skoru
döner. Bu sayede API gerçekten görüntüye bakıyor, ama görev
tanımındaki "gerçek model kullanmak zorunda değilsin" ölçeğinde, özel
veri toplamadan/etiketlemeden kalıyoruz.

Softmax'tan önce CLIP'in kendi öğrendiği `logit_scale` katsayısıyla
(~100) ölçekleme yapılıyor; bu adım atlanırsa kosinüs benzerlikleri
birbirine çok yakın çıktığı için softmax neredeyse düz bir dağılım
üretir ve en olası sınıf bile net bir şekilde öne çıkmaz.

**Gerçek görüntülerle doğrulandı:** Wikimedia Commons'tan (telifsiz)
indirilen 4 gerçek fotoğrafla test edildi, hepsi doğru sınıflandırıldı:

| Görüntü | Beklenen | Sonuç | Confidence |
|---|---|---|---|
| Alevli orman yangını fotoğrafı | fire | fire ✓ | 0.9999 |
| Su basmış ev fotoğrafı | flood | flood ✓ | 0.9978 |
| Çökmüş bina fotoğrafı | collapsed_building | collapsed_building ✓ | 0.9958 |
| Heyelanla kapanmış karayolu fotoğrafı | blocked_road | blocked_road ✓ | 0.9991 |

Bu 4 örnek, sınıflar arasında net görsel farkı olan tipik fotoğraflardı.
Zero-shot bir model olduğu için (özel afet verisiyle fine-tune
edilmedi) belirsiz açılardan çekilmiş, birden fazla afeti aynı anda
gösteren veya düşük kaliteli görsellerde doğruluk düşebilir. Üretim
kullanımı için gerçek etiketli verilerle bir doğruluk/hata analizi
(confusion matrix) yapılması gerekir.

## Kurulum ve Çalıştırma

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```

İlk çalıştırmada CLIP'in model ağırlıkları (~350MB) internetten indirilip
yerel önbelleğe kaydedilir; sonraki çalıştırmalarda tekrar indirilmez.
API `http://localhost:8000` adresinde ayağa kalkar. İnteraktif dokümantasyon
(Swagger UI) için `http://localhost:8000/docs` adresine gidip "Try it out"
ile tarayıcıdan doğrudan dosya yükleyip test edebilirsin, ayrı bir demo
arayüzü yazmaya gerek kalmıyor.

### Docker ile çalıştırma

```bash
docker build -t disaster-image-api .
docker run -p 8000:8000 disaster-image-api
```

## API

| Endpoint | Method | Açıklama |
|---|---|---|
| `/health` | GET | Basit ayakta mı kontrolü |
| `/predict` | POST | Multipart form-data ile görüntü yükler, tahmin döner |

### Örnek istek

```bash
curl -F "file=@ornek.jpg" http://localhost:8000/predict
```

### Örnek cevap

```json
{
  "predicted_class": "fire",
  "confidence": 0.8734
}
```

### Hata durumları

| Durum | HTTP Kodu | Sebep |
|---|---|---|
| Desteklenmeyen content-type | 415 | `image/jpeg`, `image/png`, `image/webp` dışında bir dosya |
| Dosya çok büyük | 413 | 10MB limiti aşıldı |
| Boş dosya | 400 | İçerik uzunluğu 0 |
| Bozuk/geçersiz görüntü | 400 | Content-type doğru ama PIL dosyayı açamıyor |
| Beklenmeyen hata | 500 | Detay client'a sızdırılmaz, sunucu logunda tutulur |

Tüm hata cevapları `{"detail": "..."}` formatında tutarlıdır.

## Testler

```bash
pytest tests/ -v
```

6 test: health check, geçerli görüntü ile başarılı tahmin, yanlış
content-type reddi, bozuk görüntü reddi, boş dosya reddi, ve modelin
confidence skorunun geçerli aralıkta olduğu doğrulaması. Testler
`TestClient` (httpx tabanlı) kullanır ve gerçek dosya sistemine dokunmaz;
görüntüler PIL ile bellek içinde (in-memory) üretilir. İlk test
çalıştırmasında CLIP modeli yüklendiği için birkaç saniye ek süre alır.

## Teknik Yaklaşım

- **Katmanlı ayrım**: HTTP katmanı (`main.py`), veri şeması (`schemas.py`),
  iş mantığı (`model.py`) ve sabitler (`config.py`) birbirinden ayrı.
- **Zero-shot sınıflandırma**: Özel veri toplamadan/etiketlemeden, CLIP'in
  genel görüntü-metin anlayışına dayanarak gerçek bir tahmin üretiliyor.
  Model ağırlıkları uygulama başlarken bir kez yüklenir, her istekte
  tekrar yüklenmez.
- **Aşamalı doğrulama**: content-type → dosya boyutu → boş dosya kontrolü →
  görüntünün gerçekten açılabilir olması. Her adım en ucuz kontrolden en
  pahalıya doğru sıralı, gereksiz işlem yapılmıyor.
- **Pydantic ile response garantisi**: `confidence` alanı 0-1 aralığı
  dışında asla dönemez, bunu manuel kontrol etmeye gerek yok.
- **Genel exception handler**: beklenmeyen hatalarda stack trace client'a
  sızmaz, ama sunucu tarafında loglanır.

---

## "Bu sistemi NVIDIA Jetson üzerinde gerçek zamanlı drone görüntüsü analiz edecek hale getirseydin neleri değiştirirdin?"

Şu anki API senkron, tek görüntü/istek üzerine kurulu bir REST servisi.
Jetson üzerinde gerçek zamanlı drone görüntü akışı işlemek için mimarinin
büyük kısmı değişirdi:

**1. Model formatı ve donanım hızlandırma**
CLIP, boyutu ve genel amaçlı yapısı nedeniyle Jetson'da gerçek zamanlı
video için uygun değil; yerine görevle sınırlı, küçük ve hızlı bir
model (örneğin gerçek etiketli verilerle eğitilmiş bir YOLO/MobileNet
sınıflandırıcı) kullanır, ONNX'e ve oradan TensorRT engine'e çevirirdim.
Jetson Orin NX üzerinde YOLO11s ile ölçülen gerçek sayılar:
FP32'de 14.53ms/68.8 FPS, FP16'da 7.91ms/126 FPS, INT8'de 6.05ms/165 FPS
([Ultralytics Jetson kılavuzu](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)).
INT8 quantization için ~500 görüntülük bir kalibrasyon seti gerekiyor ama
karşılığında hem gecikme hem güç tüketimi belirgin şekilde düşüyor.

**2. REST/multipart yerine video pipeline**
Tek tek HTTP isteğiyle görüntü yüklemek yerine NVIDIA DeepStream
(GStreamer tabanlı, CUDA-native) kullanırdım. Decode, inference ve encode
adımlarının tamamı GPU belleğinde kalır, CPU-GPU arası gereksiz kopyalar
elenir ([DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)).

**3. Senkron endpoint yerine async/kuyruk mimarisi**
Gerçek zamanlı bir video akışında her frame'i ayrı ayrı, sırayla işlemek
darboğaz yaratır. Frame skipping ve batch inference ile asenkron bir kuyruk
yapısına geçerdim; API katmanı sadece sonuçları yayınlar, inference ayrı
bir worker'da koşar.

**4. Güç ve termal kısıtlar**
Jetson Orin Nano 7-15W arası çalışır; 15W modda tipik inference çekişi
8-12W civarında, 10W üstünde aktif soğutma (fan) neredeyse zorunlu, aksi
halde birkaç dakika içinde thermal throttling başlıyor
([Jetson Orin Nano güç analizi](https://edgeaistack.ai/blog/jetson-orin-nano-power-consumption/)).
Bu da model boyutu, hedef FPS ve güç bütçesi arasında bilinçli bir denge
kurmayı gerektirir; drone'un batarya ömrü doğrudan buna bağlı.

**5. Eşzamanlı akış sınırı**
15W modda Orin Nano ~4-6 eşzamanlı 1080p akışı (INT8, 15fps) kaldırabiliyor.
Drone'da genelde tek kamera akışı olacağı için Nano yeterli olur, ama
çoklu kamera/sensör füzyonu gerekiyorsa Orin NX'e geçmek gerekir.

**6. Bağlantı kopuklukları için dayanıklılık**
Drone'lar sık sık bağlantı kaybeder. Sonuçları anlık olarak buluta/yer
istasyonuna göndermek yerine yerel bir kuyrukta tutup bağlantı geldiğinde
senkronize eden bir yapı eklerdim; API'nin "her istek anında cevap
bekler" varsayımı burada geçerli olmaz.

Özetle: genel amaçlı CLIP → göreve özel, küçük TensorRT engine,
REST/multipart → DeepStream video pipeline, senkron endpoint → async
kuyruk, ve güç/termal kısıtlara göre
tasarlanmış bir model/FPS dengesi. Şu anki kod tabanının katmanlı yapısı
(`model.py`'ın izole arayüzü gibi) bu geçişi kolaylaştırır, ama HTTP
katmanının kendisi büyük ölçüde yeniden düşünülmesi gereken bir katman
olurdu.
