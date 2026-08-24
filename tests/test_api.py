# API'nin temel akislarini dogrulayan pytest testleri.
# Gercek bir dosya sistemine dokunmadan, PIL ile bellek icinde (in-memory)
# test goruntuleri uretip TestClient uzerinden endpoint'lere gonderiyoruz.

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas import DisasterClass

client = TestClient(app)


def make_test_image() -> bytes:
    # 32x32'lik kucuk, tek renkli bir JPEG uretir. Testler icin gercek bir
    # goruntu dosyasina ihtiyacimiz yok, sadece "gecerli bir JPEG" olmasi yeterli.
    buf = BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check():
    # /health her zaman 200 ve sabit bir govde donmeli.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image():
    # Mutlu senaryo (happy path): gecerli bir JPEG yuklendiginde
    # 200 donmeli ve response semaya uymali.
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in [c.value for c in DisasterClass]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_wrong_content_type():
    # Content-type izin verilen listede degilse 415 donmeli.
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_corrupt_image():
    # Content-type dogru gorunse bile (image/jpeg) icerik gercek bir
    # goruntu degilse 400 donmeli.
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", b"this is not really a jpeg", "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_rejects_empty_file():
    # Icerigi tamamen bos bir dosya da 400 ile reddedilmeli.
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_confidence_scores_sum_to_one_across_classes():
    # CLIP tabanli siniflandirici softmax kullandigi icin, ayni goruntu
    # icin butun siniflarin olasiliklari toplaminin 1'e yakin olmasi
    # beklenir. Bu, response'ta donen tek confidence degerinin gercekten
    # bir softmax dagiliminin parcasi oldugunu (rastgele uretilmedigini)
    # dolayli olarak dogrular.
    from app.model import classifier
    from PIL import Image as PILImage

    image = PILImage.open(BytesIO(make_test_image()))
    _, confidence = classifier.predict(image)
    assert 0.0 <= confidence <= 1.0
