# FastAPI uygulamasinin giris noktasi.
# Burada sadece HTTP katmani (endpoint'ler, hata kodlari) var; siniflandirma
# mantigi model.py'da, veri sekilleri schemas.py'da, sabitler config.py'da.

import logging
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.model import classifier
from app.schemas import PredictionResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disaster Image Analysis API",
    description="Yuklenen bir goruntuyu fire, flood, collapsed_building veya blocked_road olarak siniflandirir.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    # Basit bir "ayakta miyim" kontrolu; deployment/monitoring icin kullanilir.
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """
    Multipart form-data ile gelen bir goruntu dosyasini alir, dogrular ve
    CLIP tabanli siniflandiricidan bir tahmin doner. Hata kontrolu asamali
    ilerler: once content-type, sonra boyut, en son goruntunun gecerliligi.
    """

    # 1) Content-type kontrolu: tarayici/istemci ne beyan ettiyse ona bakiyoruz.
    # Bu ucuz bir kontrol oldugu icin dosya icerigini okumadan once yapiyoruz.
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Desteklenmeyen content-type: '{file.content_type}'. Izin verilenler: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    content = await file.read()

    # 2) Boyut kontrolu: cok buyuk dosyalar sunucuyu gereksiz yormasin diye
    # iceriği tamamen belleğe okuduktan hemen sonra kontrol ediyoruz.
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dosya {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limitini asiyor.",
        )

    # 3) Bos dosya kontrolu: content-type dogru olsa bile icerik bos gelebilir.
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Yuklenen dosya bos.")

    # 4) Goruntunun gercekten acilabilir/bozuk olmadigini dogruluyoruz.
    # image.verify() dosyanin yapisini kontrol eder ama sonrasinda image
    # nesnesini kullanilmaz hale getirir; bu yuzden verify'dan sonra
    # dosyayi tekrar aciyoruz.
    try:
        image = Image.open(BytesIO(content))
        image.verify()
        image = Image.open(BytesIO(content))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Yuklenen dosya gecerli bir goruntu degil.") from exc

    # Buraya kadar geldiysek elimizde gercekten acilabilen bir goruntu var.
    predicted_class, confidence = classifier.predict(image)
    return PredictionResponse(predicted_class=predicted_class, confidence=confidence)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Yukarida yakalanmayan her turlu beklenmedik hata buraya duser.
    # Kullaniciya stack trace gostermiyoruz (guvenlik), ama sunucu
    # loglarina tam hatayi yaziyoruz ki debug edilebilsin.
    logger.exception("Beklenmeyen bir hata olustu")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Sunucu tarafinda beklenmeyen bir hata olustu."},
    )
