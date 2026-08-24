# Pydantic modelleri: request/response verisinin sekli burada tanimlanir.
# FastAPI bu modelleri hem validation hem de otomatik /docs (Swagger) ciktisi icin kullanir.

from enum import Enum

from pydantic import BaseModel, Field


class DisasterClass(str, Enum):
    """
    Modelin tahmin edebilecegi 4 afet sinifi.
    str'den turetmemizin nedeni: FastAPI/Pydantic bunu otomatik olarak
    JSON'da duz bir string olarak serialize edebilsin (ornek: "fire").
    """

    FIRE = "fire"
    FLOOD = "flood"
    COLLAPSED_BUILDING = "collapsed_building"
    BLOCKED_ROAD = "blocked_road"


class PredictionResponse(BaseModel):
    """
    /predict endpoint'inin donecegi response govdesi.
    confidence alani 0 ile 1 arasinda olmak zorunda; Pydantic bunu
    otomatik dogrular, biz manuel kontrol yazmak zorunda kalmayiz.
    """

    predicted_class: DisasterClass
    confidence: float = Field(ge=0.0, le=1.0, description="Modelin tahminine olan guveni (0-1 arasi)")
