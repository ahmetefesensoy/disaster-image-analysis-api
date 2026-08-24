# Gercek goruntu siniflandirici burada tanimli.
# OpenAI'nin CLIP modelini (ViT-B/32, ImageNet uzerinde degil, 400M
# goruntu-metin ciftiyle egitilmis) zero-shot siniflandirma icin kullaniyoruz.
# Boylece "fire", "flood" gibi ozel bir egitim verisi toplamadan, modelin
# goruntu ile dogal dil aciklamalari arasindaki benzerligine dayanarak
# gercek bir tahmin uretiyoruz. Bu bir mock degil, gercekten goruntuye
# bakan bir model.

import clip
import torch
from PIL.Image import Image

from app.schemas import DisasterClass

# Her sinif icin CLIP'e verilecek dogal dil aciklamalari. Birden fazla
# varyasyon vermek (prompt ensembling) dogrulugu artirir; CLIP metin ile
# goruntu embedding'lerini karsilastirdigi icin aciklama ne kadar net ve
# gorsel olursa eslesme o kadar isabetli olur.
CLASS_PROMPTS: dict[DisasterClass, list[str]] = {
    DisasterClass.FIRE: [
        "a photo of a fire",
        "a photo of a wildfire burning",
        "a photo of a building on fire with smoke and flames",
    ],
    DisasterClass.FLOOD: [
        "a photo of a flood",
        "a photo of a flooded street with water covering the road",
        "a photo of floodwater surrounding houses",
    ],
    DisasterClass.COLLAPSED_BUILDING: [
        "a photo of a collapsed building",
        "a photo of building rubble and debris after an earthquake",
        "a photo of a destroyed building",
    ],
    DisasterClass.BLOCKED_ROAD: [
        "a photo of a road blocked by debris",
        "a photo of a road blocked by a fallen tree or rubble",
        "a photo of a landslide covering a road",
    ],
}


class ClipDisasterClassifier:
    """
    CLIP'i zero-shot siniflandirici olarak kullanir: goruntu embedding'i
    ile her sinifin metin embedding'leri arasindaki kosinus benzerligine
    bakar, en yuksek benzerlige sahip sinifi ve softmax ile normalize
    edilmis confidence degerini doner.
    """

    def __init__(self) -> None:
        # CPU yeterli; goruntu basina inference suresi ~100-300ms civarinda.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ViT-B/32: CLIP'in en kucuk/hizli varyanti, demo/API kullanimi icin yeterli.
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # CLIP'in kendi ogrendigi bir olcekleme katsayisi (genelde ~100).
        # Kosinus benzerlikleri (-1, 1) araliginda ve birbirine çok yakin
        # cikar; bu katsayi olmadan softmax neredeyse duz bir dagilim
        # uretir ve en yuksek benzerlige sahip sinif bile net one cikmaz.
        # CLIP'in resmi kullanimi da logit'leri bu katsayiyla carpar.
        self.logit_scale = self.model.logit_scale.exp().item()

        self.classes = list(CLASS_PROMPTS.keys())

        # Her sinifin birden fazla prompt'unun metin embedding'lerini
        # onceden (uygulama baslarken) hesaplayip belleğe aliyoruz.
        # Boylece her /predict istegi sadece goruntu embedding'i cikarir,
        # metin taraf her seferinde yeniden hesaplanmaz.
        with torch.no_grad():
            self.text_features_per_class: list[torch.Tensor] = []
            for disaster_class in self.classes:
                prompts = CLASS_PROMPTS[disaster_class]
                tokens = clip.tokenize(prompts).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                # Ayni sinifin birden fazla prompt'unu ortalayarak tek bir
                # temsilci vektore indiriyoruz (prompt ensembling).
                averaged = features.mean(dim=0)
                averaged = averaged / averaged.norm()
                self.text_features_per_class.append(averaged)

    def predict(self, image: Image) -> tuple[DisasterClass, float]:
        image_input = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Goruntu embedding'ini her sinifin metin embedding'iyle karsilastirip
            # benzerlik skorlarini (logit) cikariyoruz, logit_scale ile
            # olcekleyip softmax ile 0-1 arasi bir olasilik dagilimina ceviriyoruz.
            text_features = torch.stack(self.text_features_per_class)
            similarities = self.logit_scale * (image_features @ text_features.T)
            probabilities = similarities.softmax(dim=-1).squeeze(0)

        best_index = int(probabilities.argmax())
        predicted_class = self.classes[best_index]
        confidence = round(float(probabilities[best_index]), 4)
        return predicted_class, confidence


# Model agirliklari (ilk calistirmada internetten indirilir, sonra
# yerel cache'den yuklenir) sadece uygulama baslarken bir kere yuklenir.
# Her /predict istegi bu tek instance'i kullanir.
classifier = ClipDisasterClassifier()
