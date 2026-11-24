"""Bir motorun sensör geçmişinden kalan ömür tahmini ve bakım önerisi (FD001 modeli). API bunu kullanıyor."""
import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .ozellikler import pencere_ozellikleri
from .veri import KOK, RUL_TAVANI

MODEL_KLASORU = Path(os.environ.get("MODEL_KLASORU", KOK / "models"))


def durum(rul, esik):
    if rul <= esik:
        return "hemen bakım"
    if rul <= 30:
        return "bakım planla"
    if rul <= 80:
        return "izlenmeli"
    return "sağlıklı"


class OmurTahmincisi:
    def __init__(self):
        self.model = lgb.Booster(model_file=str(MODEL_KLASORU / "lgbm_fd001.txt"))
        self.norm = joblib.load(MODEL_KLASORU / "normalizer_fd001.joblib")
        bakim = json.loads((MODEL_KLASORU / "bakim_esigi.json").read_text())
        self.esik = bakim["esik"]

    def tahmin_et(self, olcumler, ilk_dongu=1):
        """olcumler: döngü sırasıyla sensör sözlükleri (en az 1). Son döngü için kalan ömür."""
        df = pd.DataFrame(olcumler)
        df.insert(0, "motor", 1)
        df.insert(1, "dongu", np.arange(ilk_dongu, ilk_dongu + len(df)))
        for a in ("ayar1", "ayar2", "ayar3"):
            df[a] = 0.0
        F = pencere_ozellikleri(self.norm.transform(df))
        gecmis = np.clip(self.model.predict(F), 0, RUL_TAVANI)
        rul = float(gecmis[-1])
        return {"kalan_omur": round(rul, 1), "durum": durum(rul, self.esik), "bakim_esigi": self.esik,
                "son_10_dongu_tahmini": [round(float(x), 1) for x in gecmis[-10:]]}
