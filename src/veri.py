"""NASA C-MAPSS turbofan motor bozulma simülasyonu.

Her motor sağlam başlıyor, bir arıza gelişiyor ve bozulana kadar çalışıyor (eğitim). Test motorlarında kayıt
arızadan bir süre önce kesiliyor; görev o andaki kalan ömrü (RUL, döngü) tahmin etmek.

FD001: tek çalışma koşulu, tek arıza türü    FD002: 6 koşul, tek arıza
FD003: tek koşul, iki arıza türü               FD004: 6 koşul, iki arıza (en zoru)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "cmapss"

SUTUNLAR = ["motor", "dongu", "ayar1", "ayar2", "ayar3"] + [f"s{i}" for i in range(1, 22)]
# Tüm alt kümelerde neredeyse sabit kalan ya da bilgi taşımayan sensörler
SENSORLER = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
RUL_TAVANI = 125  # Motorun ilk dönemi sağlıklı; bozulma belirli bir noktadan sonra başlıyor


def oku(alt_kume, bolum):
    df = pd.read_csv(VERI / f"{bolum}_{alt_kume}.txt", sep=r"\s+", header=None, names=SUTUNLAR)
    if bolum == "train":
        omur = df.groupby("motor")["dongu"].transform("max")
        df["rul"] = (omur - df["dongu"]).clip(upper=RUL_TAVANI)
    return df


def gercek_rul(alt_kume):
    return np.loadtxt(VERI / f"RUL_{alt_kume}.txt")


class Normalizer:
    """Çoklu çalışma koşulu olan alt kümelerde (FD002, FD004) sensör değerleri koşula göre çok değişiyor.
    Uçuş ayarlarını kümeleyip her koşul içinde ayrı standartlaştırıyorum; tek koşullu kümelerde tek grup."""

    def __init__(self, kosul_sayisi):
        self.k = kosul_sayisi

    def fit(self, df):
        ayar = df[["ayar1", "ayar2", "ayar3"]].round(1).values
        self.km = KMeans(self.k, n_init=10, random_state=0).fit(ayar) if self.k > 1 else None
        kosul = self._kosul(df)
        self.ort = {c: df.loc[kosul == c, SENSORLER].mean() for c in range(self.k)}
        self.std = {c: df.loc[kosul == c, SENSORLER].std().replace(0, 1) for c in range(self.k)}
        return self

    def _kosul(self, df):
        if self.km is None:
            return np.zeros(len(df), int)
        return self.km.predict(df[["ayar1", "ayar2", "ayar3"]].round(1).values)

    def transform(self, df):
        df = df.copy()
        df[SENSORLER] = df[SENSORLER].astype(float)
        kosul = self._kosul(df)
        for c in range(self.k):
            m = kosul == c
            df.loc[m, SENSORLER] = (df.loc[m, SENSORLER] - self.ort[c]) / self.std[c]
        df["kosul"] = kosul
        return df


KOSUL_SAYISI = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def motor_bol(df, oran=0.2, seed=0):
    """Doğrulama için motorların %20'si ayrılıyor (aynı motorun döngüleri iki tarafa bölünmesin)."""
    motorlar = df["motor"].unique()
    rng = np.random.default_rng(seed)
    dogrulama = set(rng.choice(motorlar, int(len(motorlar) * oran), replace=False))
    m = df["motor"].isin(dogrulama)
    return df[~m], df[m]
