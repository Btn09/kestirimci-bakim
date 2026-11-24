"""Bakım kararı: tahmine dayalı bakım ile sabit aralıklı (periyodik) bakımın maliyet karşılaştırması (FD001).

Test motorlarının kaydı arızadan önce kesildiği için bir bakım stratejisini baştan sona sınamak mümkün değil.
Bu yüzden eğitimdeki 100 motorun (hepsi bozulana kadar çalışmış) tüm ömrü boyunca tahmin üretiyorum:
motorları 5 gruba bölüp her grubu, diğer 4 grupla eğitilen modelle tahmin ediyorum (çapraz tahmin).
Stratejinin eşiği de aynı şekilde: her grup için eşik diğer gruplarda seçiliyor, yani hiçbir motor kendi
sonucunu gördüğü bir ayarla değerlendirilmiyor.

Maliyet varsayımları (göreli birim):
  - Plansız arıza: 100 (motor uçuşta ya da sahada bozulur, ikincil hasar, uçuş iptali)
  - Planlı bakım: 20
  - Boşa giden ömür: bakım sırasında motorun kalan her döngüsü için 0,1 (parça henüz kullanılabilirken değişiyor)

Çalıştırma:  python -m src.bakim   (önce python -m src.egit)
"""
import json

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

from .egit import GRAFIK, LGBM_AYAR, RAPOR, MODELLER
from .ozellikler import pencere_ozellikleri
from .veri import RUL_TAVANI, Normalizer, oku

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARIZA, PLANLI, BOSA_DONGU = 100.0, 20.0, 0.1


def capraz_tahmin(df, F, agac, kat=5, seed=0):
    motorlar = df["motor"].unique()
    gruplar = np.random.default_rng(seed).permutation(len(motorlar)) % kat
    grup_of = dict(zip(motorlar, gruplar))
    g = df["motor"].map(grup_of).values
    p = np.zeros(len(df))
    for k in range(kat):
        m = lgb.train(LGBM_AYAR, lgb.Dataset(F[g != k], df["rul"][g != k]), agac)
        p[g == k] = np.clip(m.predict(F[g == k]), 0, RUL_TAVANI)
    return p, g


def motor_maliyeti(dongu, tahmin, omur, esik):
    """Tahmin ilk kez eşiğin altına indiğinde bakım. Hiç inmezse motor ömrünün sonunda bozulur."""
    alarm = np.flatnonzero(tahmin <= esik)
    if len(alarm) == 0:
        return ARIZA, True
    kalan = omur - dongu[alarm[0]]
    return PLANLI + BOSA_DONGU * kalan, False


def periyodik_maliyet(omur, aralik):
    """Her motor 'aralik' döngüde bakıma alınır; ömrü daha kısaysa ondan önce bozulur."""
    return (ARIZA, True) if omur <= aralik else (PLANLI + BOSA_DONGU * (omur - aralik), False)


def main():
    df = oku("FD001", "train")
    df = Normalizer(1).fit(df).transform(df)
    F = pencere_ozellikleri(df)
    agac = int((MODELLER / "lgbm_fd001_agac.txt").read_text())
    p, grup = capraz_tahmin(df, F, agac)
    df["tahmin"] = p
    motorlar = [(m, x["dongu"].values, x["tahmin"].values, x["dongu"].max(), grup[x.index[0]])
                for m, x in df.groupby("motor")]

    esikler = np.arange(5, 61, 1)
    araliklar = np.arange(100, 301, 1)

    def toplam(fn, ayar, secim):
        return sum(fn(m, ayar)[0] for m in secim)

    sonuc = {"Tahmine dayalı": [], "Periyodik": []}
    secilen = {"esik": [], "aralik": []}
    for k in range(5):
        egitim = [m for m in motorlar if m[4] != k]
        test = [m for m in motorlar if m[4] == k]
        e = min(esikler, key=lambda t: toplam(lambda m, a: motor_maliyeti(m[1], m[2], m[3], a), t, egitim))
        a = min(araliklar, key=lambda t: toplam(lambda m, x: periyodik_maliyet(m[3], x), t, egitim))
        secilen["esik"].append(int(e))
        secilen["aralik"].append(int(a))
        sonuc["Tahmine dayalı"] += [motor_maliyeti(m[1], m[2], m[3], e) for m in test]
        sonuc["Periyodik"] += [periyodik_maliyet(m[3], a) for m in test]
    # Kusursuz bilgi: her motor tam bozulmadan bir döngü önce bakıma alınır (alt sınır)
    sonuc["Kusursuz bilgi"] = [(PLANLI + BOSA_DONGU * 1, False) for _ in motorlar]
    sonuc["Bakım yok (bozulunca değiştir)"] = [(ARIZA, True) for _ in motorlar]

    satirlar = []
    for ad, liste in sonuc.items():
        maliyet = np.array([x[0] for x in liste])
        planli = [c for c, ariza in liste if not ariza]
        satirlar.append({"strateji": ad, "motor_basi_maliyet": round(float(maliyet.mean()), 1),
                         "ariza_sayisi": int(sum(x[1] for x in liste)), "motor": len(liste),
                         # Planlı bakımlarda motorun kullanılmadan bırakılan ortalama ömrü (döngü)
                         "ortalama_bosa_dongu": round(float(np.mean([(c - PLANLI) / BOSA_DONGU for c in planli])), 1)
                         if planli else None})
    tablo = pd.DataFrame(satirlar)
    print(tablo.to_string(index=False))
    print("Seçilen eşikler:", secilen)

    # Boşa giden ömür: planlı bakımlarda kalan ortalama döngü
    with open(RAPOR / "bakim.json", "w", encoding="utf-8") as f:
        json.dump({"varsayimlar": {"ariza": ARIZA, "planli_bakim": PLANLI, "bosa_giden_dongu": BOSA_DONGU},
                   "stratejiler": satirlar, "secilen": secilen,
                   "capraz_tahmin_rmse": float(np.sqrt(np.mean((df["tahmin"] - df["rul"]) ** 2)))},
                  f, ensure_ascii=False, indent=2)

    # Servis, katlarda seçilen eşiklerin ortancasını kullanıyor
    with open(MODELLER / "bakim_esigi.json", "w", encoding="utf-8") as f:
        json.dump({"esik": int(np.median(secilen["esik"])),
                   "kaynak": "src.bakim çapraz doğrulamasında seçilen eşiklerin ortancası"}, f, ensure_ascii=False)

    fig, eksen = plt.subplots(figsize=(7, 3.8))
    renk = ["tab:green", "tab:orange", "tab:gray", "tab:red"]
    eksen.barh(tablo["strateji"][::-1], tablo["motor_basi_maliyet"][::-1], color=renk[::-1])
    for i, s in enumerate(satirlar[::-1]):
        eksen.text(s["motor_basi_maliyet"] + 1, i, f"{s['motor_basi_maliyet']:.1f}  ({s['ariza_sayisi']} arıza)",
                   va="center", fontsize=9)
    eksen.set_xlabel("Motor başına ortalama maliyet (göreli birim)")
    eksen.set_xlim(0, 125)
    eksen.set_title("Bakım stratejileri (FD001, 100 motor)")
    plt.tight_layout()
    plt.savefig(GRAFIK / "bakim_stratejileri.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
