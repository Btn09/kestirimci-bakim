"""Dört alt kümede LightGBM ve 1D-CNN eğitir, test motorlarında karşılaştırır. FD001 modellerini servis için kaydeder.

Çalıştırma:  python -m src.egit
"""
import json
import time

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import torch

from . import cnn
from .metrikler import nasa_skoru, rmse
from .ozellikler import diziler, pencere_ozellikleri
from .veri import KOK, KOSUL_SAYISI, RUL_TAVANI, Normalizer, gercek_rul, motor_bol, oku

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELLER = KOK / "models"
RAPOR = KOK / "reports"
GRAFIK = RAPOR / "figures"
LGBM_AYAR = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
                 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


def lgbm_egit(X_e, y_e, X_d, y_d):
    e = lgb.Dataset(X_e, y_e)
    return lgb.train(LGBM_AYAR, e, 3000, valid_sets=[lgb.Dataset(X_d, y_d, reference=e)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def alt_kume_calistir(ad):
    egitim_ham, test_ham = oku(ad, "train"), oku(ad, "test")
    y_test = gercek_rul(ad)
    norm = Normalizer(KOSUL_SAYISI[ad]).fit(egitim_ham)
    egitim, test = norm.transform(egitim_ham), norm.transform(test_ham)
    e, d = motor_bol(egitim)

    F_e, F_d, F_t = pencere_ozellikleri(e), pencere_ozellikleri(d), pencere_ozellikleri(test)
    son = test.groupby("motor").tail(1).index
    bas = time.time()
    lgbm = lgbm_egit(F_e, e["rul"], F_d, d["rul"])
    t_lgbm = time.time() - bas
    p_lgbm = np.clip(lgbm.predict(F_t.loc[son], num_iteration=lgbm.best_iteration), 0, RUL_TAVANI)

    X_e, y_e, _ = diziler(e)
    X_d, y_d, _ = diziler(d)
    X_t, _, _ = diziler(test, sadece_son=True)
    bas = time.time()
    # Tek bir başlangıç noktasına bağlı kalmamak için 3 farklı tohumla eğitip ortalamasını alıyorum
    modeller = [cnn.egit(X_e, y_e, X_d, y_d, seed=s) for s in range(3)]
    t_cnn = time.time() - bas
    p_cnn = np.clip(np.mean([cnn.tahmin(m, X_t) for m in modeller], 0), 0, RUL_TAVANI)

    sonuc = {}
    for model_adi, p in [("LightGBM", p_lgbm), ("1D-CNN", p_cnn), ("Ortalama", (p_lgbm + p_cnn) / 2)]:
        sonuc[model_adi] = {"rmse": rmse(y_test, p), "nasa_skoru": nasa_skoru(y_test, p)}
    sonuc["egitim_sn"] = {"LightGBM": round(t_lgbm, 1), "1D-CNN": round(t_cnn, 1)}
    sonuc["test_motor"] = int(len(y_test))
    print(ad, {k: (round(v["rmse"], 2), round(v["nasa_skoru"])) for k, v in sonuc.items()
               if isinstance(v, dict) and "rmse" in v},
          sonuc["egitim_sn"], flush=True)

    if ad == "FD001":
        lgbm.save_model(str(MODELLER / "lgbm_fd001.txt"), num_iteration=lgbm.best_iteration)
        torch.save([{k: v.cpu() for k, v in m.state_dict().items()} for m in modeller], MODELLER / "cnn_fd001.pt")
        joblib.dump(norm, MODELLER / "normalizer_fd001.joblib")
        (MODELLER / "lgbm_fd001_agac.txt").write_text(str(lgbm.best_iteration))
        grafikler(y_test, p_lgbm, p_cnn, test, lgbm, modeller, F_t)
    return sonuc


def grafikler(y_test, p_lgbm, p_cnn, test, lgbm, modeller, F_t):
    sira = np.argsort(y_test)
    fig, eksen = plt.subplots(figsize=(10, 4))
    eksen.plot(y_test[sira], "k-", lw=2, label="Gerçek kalan ömür")
    eksen.plot(((p_lgbm + p_cnn) / 2)[sira], "o", ms=4, label="Tahmin (ortalama)")
    eksen.set_xlabel("Test motorları (gerçek kalan ömre göre sıralı)")
    eksen.set_ylabel("Kalan ömür (döngü)")
    eksen.set_title("FD001 test seti: 100 motor")
    eksen.legend()
    plt.tight_layout()
    plt.savefig(GRAFIK / "test_tahminleri.png", dpi=140)
    plt.close(fig)

    # Bir motorun tüm kaydı boyunca tahmin
    motor = test.groupby("motor")["dongu"].max().idxmax()
    m = test[test.motor == motor]
    p = np.clip(lgbm.predict(F_t.loc[m.index]), 0, RUL_TAVANI)
    son_gercek = y_test[motor - 1]
    gercek = np.minimum(son_gercek + (m["dongu"].max() - m["dongu"].values), RUL_TAVANI)
    fig, eksen = plt.subplots(figsize=(9, 4))
    eksen.plot(m["dongu"], gercek, "k-", lw=2, label="Gerçek (125'te kırpılmış)")
    eksen.plot(m["dongu"], p, label="LightGBM tahmini")
    eksen.set_xlabel("Döngü")
    eksen.set_ylabel("Kalan ömür")
    eksen.set_title(f"Test motoru {motor}: çalışma boyunca tahmin")
    eksen.legend()
    plt.tight_layout()
    plt.savefig(GRAFIK / "motor_omru.png", dpi=140)
    plt.close(fig)

    onem = pd.Series(lgbm.feature_importance("gain"), index=lgbm.feature_name()).nlargest(12)[::-1]
    fig, eksen = plt.subplots(figsize=(7, 4.5))
    eksen.barh(onem.index, onem.values)
    eksen.set_title("LightGBM: en etkili özellikler (FD001)")
    plt.tight_layout()
    plt.savefig(GRAFIK / "ozellik_onemi.png", dpi=140)
    plt.close(fig)


def main():
    for k in (MODELLER, GRAFIK):
        k.mkdir(parents=True, exist_ok=True)
    sonuclar = {ad: alt_kume_calistir(ad) for ad in ["FD001", "FD002", "FD003", "FD004"]}
    with open(RAPOR / "sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
