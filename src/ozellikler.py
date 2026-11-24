"""Her döngü için geçmiş pencereden özellikler (LightGBM) ve sabit uzunlukta diziler (CNN)."""
import numpy as np
import pandas as pd

from .veri import SENSORLER

PENCERE = 30


def pencere_ozellikleri(df):
    """Her (motor, döngü) için son PENCERE döngünün sensör özeti: son değer, ortalama, std, eğim."""
    parcalar = []
    t = np.arange(PENCERE) - (PENCERE - 1) / 2
    for _, m in df.groupby("motor", sort=False):
        x = m[SENSORLER].values
        n = len(x)
        # Başta yeterli geçmiş yoksa ilk değerle doldur
        dolgu = np.vstack([np.repeat(x[:1], PENCERE - 1, 0), x])
        pencereler = np.lib.stride_tricks.sliding_window_view(dolgu, PENCERE, axis=0)  # (n, sensör, PENCERE)
        egim = (pencereler * t).sum(-1) / (t ** 2).sum()
        ozet = np.concatenate([x, pencereler.mean(-1), pencereler.std(-1), egim], axis=1)
        parca = pd.DataFrame(ozet, index=m.index, columns=[f"{s}_{a}" for a in ("son", "ort", "std", "egim")
                                                            for s in SENSORLER])
        parca["dongu"] = m["dongu"].values
        parca["kosul"] = m["kosul"].values if "kosul" in m else 0
        parcalar.append(parca)
        assert len(parca) == n
    return pd.concat(parcalar)


def diziler(df, sadece_son=False):
    """CNN için (örnek, PENCERE, sensör) diziler. sadece_son: her motorun yalnızca son döngüsü (test)."""
    X, y, index = [], [], []
    for _, m in df.groupby("motor", sort=False):
        x = m[SENSORLER].values.astype(np.float32)
        dolgu = np.vstack([np.repeat(x[:1], PENCERE - 1, 0), x])
        pencereler = np.lib.stride_tricks.sliding_window_view(dolgu, PENCERE, axis=0).transpose(0, 2, 1)
        if sadece_son:
            pencereler = pencereler[-1:]
        X.append(pencereler)
        if "rul" in m:
            y.append(m["rul"].values[-len(pencereler):])
        index.append(m.index[-len(pencereler):])
    return (np.concatenate(X), np.concatenate(y).astype(np.float32) if y else None, np.concatenate(index))
