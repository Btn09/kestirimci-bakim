import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def nasa_skoru(y, p):
    """PHM08 yarışmasının asimetrik skoru: geç tahmin (ömrü olduğundan uzun sanmak) erken tahminden daha ağır
    cezalandırılıyor, çünkü geç kalınırsa motor bakımdan önce bozulur. Düşük daha iyi."""
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
