"""Sensör pencerelerinden kalan ömür tahmini için 1 boyutlu evrişimli ağ."""
import numpy as np
import torch
from torch import nn


class OmurCNN(nn.Module):
    def __init__(self, sensor=14, kanal=64, birakma=0.2):
        super().__init__()
        self.govde = nn.Sequential(
            nn.Conv1d(sensor, kanal, 5, padding=2), nn.ReLU(),
            nn.Conv1d(kanal, kanal, 5, padding=2), nn.ReLU(),
            nn.Conv1d(kanal, kanal, 3, padding=1), nn.ReLU(),
        )
        self.kafa = nn.Sequential(nn.Flatten(), nn.Dropout(birakma), nn.Linear(kanal * 30, 100), nn.ReLU(),
                                  nn.Linear(100, 1))

    def forward(self, x):            # x: (grup, 30, sensör)
        return self.kafa(self.govde(x.transpose(1, 2))).squeeze(-1)


def egit(X_e, y_e, X_d, y_d, tur=60, grup=256, ogrenme=1e-3, cihaz=None, seed=0, sabir=10):
    cihaz = cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = OmurCNN(X_e.shape[2]).to(cihaz)
    opt = torch.optim.Adam(model.parameters(), lr=ogrenme)
    Xe, ye = torch.from_numpy(X_e).to(cihaz), torch.from_numpy(y_e).to(cihaz)
    Xd, yd = torch.from_numpy(X_d).to(cihaz), torch.from_numpy(y_d).to(cihaz)
    en_iyi, durum, bekleme = np.inf, None, 0
    for _ in range(tur):
        model.train()
        for sira in torch.randperm(len(Xe), device=cihaz).split(grup):
            kayip = ((model(Xe[sira]) - ye[sira]) ** 2).mean()
            opt.zero_grad()
            kayip.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            rmse = ((model(Xd) - yd) ** 2).mean().sqrt().item()
        if rmse < en_iyi:
            en_iyi, durum, bekleme = rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bekleme += 1
            if bekleme >= sabir:
                break
    model.load_state_dict(durum)
    return model.eval()


@torch.no_grad()
def tahmin(model, X):
    cihaz = next(model.parameters()).device
    return model(torch.from_numpy(X).to(cihaz)).cpu().numpy()
