import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app

ORNEKLER = json.loads((Path(__file__).parent / "ornek_motorlar.json").read_text())


@pytest.fixture(scope="module")
def istemci():
    with TestClient(app) as c:
        yield c


def istek(ad, **ek):
    return {"olcumler": ORNEKLER[ad]["olcumler"], **ek}


def test_saglik(istemci):
    s = istemci.get("/saglik").json()
    assert s["durum"] == "calisiyor" and len(s["sensorler"]) == 14


def test_bozulmak_uzere_olan_motor(istemci):
    s = istemci.post("/motor", json=istek("bozulmak_uzere")).json()
    assert s["durum"] == "hemen bakım"
    assert abs(s["kalan_omur"] - ORNEKLER["bozulmak_uzere"]["gercek_kalan_omur"]) < 15


def test_saglikli_motor(istemci):
    s = istemci.post("/motor", json=istek("saglikli")).json()
    assert s["durum"] == "sağlıklı"
    assert s["kalan_omur"] > 80


def test_tahmin_zamanla_azalir(istemci):
    # Aynı motorun kaydının ilk yarısında kalan ömür, tamamına göre daha uzun tahmin edilmeli
    olcumler = ORNEKLER["bozulmak_uzere"]["olcumler"]
    yarim = istemci.post("/motor", json={"olcumler": olcumler[: len(olcumler) // 2]}).json()["kalan_omur"]
    tam = istemci.post("/motor", json={"olcumler": olcumler}).json()["kalan_omur"]
    assert yarim > tam


def test_eksik_sensor(istemci):
    olcumler = [dict(o) for o in ORNEKLER["saglikli"]["olcumler"]]
    del olcumler[-1]["s11"]
    assert istemci.post("/motor", json={"olcumler": olcumler}).status_code == 422


def test_bos_liste(istemci):
    assert istemci.post("/motor", json={"olcumler": []}).status_code == 422
