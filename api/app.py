"""Turbofan motor kalan ömür tahmini ve bakım önerisi servisi.

Çalıştırma:  uvicorn api.app:app
Dokümantasyon: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from src.cikarim import OmurTahmincisi
from src.veri import SENSORLER

Olcum = create_model("Olcum", **{s: (float, ...) for s in SENSORLER})


class Istek(BaseModel):
    olcumler: Annotated[list[Olcum], Field(min_length=1, max_length=1000,
                                           description="Döngü sırasıyla sensör ölçümleri; son eleman en güncel")]
    ilk_dongu: int = Field(1, ge=1, description="Listedeki ilk ölçümün döngü numarası")


class Cevap(BaseModel):
    kalan_omur: float
    durum: Literal["sağlıklı", "izlenmeli", "bakım planla", "hemen bakım"]
    bakim_esigi: int
    son_10_dongu_tahmini: list[float]


tahminci = {}


@asynccontextmanager
async def lifespan(app):
    tahminci["t"] = OmurTahmincisi()
    yield
    tahminci.clear()


app = FastAPI(title="Kestirimci Bakım", lifespan=lifespan)


@app.get("/saglik")
def saglik():
    return {"durum": "calisiyor", "bakim_esigi": tahminci["t"].esik, "sensorler": SENSORLER}


@app.post("/motor", response_model=Cevap)
def motor(istek: Istek):
    return tahminci["t"].tahmin_et([o.model_dump() for o in istek.olcumler], istek.ilk_dongu)
