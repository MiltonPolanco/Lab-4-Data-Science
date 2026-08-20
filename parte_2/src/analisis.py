"""Resúmenes y figuras del avance de la parte 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preparacion import ARCHIVO_DATOS, DIR_FIGURES, DIR_PROCESSED


PREDICTORES_NUMERICOS = [
    "B02",
    "B03",
    "B08",
    "B11",
    "B12",
    "ndwi",
    "x_utm",
    "y_utm",
    "anio",
    "dia_ano_sen",
    "dia_ano_cos",
]
PREDICTORES_CATEGORICOS = ["lago"]
EXCLUIDAS_POR_FUGA = [
    "B04",
    "B05",
    "ndci",
    "cianobacteria_mg_m3",
    "ndvi",
]


def cargar_datos(path: Path = ARCHIVO_DATOS) -> pd.DataFrame:
    datos = pd.read_parquet(path)
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    return datos


def construir_resumenes(datos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    por_lago = (
        datos.groupby("lago", as_index=False)
        .agg(
            observaciones=("alta_cianobacteria", "size"),
            alta=("alta_cianobacteria", "sum"),
            cianobacteria_promedio=("cianobacteria_mg_m3", "mean"),
            cianobacteria_mediana=("cianobacteria_mg_m3", "median"),
        )
        .assign(
            baja=lambda tabla: tabla["observaciones"] - tabla["alta"],
            alta_pct=lambda tabla: 100 * tabla["alta"] / tabla["observaciones"],
        )
    )
    por_fecha = (
        datos.groupby(["lago", "fecha"], as_index=False)
        .agg(
            observaciones=("alta_cianobacteria", "size"),
            alta=("alta_cianobacteria", "sum"),
            cianobacteria_promedio=("cianobacteria_mg_m3", "mean"),
            cianobacteria_mediana=("cianobacteria_mg_m3", "median"),
        )
        .assign(
            baja=lambda tabla: tabla["observaciones"] - tabla["alta"],
            alta_pct=lambda tabla: 100 * tabla["alta"] / tabla["observaciones"],
        )
    )
    descriptivas = datos[
        [
            "B02",
            "B03",
            "B04",
            "B05",
            "B08",
            "B11",
            "B12",
            "ndvi",
            "ndwi",
            "cianobacteria_mg_m3",
        ]
    ].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    por_lago.to_csv(
        DIR_PROCESSED / "distribucion_clase_lago.csv",
        index=False,
        encoding="utf-8-sig",
    )
    por_fecha.to_csv(
        DIR_PROCESSED / "distribucion_clase_fecha.csv",
        index=False,
        encoding="utf-8-sig",
    )
    descriptivas.to_csv(
        DIR_PROCESSED / "estadisticas_descriptivas.csv", encoding="utf-8-sig"
    )
    return {"por_lago": por_lago, "por_fecha": por_fecha, "descriptivas": descriptivas}


def _guardar(figura: plt.Figure, nombre: str) -> Path:
    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    salida = DIR_FIGURES / nombre
    figura.savefig(salida, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figura)
    return salida


def generar_figuras(datos: pd.DataFrame) -> list[Path]:
    colores = {"Amatitlán": "#d95f02", "Atitlán": "#1b9e77"}
    muestra = datos.sample(min(120_000, len(datos)), random_state=3084)
    salidas: list[Path] = []

    limite = float(datos["cianobacteria_mg_m3"].quantile(0.99))
    figura, eje = plt.subplots(figsize=(9, 4.8))
    for lago, grupo in muestra.groupby("lago"):
        eje.hist(
            grupo["cianobacteria_mg_m3"].clip(upper=limite),
            bins=55,
            alpha=0.58,
            density=True,
            color=colores[lago],
            label=lago,
        )
    eje.axvline(12, color="#b2182b", linestyle="--", linewidth=2, label="Umbral: 12 mg/m³")
    eje.set_xlabel("Clorofila-a estimada (mg/m³; valores sobre P99 recortados)")
    eje.set_ylabel("Densidad")
    eje.set_title("Distribución del indicador de cianobacteria")
    eje.legend(frameon=False)
    eje.grid(axis="y", alpha=0.2)
    salidas.append(_guardar(figura, "distribucion_cianobacteria.png"))

    por_lago = (
        datos.groupby("lago")["alta_cianobacteria"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        * 100
    )
    figura, eje = plt.subplots(figsize=(7.5, 4.8))
    por_lago.plot(
        kind="bar",
        stacked=True,
        color=["#b8c9d4", "#b2182b"],
        ax=eje,
        width=0.62,
    )
    eje.set_xlabel("")
    eje.set_ylabel("Porcentaje de observaciones")
    eje.set_title("Distribución de la variable respuesta por lago")
    eje.legend(["Baja o ausente (0)", "Alta (1)"], frameon=False, loc="upper right")
    eje.tick_params(axis="x", rotation=0)
    eje.set_ylim(0, 100)
    eje.grid(axis="y", alpha=0.2)
    salidas.append(_guardar(figura, "clase_por_lago.png"))

    por_fecha = (
        datos.groupby(["lago", "fecha"], as_index=False)["alta_cianobacteria"]
        .mean()
        .assign(alta_pct=lambda tabla: 100 * tabla["alta_cianobacteria"])
    )
    figura, ejes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=False)
    for eje, (lago, grupo) in zip(ejes, por_fecha.groupby("lago", sort=False)):
        grupo = grupo.sort_values("fecha")
        eje.plot(
            grupo["fecha"],
            grupo["alta_pct"],
            color=colores[lago],
            marker="o",
            linewidth=2,
        )
        eje.set_title(f"Lago {lago}", loc="left")
        eje.set_ylabel("Clase alta (%)")
        eje.set_ylim(-2, 102)
        eje.grid(alpha=0.25)
        eje.tick_params(axis="x", rotation=30)
    figura.suptitle("Proporción de píxeles con alta presencia por fecha")
    figura.tight_layout()
    salidas.append(_guardar(figura, "alta_por_fecha.png"))

    correlacion = muestra[PREDICTORES_NUMERICOS].corr()
    figura, eje = plt.subplots(figsize=(8.5, 7))
    imagen = eje.imshow(correlacion, cmap="RdBu_r", vmin=-1, vmax=1)
    eje.set_xticks(range(len(correlacion.columns)), correlacion.columns, rotation=45, ha="right")
    eje.set_yticks(range(len(correlacion.index)), correlacion.index)
    for i in range(len(correlacion.index)):
        for j in range(len(correlacion.columns)):
            valor = correlacion.iloc[i, j]
            eje.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=7)
    figura.colorbar(imagen, ax=eje, shrink=0.82, label="Correlación de Pearson")
    eje.set_title("Correlación entre predictores numéricos propuestos")
    figura.tight_layout()
    salidas.append(_guardar(figura, "correlacion_predictores.png"))

    figura, ejes = plt.subplots(1, 2, figsize=(11, 5.2))
    for eje, (lago, grupo) in zip(ejes, muestra.groupby("lago", sort=False)):
        baja = grupo[grupo["alta_cianobacteria"] == 0]
        alta = grupo[grupo["alta_cianobacteria"] == 1]
        eje.scatter(baja["longitud"], baja["latitud"], s=1, alpha=0.12, color="#7f9bad")
        eje.scatter(alta["longitud"], alta["latitud"], s=2, alpha=0.45, color="#b2182b")
        eje.set_title(lago)
        eje.set_xlabel("Longitud")
        eje.set_ylabel("Latitud")
        eje.set_aspect("equal", adjustable="datalim")
        eje.grid(alpha=0.15)
    figura.suptitle("Distribución espacial de la clase en una muestra de píxeles")
    figura.tight_layout()
    salidas.append(_guardar(figura, "distribucion_espacial_clase.png"))
    return salidas


if __name__ == "__main__":
    conjunto = cargar_datos()
    construir_resumenes(conjunto)
    for archivo in generar_figuras(conjunto):
        print(archivo)

