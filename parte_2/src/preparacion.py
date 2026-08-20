"""Construcción del conjunto de datos por píxel para Machine Learning.

Cada archivo intermedio corresponde a una combinación lago-fecha. Esto permite
reanudar la descarga sin repetir observaciones que ya fueron procesadas.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
import shutil
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from rasterio.transform import from_origin
from rasterio.warp import transform as transformar_coordenadas
from rasterio.warp import transform_bounds
from scipy import ndimage


RAIZ = Path(__file__).resolve().parents[2]
DIR_PARTE2 = RAIZ / "parte_2"
DIR_INTERIM = DIR_PARTE2 / "data" / "interim"
DIR_PROCESSED = DIR_PARTE2 / "data" / "processed"
DIR_FIGURES = DIR_PARTE2 / "figures"

ARCHIVO_DATOS = DIR_PROCESSED / "dataset_pixeles.parquet"
ARCHIVO_OBSERVACIONES = DIR_PROCESSED / "resumen_observaciones.csv"
ARCHIVO_VARIABLES = DIR_PROCESSED / "resumen_variables.csv"

URL_STAC = "https://earth-search.aws.element84.com/v1/search"
COLECCION_STAC = "sentinel-2-l2a"
CRS_DESTINO = "EPSG:32615"
RESOLUCION = 20
UMBRAL_ALTO = 12.0
AREA_MINIMA_COMPONENTE_KM2 = 0.1

LAGOS = {
    "Amatitlán": {
        "west": -90.638065,
        "east": -90.512924,
        "south": 14.412347,
        "north": 14.493799,
    },
    "Atitlán": {
        "west": -91.326256,
        "east": -91.071510,
        "south": 14.594800,
        "north": 14.750979,
    },
}

FECHAS = pd.DataFrame(
    [
        ("Amatitlán", "2025-01-28", 0.06, "Sentinel-2B"),
        ("Amatitlán", "2025-04-15", 0.09, "Sentinel-2A"),
        ("Amatitlán", "2025-04-28", 1.03, "Sentinel-2B"),
        ("Amatitlán", "2025-11-24", 0.50, "Sentinel-2B"),
        ("Amatitlán", "2026-01-08", 0.77, "Sentinel-2C"),
        ("Amatitlán", "2026-02-02", 0.39, "Sentinel-2B"),
        ("Amatitlán", "2026-02-07", 0.02, "Sentinel-2C"),
        ("Amatitlán", "2026-03-29", 0.01, "Sentinel-2C"),
        ("Amatitlán", "2026-04-13", 0.09, "Sentinel-2B"),
        ("Amatitlán", "2026-04-28", 4.96, "Sentinel-2C"),
        ("Amatitlán", "2026-06-19", 13.00, "Sentinel-2A"),
        ("Atitlán", "2025-01-18", 0.02, "Sentinel-2B"),
        ("Atitlán", "2025-04-13", 0.54, "Sentinel-2C"),
        ("Atitlán", "2025-05-13", 4.37, "Sentinel-2C"),
        ("Atitlán", "2025-07-17", 3.57, "Sentinel-2A"),
        ("Atitlán", "2025-11-21", 3.15, "Sentinel-2A"),
        ("Atitlán", "2025-12-29", 3.17, "Sentinel-2C"),
        ("Atitlán", "2026-02-12", 0.04, "Sentinel-2B"),
        ("Atitlán", "2026-03-24", 3.17, "Sentinel-2B"),
        ("Atitlán", "2026-04-13", 0.01, "Sentinel-2B"),
        ("Atitlán", "2026-04-28", 4.96, "Sentinel-2C"),
        ("Atitlán", "2026-07-22", 4.02, "Sentinel-2B"),
    ],
    columns=["lago", "fecha", "nubosidad_oficial_pct", "satelite"],
)
FECHAS["fecha"] = pd.to_datetime(FECHAS["fecha"])


def asegurar_directorios() -> None:
    for directorio in (DIR_INTERIM, DIR_PROCESSED, DIR_FIGURES):
        directorio.mkdir(parents=True, exist_ok=True)


def _slug(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c)).lower()


def _archivo_observacion(lago: str, fecha: str) -> Path:
    return DIR_INTERIM / f"{_slug(lago)}_{fecha}.parquet"


def buscar_escenas(
    bbox: dict, fecha: str, nubosidad: float, satelite: str
) -> list[dict]:
    respuesta = requests.post(
        URL_STAC,
        json={
            "collections": [COLECCION_STAC],
            "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
            "datetime": f"{fecha}T00:00:00Z/{fecha}T23:59:59Z",
            "limit": 20,
        },
        timeout=45,
    )
    respuesta.raise_for_status()
    escenas = [
        escena
        for escena in respuesta.json().get("features", [])
        if escena["properties"].get("platform", "").lower() == satelite.lower()
    ]
    if not escenas:
        raise LookupError(f"No se encontró {satelite} para {fecha}.")
    escenas.sort(
        key=lambda escena: abs(
            float(escena["properties"].get("eo:cloud_cover", 100)) - nubosidad
        )
    )
    return escenas


def _malla_destino(bbox: dict):
    limites = transform_bounds(
        "EPSG:4326",
        CRS_DESTINO,
        bbox["west"],
        bbox["south"],
        bbox["east"],
        bbox["north"],
        densify_pts=21,
    )
    ancho = ceil((limites[2] - limites[0]) / RESOLUCION)
    alto = ceil((limites[3] - limites[1]) / RESOLUCION)
    transformacion = from_origin(limites[0], limites[3], RESOLUCION, RESOLUCION)
    return transformacion, ancho, alto


def _division_segura(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            numerador,
            denominador,
            out=np.full_like(numerador, np.nan, dtype="float32"),
            where=np.abs(denominador) > 1e-8,
        )


def calcular_variables(bandas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calcula índices y una máscara de agua limpia.

    Se conservan los componentes de agua con al menos 0.1 km2. El criterio
    retira manchas aisladas, pero mantiene los dos sectores de Amatitlán que
    aparecen separados por el relleno construido sobre el lago.
    """

    azul = bandas["B02"]
    verde = bandas["B03"]
    rojo = bandas["B04"]
    borde_rojo = bandas["B05"]
    nir = bandas["B08"]
    swir1 = bandas["B11"]
    swir2 = bandas["B12"]
    scl = bandas["SCL"].astype("int16")

    ndvi = _division_segura(nir - rojo, nir + rojo)
    ndwi = _division_segura(verde - nir, verde + nir)
    mndwi = _division_segura(verde - swir1, verde + swir1)
    ndwi_hojas = _division_segura(nir - swir1, nir + swir1)
    awei_sombra = azul + 2.5 * verde - 1.5 * (nir + swir1) - 0.25 * swir2
    awei_sin_sombra = 4 * (verde - swir1) - (0.25 * nir + 2.75 * swir1)
    dbsi = _division_segura(swir1 - verde, swir1 + verde) - ndvi

    agua = (
        (mndwi > 0.42)
        | (ndwi > 0.40)
        | (awei_sin_sombra > 0.1879)
        | (awei_sombra > 0.1112)
        | (ndvi < -0.20)
        | (ndwi_hojas > 1)
    )
    agua &= ~((awei_sin_sombra <= -0.03) | (dbsi > 0))

    cielo_valido = ~np.isin(scl, [0, 3, 8, 9, 10, 11])
    datos_disponibles = np.all(
        np.stack([bandas[nombre] > 0 for nombre in bandas if nombre != "SCL"]),
        axis=0,
    )
    candidatos = agua & cielo_valido & datos_disponibles

    etiquetas, cantidad = ndimage.label(candidatos, structure=np.ones((3, 3)))
    if cantidad:
        tamanos = np.bincount(etiquetas.ravel())
        tamanos[0] = 0
        pixeles_minimos = ceil(
            AREA_MINIMA_COMPONENTE_KM2 * 1_000_000 / RESOLUCION**2
        )
        pixeles_validos = tamanos[etiquetas] >= pixeles_minimos
    else:
        pixeles_validos = candidatos

    ndci = _division_segura(borde_rojo - rojo, borde_rojo + rojo)
    cianobacteria = 826.57 * ndci**3 - 176.43 * ndci**2 + 19 * ndci + 4.071
    pixeles_validos &= np.isfinite(cianobacteria)
    pixeles_validos &= (cianobacteria >= 0) & (cianobacteria <= 500)

    return {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndci": ndci,
        "cianobacteria_mg_m3": cianobacteria,
        "pixeles_validos": pixeles_validos,
    }


def construir_tabla(
    lago: str,
    fecha: str,
    satelite: str,
    bandas: dict[str, np.ndarray],
    variables: dict[str, np.ndarray],
    bbox: dict,
) -> pd.DataFrame:
    mascara = variables["pixeles_validos"]
    filas, columnas = np.where(mascara)
    transformacion, _, _ = _malla_destino(bbox)
    x_utm = transformacion.c + (columnas + 0.5) * transformacion.a
    y_utm = transformacion.f + (filas + 0.5) * transformacion.e
    longitudes, latitudes = transformar_coordenadas(
        CRS_DESTINO, "EPSG:4326", x_utm.tolist(), y_utm.tolist()
    )

    fecha_dt = pd.Timestamp(fecha)
    dia_ano = fecha_dt.dayofyear
    tabla = pd.DataFrame(
        {
            "lago": lago,
            "fecha": fecha_dt,
            "satelite": satelite,
            "fila": filas.astype("int32"),
            "columna": columnas.astype("int32"),
            "x_utm": x_utm.astype("float32"),
            "y_utm": y_utm.astype("float32"),
            "longitud": np.asarray(longitudes, dtype="float32"),
            "latitud": np.asarray(latitudes, dtype="float32"),
            "anio": np.full(filas.size, fecha_dt.year, dtype="int16"),
            "mes": np.full(filas.size, fecha_dt.month, dtype="int8"),
            "dia_ano_sen": np.full(
                filas.size, np.sin(2 * np.pi * dia_ano / 365.25), dtype="float32"
            ),
            "dia_ano_cos": np.full(
                filas.size, np.cos(2 * np.pi * dia_ano / 365.25), dtype="float32"
            ),
        }
    )
    for banda in ("B02", "B03", "B04", "B05", "B08", "B11", "B12"):
        tabla[banda] = bandas[banda][mascara].astype("float32")
    for nombre in ("ndvi", "ndwi", "ndci", "cianobacteria_mg_m3"):
        tabla[nombre] = variables[nombre][mascara].astype("float32")
    tabla["alta_cianobacteria"] = (
        tabla["cianobacteria_mg_m3"] >= UMBRAL_ALTO
    ).astype("int8")
    return tabla


def procesar_observacion(fila: pd.Series, sobrescribir: bool = False) -> dict:
    from src.rasters import leer_bandas

    lago = fila["lago"]
    fecha = fila["fecha"].strftime("%Y-%m-%d")
    salida = _archivo_observacion(lago, fecha)
    if salida.exists() and not sobrescribir:
        tabla = pd.read_parquet(salida, columns=["alta_cianobacteria"])
        return {
            "lago": lago,
            "fecha": fecha,
            "satelite": fila["satelite"],
            "estado": "reutilizado",
            "observaciones": len(tabla),
            "alta_cianobacteria": int(tabla["alta_cianobacteria"].sum()),
        }

    bbox = LAGOS[lago]
    escenas = buscar_escenas(
        bbox,
        fecha,
        float(fila["nubosidad_oficial_pct"]),
        fila["satelite"],
    )
    bandas = leer_bandas(escenas, bbox)
    variables = calcular_variables(bandas)
    tabla = construir_tabla(
        lago, fecha, fila["satelite"], bandas, variables, bbox
    )
    tabla.to_parquet(salida, index=False, compression="zstd")
    return {
        "lago": lago,
        "fecha": fecha,
        "satelite": fila["satelite"],
        "estado": "procesado",
        "observaciones": len(tabla),
        "alta_cianobacteria": int(tabla["alta_cianobacteria"].sum()),
    }


def unir_archivos(archivos: list[Path], salida: Path = ARCHIVO_DATOS) -> Path:
    escritor = None
    temporal = salida.with_suffix(".tmp.parquet")
    try:
        for archivo in archivos:
            tabla = pq.read_table(archivo)
            if escritor is None:
                escritor = pq.ParquetWriter(
                    temporal, tabla.schema, compression="zstd", use_dictionary=True
                )
            escritor.write_table(tabla)
    finally:
        if escritor is not None:
            escritor.close()
    if escritor is None:
        raise RuntimeError("No se generaron archivos con observaciones válidas.")
    shutil.move(temporal, salida)
    return salida


def resumen_variables(datos: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in datos.columns:
        filas.append(
            {
                "variable": columna,
                "tipo": str(datos[columna].dtype),
                "faltantes_pct": 100 * datos[columna].isna().mean(),
                "valores_unicos": int(datos[columna].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(filas)


def ejecutar_preparacion(sobrescribir: bool = False) -> pd.DataFrame:
    asegurar_directorios()
    resultados = []
    total = len(FECHAS)
    for numero, fila in FECHAS.iterrows():
        fecha = fila["fecha"].strftime("%Y-%m-%d")
        print(f"[{numero + 1:02d}/{total}] {fila['lago']} - {fecha}", flush=True)
        try:
            resultado = procesar_observacion(fila, sobrescribir=sobrescribir)
        except LookupError as error:
            print(f"  Aviso: {error}", flush=True)
            resultado = {
                "lago": fila["lago"],
                "fecha": fecha,
                "satelite": fila["satelite"],
                "estado": "sin escena",
                "observaciones": 0,
                "alta_cianobacteria": 0,
            }
        resultados.append(resultado)

    resumen = pd.DataFrame(resultados)
    resumen["alta_pct"] = np.where(
        resumen["observaciones"] > 0,
        100 * resumen["alta_cianobacteria"] / resumen["observaciones"],
        np.nan,
    )
    resumen.to_csv(ARCHIVO_OBSERVACIONES, index=False, encoding="utf-8-sig")

    archivos = [
        _archivo_observacion(fila["lago"], fila["fecha"].strftime("%Y-%m-%d"))
        for _, fila in FECHAS.iterrows()
    ]
    unir_archivos([archivo for archivo in archivos if archivo.exists()])

    # La lectura completa se hace una vez para el inventario solicitado.
    datos = pd.read_parquet(ARCHIVO_DATOS)
    resumen_variables(datos).to_csv(
        ARCHIVO_VARIABLES, index=False, encoding="utf-8-sig"
    )
    return resumen


if __name__ == "__main__":
    print(ejecutar_preparacion(sobrescribir=False).to_string(index=False))
