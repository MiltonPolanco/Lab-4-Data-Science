from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds


ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}

RESOLUCION = 20
CRS_DESTINO = "EPSG:32615"


def _malla_destino(bbox):
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
    transformacion = from_origin(
        limites[0], limites[3], RESOLUCION, RESOLUCION
    )
    return transformacion, ancho, alto


def _leer_banda(escenas, asset, bbox):
    transformacion, ancho, alto = _malla_destino(bbox)
    mosaico = np.zeros((alto, ancho), dtype="float32")
    remuestreo = Resampling.nearest if asset == "scl" else Resampling.bilinear

    opciones_gdal = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
    }

    with rasterio.Env(**opciones_gdal):
        for escena in escenas:
            recurso = escena.get("assets", {}).get(asset)
            if recurso is None:
                continue

            with rasterio.open(recurso["href"]) as archivo:
                with WarpedVRT(
                    archivo,
                    crs=CRS_DESTINO,
                    transform=transformacion,
                    width=ancho,
                    height=alto,
                    src_nodata=0,
                    nodata=0,
                    resampling=remuestreo,
                ) as raster:
                    tesela = raster.read(1, out_dtype="float32")

            completar = (mosaico == 0) & (tesela > 0)
            mosaico[completar] = tesela[completar]

    return mosaico


def leer_bandas(escenas, bbox):
    """Lee las bandas necesarias dentro del rectángulo de un lago."""
    bandas = {}

    # Cada banda es un archivo independiente, por eso se pueden leer varias a la vez.
    with ThreadPoolExecutor(max_workers=5) as ejecutor:
        tareas = {
            ejecutor.submit(_leer_banda, escenas, asset, bbox): banda
            for banda, asset in ASSETS.items()
        }
        for tarea in as_completed(tareas):
            bandas[tareas[tarea]] = tarea.result()

    for nombre in bandas:
        if nombre != "SCL":
            bandas[nombre] = bandas[nombre] / 10000.0

    return bandas
