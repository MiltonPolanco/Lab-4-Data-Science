# Laboratorio 4 - Datos geoespaciales

## Integrantes

- Milton Polanco
- Osman de León

## Avance

Esta entrega contiene los ejercicios 1 al 4 del laboratorio:

1. conexión con una API de Sentinel-2;
2. consulta de las bandas necesarias para los dos lagos;
3. cálculo de NDVI, NDWI y del indicador de cianobacteria;
4. promedio por fecha, gráfica temporal e identificación de los valores más altos.

El procedimiento está desarrollado en `notebooks/laboratorio_4_avance.ipynb`. El único archivo auxiliar es `src/rasters.py`, que se encarga de leer y ajustar las bandas remotas al área de cada lago.

## Datos

La disponibilidad de Sentinel-2 L2A se comprueba con la API openEO de Copernicus Data Space. Las imágenes se consultan en el catálogo STAC de Earth Search y se leen como COG, por lo que no se descargan escenas completas. Los CSV y el JSON incluidos son resultados pequeños que permiten revisar el avance sin repetir todas las consultas.

## Ejecución

```bash
pip install -r requirements.txt
jupyter notebook notebooks/laboratorio_4_avance.ipynb
```

El notebook reutiliza `data/processed/resumen_temporal.csv` si ya existe. Para repetir la consulta de las imágenes se debe cambiar `VOLVER_A_DESCARGAR` a `True`.

Los ejercicios 5 al 8 todavía no forman parte de esta entrega.
