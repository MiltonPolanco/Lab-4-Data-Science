# Laboratorio 4 - Datos geoespaciales

## Integrantes

- Milton Polanco
- Osman de León

## Parte 1

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

## Parte 2 - avance

El directorio `parte_2` contiene los ejercicios 1, 2 y 3 solicitados para el
avance del 20 de agosto de 2026:

1. construcción y limpieza del conjunto de datos por píxel;
2. definición y análisis de la variable respuesta binaria;
3. selección de variables predictoras sin fuga de información.

El conjunto final tiene 1,781,186 observaciones válidas y está guardado como
`parte_2/data/processed/dataset_pixeles.parquet`. El notebook ejecutado se
encuentra en `parte_2/notebooks/laboratorio_4_parte_2_avance.ipynb` y el informe
en `parte_2/reports/avance_laboratorio_4_parte_2.pdf`.

Para reconstruir los datos y el análisis:

```bash
python -m parte_2.src.preparacion
python -m parte_2.src.analisis
jupyter notebook parte_2/notebooks/laboratorio_4_parte_2_avance.ipynb
```

Los archivos intermedios se guardan por lago y fecha para que la descarga pueda
reanudarse. No se incluyen en Git; el archivo Parquet final y los resúmenes sí
forman parte del avance.
