# Protocolo experimental congelado

Autor: Sebastián Rojas · srojas.hz@gmail.com · 2026.

Este documento describe el comportamiento existente. Las constantes, operaciones
y decisiones del experimento siguen en sus fuentes originales; la organización
S01–S25 no introduce una configuración experimental nueva.

## Datos y particiones

Se utilizan 126 nubes anotadas de Pheno4D: 49 de maíz y 77 de tomate. Los
originales permanecen en `data/raw/Pheno4D/`. Maíz usa Leaf Collar, columna de
índice 3; tomate usa la etiqueta de esa misma columna. El loader elimina etiqueta
original 0, remapea 1 a tallo/clase 0 y etiquetas >= 2 a hoja/clase 1.

| Especie | Train | Validation | Test |
|---|---|---|---|
| maize | Maize01, Maize02, Maize04, Maize05 | Maize03 | Maize06, Maize07 |
| tomato | Tomato01, Tomato02, Tomato03, Tomato04 | Tomato05 | Tomato06, Tomato07 |

`configs/splits.yaml` es la partición vigente. Los seis manifiestos tienen,
respectivamente, 28/7/14 nubes de maíz y 44/11/22 de tomate. El split por planta
evita compartir observaciones temporales de una planta entre particiones.

## Muestreo y control

| Elemento | Valor congelado |
|---|---|
| Samplers factoriales | Random, Voxel Grid/UVS, FPS |
| Cardinalidades | 16,384; 8,192; 4,096; 2,048; 1,024 |
| Sampling seed | 42 |
| Control | Random sin reemplazo, 48,000 puntos, seed 42 |
| Nubes factoriales | 1,890 |
| Nubes de control | 126 |

Los samplers conservan puntos originales y sus etiquetas. Voxel selecciona el
punto original más cercano al centro de cada voxel y ajusta cardinalidad mediante
FPS. FPS externo comienza en índice 0. El FPS interno de PointNet++ tiene su
propia selección inicial aleatoria; ambas implementaciones permanecen separadas.

El identificador `control_48000` representa el control práctico de alta
resolución. La referencia geométrica es la nube original post-ground, independiente
del control. Chamfer y H99 finales cubren las 30 configuraciones factoriales;
las columnas geométricas del control permanecen vacías.

## Entrenamiento y evaluación

| Elemento | Valor congelado |
|---|---|
| Modelo | PointNet++ SSG, implementación conservada en `src/models/pointnet2/` |
| Entrada / clases | XYZ / tallo y hoja |
| Training seeds | 0, 1, 2, 3, 4 |
| Épocas | 50 |
| Batch size | 2 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Scheduler | StepLR, step_size=10, gamma=1.0 |
| Augmentation | Desactivada |
| Checkpoint | Mejor mIoU de validación |
| Eval seed | 12345 |
| Corridas | 150 factoriales + 10 de control |

El dataset mantiene la normalización y conversión a tensores existentes. El
orden de archivos, las operaciones de dtype, la generación de batches y el
guardado/restauración de RNG durante validación forman parte del comportamiento
reproducible. El dataset consume los NPZ de cada carpeta; no añadir archivos a
esas carpetas durante una reconstrucción.

Los checkpoints formales se guardan bajo
`data/results/training/{species}/{method}/{target}/seed_{seed}/no_aug/`.
El directorio de NPZ del control utiliza
`data/processed/{species}/control_48000/{split}/`, sin nivel `48000`.

## Métricas finales

- Segmentación: mIoU, IoU stem, IoU leaf; media y desviación estándar sobre cinco
  semillas. Se conservan las reglas de matriz de confusión y clases vacías.
- Chamfer: L2 euclidiana no cuadrática, media simétrica multiplicada por 0.5.
  Para subconjuntos de puntos originales, el término reducida→original es cero.
- H99: percentil 99 de las distancias original→reducida, método NumPy `linear`.
- Recursos: sampling, entrenamiento, inferencia y memoria GPU según los registros
  existentes. La evaluación funcional y el benchmark dedicado tienen mediciones
  diferentes; el tradeoff utiliza el benchmark dedicado para inferencia.

El benchmark utiliza el primer batch completo de test, batch 2, cinco warmups y
veinte repeticiones. Mide forward con sincronización CUDA. La semilla 12345 se
fija en el lugar original del proceso; no reiniciarla por configuración.

## Estadística y entrega

Friedman por especie×resolución utiliza los tres samplers y las cinco semillas.
El análisis global utiliza 50 bloques especie×resolución×semilla. El control
queda excluido de Friedman.

El post-hoc local analiza los Friedman significativos. Los post-hoc entre
samplers conservan Wilcoxon con `method="auto"`; las comparaciones contra control
conservan Wilcoxon exacto bilateral, sus comprobaciones y Holm sobre las 15
comparaciones de cada especie. Se mantienen Kendall's W y rank-biserial
correlation. La ausencia de significación no se interpreta como equivalencia.

La entrega incluye cuatro tablas CSV/TEX, curvas técnicas de tiempo frente al
número de puntos y cinco figuras de publicación PNG/PDF. El flujo de integración
es S16→S17→S21→S22→S23; S18–S20 producen las fuentes estadísticas de S22.
