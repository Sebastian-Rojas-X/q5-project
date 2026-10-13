# Bitácora Checkpoint Q5 — 2026-09-02

> **Registro histórico.** Esta bitácora reúne checkpoints de septiembre de 2026. Durante la limpieza final se retiraron referencias a herramientas descartadas e instrucciones de pruebas preliminares. Los resultados numéricos y los respaldos registrados se conservan; las instrucciones vigentes de reproducción están en `README.md`.

## 1. Contexto del proyecto

Este documento registra estados históricos del trabajo correspondiente a la **Pregunta 5 (Q5)** del proyecto **2026 DSI Examen Calificación**.

El objetivo experimental es evaluar métodos de **submuestreo de nubes de puntos 3D** y estudiar su efecto sobre:

- desempeño de segmentación semántica de órganos vegetales;
- fidelidad geométrica;
- costo computacional.

El dataset principal es **Pheno4D**, trabajando con maíz y tomate.

Este checkpoint corresponde al estado del proyecto **antes de comenzar la integración de PointNet++ para segmentación semántica**.

---

## 2. Estado general alcanzado

Hasta este punto se completaron y validaron los siguientes bloques:

1. entorno Docker con PyTorch + CUDA;
2. inspección y catalogación de Pheno4D;
3. definición y validación de splits por planta;
4. loader de Pheno4D;
5. remapeo de clases a tallo y hoja;
6. visualización 3D con Open3D;
7. implementación de Random Sampling;
8. implementación de Voxel Grid / UVS;
9. implementación de Farthest Point Sampling (FPS);
10. validación numérica y visual de los tres samplers;
11. precomputación offline de todas las nubes reducidas;
12. cálculo de Chamfer Distance para todas las nubes;
13. generación del resumen agregado de Chamfer;
14. backup del repositorio en GitHub;
16. backup local de la imagen Docker;
17. backup comprimido del proyecto completo con datos y resultados;
18. generación de checksums SHA-256 para verificar integridad.

---

## 3. Hardware y software principal

### Hardware

- CPU: Intel Core i9-14900HX
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM aproximada: 8 GB
- RAM: 32 GB

### Sistema

- Ubuntu 26
- Docker 29.7.2
- Git 2.53.0

### Docker

Imagen principal:

```text
pheno-q5:pytorch
```

Imagen base:

```text
nvidia/cuda:13.0.2-runtime-ubuntu24.04
```

Python:

```text
Python 3.12.3
```

PyTorch:

```text
2.13.0+cu130
```

CUDA runtime dentro del contenedor:

```text
13.0
```

La GPU fue validada correctamente desde PyTorch mediante operaciones reales en CUDA.

Dependencias relevantes ya instaladas:

- PyTorch
- PyYAML
- NumPy
- SciPy
- Matplotlib
- Open3D 0.19.0

---

## 4. Repositorio Git

Ruta local:

```text
/home/seba/q5-project
```

Rama principal:

```text
main
```

Repositorio remoto:

```text
https://github.com/Sebastian-Rojas-X/q5-project
```

El repositorio ya fue subido correctamente a GitHub.


## 6. Dataset Pheno4D

Dataset descargado y almacenado dentro del proyecto.

Archivo original:

```text
data/raw/Pheno4D.zip
```

Dataset extraído:

```text
data/raw/Pheno4D/
```

Plantas:

```text
Maize01 ... Maize07
Tomato01 ... Tomato07
```

Número total de archivos catalogados:

```text
223
```

Nubes etiquetadas:

```text
Maize: 49
Tomato: 77
Total: 126
```

---

## 7. Formato de anotaciones

### Maíz

Las nubes etiquetadas contienen 5 columnas:

```text
X Y Z LeafCollarLabel LeafTipLabel
```

Para el benchmark de órganos se decidió utilizar **Leaf Collar annotation**, porque permite distinguir explícitamente tallo y hojas.

### Tomate

Las nubes etiquetadas contienen 4 columnas:

```text
X Y Z label
```

---

## 8. Decisión sobre clases

Para Q5 se decidió trabajar únicamente con órganos vegetales:

```text
stem
leaf
```

El suelo se elimina antes de cualquier submuestreo o cálculo de métricas.

Remapeo semántico:

```text
0 -> stem
1 -> leaf
```

En los archivos originales de Pheno4D:

- etiqueta original `0`: suelo → eliminada;
- etiqueta original `1`: tallo → clase semántica `0`;
- etiquetas originales `>= 2`: hojas individuales → clase semántica `1`.

---

## 9. Validaciones del loader

Ejemplo maíz:

```text
M01_0313
Puntos después de quitar suelo: 232122
Stem: 75493
Leaf: 156629
```

Ejemplo tomate:

```text
T01_0305
Puntos después de quitar suelo: 144633
Stem: 12389
Leaf: 132244
```

Ejemplo maíz maduro:

```text
M07_0325
Puntos: 1,144,667
Stem: 833,789
Leaf: 310,878
```

Se validó visualmente la asignación de clases mediante Open3D.

---

## 10. Splits experimentales

Splits actualmente codificados:

```yaml
maize:
  train: [Maize01, Maize02, Maize04, Maize05]
  val: [Maize03]
  test: [Maize06, Maize07]

tomato:
  train: [Tomato01, Tomato02, Tomato03, Tomato04]
  val: [Tomato05]
  test: [Tomato06, Tomato07]
```

Conteos:

```text
Maize train: 28
Maize val:    7
Maize test:  14

Tomato train: 44
Tomato val:  11
Tomato test: 22

Total: 126 nubes
```

### Decisión metodológica importante

El split se realiza **a nivel de planta**, no a nivel de fecha ni nube individual, para evitar leakage entre distintas observaciones temporales de la misma planta.

Estos splits permanecen técnicamente provisionales hasta el freeze final del protocolo experimental, pero ya están codificados y validados.

---

## 11. Métodos de submuestreo

Métodos experimentales:

```text
Random Sampling
Voxel Grid / UVS
Farthest Point Sampling (FPS)
```

Targets:

```text
16384
8192
4096
2048
1024
```

Semilla fija de muestreo:

```text
sampling_seed = 42
```

Esta semilla es independiente de las futuras semillas de entrenamiento.

---

## 12. Random Sampling

Características:

- muestreo sin reemplazo;
- semilla fija 42;
- conserva únicamente puntos originales;
- conserva sus etiquetas originales.

Ejemplo validado:

```text
M01_0313
Target: 16384
Stem: 5366
Leaf: 11018
```

---

## 13. Voxel Grid / UVS

La implementación fue corregida para alinearla con la estrategia UVS descrita en el trabajo comparativo de Li et al.

Decisión final:

1. construir una grilla 3D anclada en `min_bound`;
2. obtener voxels ocupados;
3. para cada voxel ocupado:
   - calcular el centro geométrico del voxel;
   - seleccionar el punto original más cercano a ese centro;
4. ajustar el tamaño de voxel hasta obtener un número de puntos cercano al target desde arriba;
5. si el número final excede el target, aplicar FPS determinístico para ajustar exactamente.

Importante:

- no se generan centroides artificiales;
- todos los puntos finales pertenecen a la nube original;
- las etiquetas permanecen asociadas al punto real seleccionado.

---

## 14. Farthest Point Sampling

FPS fue implementado y validado.

Características:

- ejecución CUDA;
- determinista;
- punto inicial fijo en índice 0;
- conserva puntos reales de la nube original;
- no depende de las futuras semillas de entrenamiento.

---

## 15. Validación visual de samplers

Se realizaron inspecciones visuales con Open3D en maíz y tomate, incluyendo plantas maduras.

Se validaron:

- Random;
- Voxel / UVS;
- FPS.

Conclusión visual:

Los tres métodos producen reducciones plausibles y mantienen la geometría global de las plantas.

FPS y Voxel presentan cobertura espacial más regular que Random.

---

## 16. Precomputación offline

Se implementó:

```text
src/pipeline/S04_precompute_downsampling.py
```

Configuración:

```text
2 especies
3 splits
3 métodos
5 targets
126 nubes etiquetadas
```

Número total de archivos procesados:

```text
126 × 3 × 5 = 1890
```

Número de combinaciones:

```text
2 × 3 × 3 × 5 = 90
```

Salida:

```text
data/processed/
```

Estructura:

```text
data/processed/
├── maize/
│   ├── random/
│   ├── voxel/
│   └── fps/
└── tomato/
    ├── random/
    ├── voxel/
    └── fps/
```

Cada método contiene:

```text
target/
└── split/
    └── cloud.npz
```

Cada `.npz` contiene:

```text
points
labels
```

También se genera:

```text
data/processed/metadata.csv
```

---

## 17. Validación final del preprocesamiento

La auditoría final obtuvo:

```text
NPZ encontrados: 1890
Filas metadata: 1890
Configuraciones: 90
Errores encontrados: 0

VALIDACIÓN FINAL
OK
```

Por tanto, el bloque de generación de datos reducidos quedó cerrado.

---

## 18. Chamfer Distance

Se implementó:

```text
src/metrics/chamfer_distance.py
```

y:

```text
src/pipeline/S06_compute_chamfer.py
```

### Definición usada

Chamfer simétrica basada en distancia Euclidiana L2 no elevada al cuadrado:

```text
CD = 0.5 * (
    mean(Original -> Reduced)
    +
    mean(Reduced -> Original)
)
```

### Propiedad especial de este benchmark

Random, UVS y FPS siempre seleccionan puntos reales de la nube original.

Por construcción:

```text
Reduced -> Original = 0
```

Por esta razón se creó una implementación optimizada:

```text
chamfer_distance_subset()
```

que calcula únicamente:

```text
Original -> Reduced
```

y fija:

```text
Reduced -> Original = 0
```

sin construir innecesariamente un KD-Tree sobre la nube original.

La equivalencia fue validada contra la implementación general.

---

## 19. Validación histórica de Chamfer

Caso:

```text
M01_0313
Target: 16384
```

Resultados:

```text
Random
Original -> Reduced: 0.12200041075828151
Chamfer: 0.061000205379140755

Voxel
Original -> Reduced: 0.11796922758779606
Chamfer: 0.05898461379389803

FPS
Original -> Reduced: 0.11138100152357609
Chamfer: 0.055690500761788046
```

Orden de fidelidad geométrica:

```text
FPS > Voxel > Random
```

donde mejor significa menor Chamfer.

---

## 20. Chamfer completo

Se calcularon todas las comparaciones:

```text
1890 / 1890
```

Archivo:

```text
data/results/chamfer_results.csv
```

---

## 21. Resumen agregado de Chamfer

Script:

```text
src/pipeline/S07_summarize_chamfer.py
```

Archivo:

```text
data/results/chamfer_summary.csv
```

Agrupación:

```text
species × method × target
```

Total:

```text
2 × 3 × 5 = 30 filas
```

---

## 22. Resultados agregados de Chamfer

### Maize

```text
FPS
16384 -> 0.132141 ± 0.065149
 8192 -> 0.186761 ± 0.091558
 4096 -> 0.263658 ± 0.130890
 2048 -> 0.373935 ± 0.189390
 1024 -> 0.534683 ± 0.275378

Random
16384 -> 0.154403 ± 0.077247
 8192 -> 0.218728 ± 0.109307
 4096 -> 0.309201 ± 0.155288
 2048 -> 0.439032 ± 0.222704
 1024 -> 0.629955 ± 0.323425

Voxel
16384 -> 0.139620 ± 0.068831
 8192 -> 0.197371 ± 0.096668
 4096 -> 0.278699 ± 0.138018
 2048 -> 0.394885 ± 0.199330
 1024 -> 0.564846 ± 0.291258
```

### Tomato

```text
FPS
16384 -> 0.131917 ± 0.095740
 8192 -> 0.185233 ± 0.133162
 4096 -> 0.260918 ± 0.187010
 2048 -> 0.369623 ± 0.265026
 1024 -> 0.526981 ± 0.379873

Random
16384 -> 0.144730 ± 0.103090
 8192 -> 0.202391 ± 0.143996
 4096 -> 0.285255 ± 0.204218
 2048 -> 0.405786 ± 0.294272
 1024 -> 0.583845 ± 0.428621

Voxel
16384 -> 0.140896 ± 0.102329
 8192 -> 0.198653 ± 0.143152
 4096 -> 0.281243 ± 0.202613
 2048 -> 0.399543 ± 0.287975
 1024 -> 0.572296 ± 0.413693
```

### Tendencia principal

En ambas especies:

```text
FPS < Voxel < Random
```

en términos de Chamfer promedio, por lo que FPS conserva mejor la cobertura geométrica.

También:

```text
16384 -> menor pérdida
1024  -> mayor pérdida
```

La desviación estándar aumenta al disminuir el target, especialmente en tomate.

---

## 23. Timings de sampling

Durante la precomputación se registró:

```text
sampling_seconds
```

Sin embargo, estos valores NO se consideran todavía la medición científica final de tiempo.

Razón:

- first-call overhead;
- caching;
- variabilidad del sistema;
- diferencias entre CPU/GPU;
- posibles efectos de inicialización.

Más adelante debe implementarse un benchmark dedicado con:

- warm-up;
- múltiples repeticiones;
- sincronización CUDA;
- mismo hardware;
- media y desviación estándar.

---

## 24. Backup GitHub

Repositorio remoto:

```text
https://github.com/Sebastian-Rojas-X/q5-project
```

Rama:

```text
main
```


---

## 25. Backup local de Docker

Imagen guardada:

```text
/home/seba/backups/q5/pheno-q5-pytorch.tar
```

Tamaño:

```text
6.3 GB
```

SHA-256:

```text
2fd6c1266d691f55a02d0a938d5fec922224f528c1114adcf1d2ee3ab1da6b00
```

Restauración:

```bash
docker load -i ~/backups/q5/pheno-q5-pytorch.tar
```

Verificación:

```bash
sha256sum -c ~/backups/q5/SHA256SUMS.txt
```

---

## 26. Backup completo del proyecto

Snapshot:

```text
/home/seba/backups/q5/q5-project-2026-09-02.tar.zst
```

Tamaño:

```text
8.8 GB
```

SHA-256:

```text
137a87a72cd2aef4419edcc0c75610551d421aeaa51c178200505543e45724b5
```

Este snapshot contiene:

- repositorio Git completo;
- código;
- Dockerfile;
- configs;
- dataset raw;
- dataset procesado;
- resultados;
- metadata;
- manifests;
- archivos `.npz`;
- historial `.git`.

Se verificó que el archivo comprimido puede listarse correctamente.

También se verificó que contiene exactamente:

```text
1890 archivos .npz
```

dentro de:

```text
data/processed/
```

---

## 27. Checksums de respaldo

Archivo:

```text
/home/seba/backups/q5/SHA256SUMS.txt
```

Contenido actual:

```text
2fd6c1266d691f55a02d0a938d5fec922224f528c1114adcf1d2ee3ab1da6b00  /home/seba/backups/q5/pheno-q5-pytorch.tar
137a87a72cd2aef4419edcc0c75610551d421aeaa51c178200505543e45724b5  /home/seba/backups/q5/q5-project-2026-09-02.tar.zst
```

---

## 28. Estado actual de reproducibilidad

Actualmente existen tres niveles de recuperación:

### GitHub

Permite recuperar:

- código;
- configuraciones;

### Imagen Docker local

Permite recuperar el entorno exacto validado.

### Snapshot local completo

Permite recuperar:

- código;
- datos;
- resultados;
- imagen del estado del proyecto;
- archivos procesados.

---

## 29. Arquitectura de segmentación pendiente

El siguiente gran bloque es integrar una implementación estable de:

```text
PointNet++
```

para segmentación semántica.

La arquitectura debe permanecer fija entre configuraciones experimentales.

PointNet++ interno puede usar FPS como parte de su arquitectura, pero esto NO debe confundirse con el método experimental de downsampling.

---

## 30. Diseño experimental de Q5

Factores principales:

```text
Species:
- maize
- tomato

Sampling methods:
- random
- voxel
- fps

Targets:
- 16384
- 8192
- 4096
- 2048
- 1024

Training seeds:
- 5 semillas
```

Número de configuraciones con downsampling:

```text
3 × 5 × 2 × 5 = 150 runs
```

Controles:

```text
10 runs
```

Total esperado:

```text
160 runs
```

La definición operativa exacta del control todavía debe congelarse al seleccionar la implementación final de PointNet++.

---

## 31. Métricas finales previstas

### Segmentación

- mIoU;
- IoU stem;
- IoU leaf;
- media;
- desviación estándar entre 5 seeds.

### Fidelidad geométrica

- Chamfer Distance;
- Original -> Reduced.

### Costo computacional

- sampling time;
- training time;
- inference time;
- peak GPU memory.

### Estadística

- prueba de Friedman;
- comparación entre los tres métodos;
- análisis por nivel de compresión;
- análisis global;
- post-hoc;
- corrección por comparaciones múltiples;
- tamaño de efecto.

---

## 32. Plan histórico anterior a la integración de PointNet++

### Paso 1 — Seleccionar implementación PointNet++

Buscar una implementación estable y suficientemente simple para:

- PyTorch;
- segmentación semántica;
- integración con XYZ + label;
- ejecución en nuestra RTX 4060;
- modificación mínima;
- reproducibilidad.

No implementar PointNet++ desde cero.

### Paso 2 — Adaptar Dataset / DataLoader

Crear pipeline que lea:

```text
data/processed/
```

y entregue:

```text
points: (N, 3)
labels: (N,)
```

con separación por:

```text
species
split
method
target
```

### Paso 4 — Congelar baseline

Una vez validado PointNet++:

- fijar arquitectura;
- fijar hiperparámetros;
- definir control;
- fijar número de épocas;
- fijar batch size;
- fijar optimizer;
- fijar learning rate;
- fijar scheduler;
- fijar seeds.

### Paso 5 — Validar métricas

Implementar y verificar:

```text
IoU stem
IoU leaf
mIoU
```

### Paso 6 — Benchmark de tiempos

Crear benchmark específico para:

- sampling;
- entrenamiento;
- inferencia;
- memoria GPU.

### Paso 8 — Campaña experimental completa

Solo después de que:

- dataset;
- PointNet++;
- métricas;
- timing;
- logging;
- control;
- seeds;

estén congelados.

### Paso 9 — Estadística

Aplicar Friedman y post-hoc sobre resultados consolidados.

### Paso 10 — Tablas y figuras finales

Generar:

- tablas mIoU;
- IoU por clase;
- Chamfer;
- tiempo;
- memoria;
- visualizaciones comparativas;
- figuras finales para examen.

---

## 33. Decisiones que NO deben cambiarse sin justificación

1. Soil se elimina antes del benchmark.
2. Clases finales: stem y leaf.
3. Maíz usa Leaf Collar annotation.
4. Sampling seed fija = 42.
5. Training seeds separadas del sampling seed.
6. Random, UVS y FPS deben preservar puntos reales.
7. Voxel usa centro geométrico del voxel.
8. FPS experimental es independiente del FPS interno de PointNet++.
9. Same hardware para timings finales.
10. La campaña de 160 runs se ejecutó tras validar el pipeline.
11. Chamfer compara nube original post-soil-removal vs nube reducida.
12. Splits por planta para evitar leakage temporal.

---

## 34. Archivos principales existentes

Los ejecutables vigentes están en `src/pipeline/`, numerados S01–S25 según el
flujo del experimento. La tabla completa y los comandos actuales están en
[README.md](../../README.md). Las bibliotecas permanecen en `src/data/`,
`src/sampling/`, `src/models/pointnet2/` y `src/metrics/`; el split permanece en
`configs/splits.yaml`. La reorganización conserva los registros numéricos de
esta bitácora y las rutas de los artefactos externos.

---

## 35. Punto exacto del proyecto

El proyecto está actualmente en:

```text
FASE COMPLETADA:
Dataset + splits + loader
        ↓
Random / UVS / FPS
        ↓
Validación visual
        ↓
Precomputación 1890 nubes
        ↓
Chamfer completo
        ↓
Resumen Chamfer
        ↓
Backup reproducible
```

Siguiente fase:

```text
PointNet++ semantic segmentation
```

---

## 37. Estado del checkpoint

Fecha:

```text
2026-09-02
```

Estado:

```text
CHECKPOINT COMPLETO
```

Bloques cerrados:

```text
GitHub
Docker backup
Project backup
Dataset preprocessing
Chamfer Distance
```

Bloque siguiente:

```text
PointNet++
```

---

# Checkpoint 2026-09-04 — Protocolo experimental congelado

## PointNet++ y pipeline

Se integró y validó el pipeline de segmentación semántica basado en PointNet++ SSG para Pheno4D.

Configuración congelada:

- Arquitectura: PointNet++ SSG
- Radios SA: `0.1 / 0.2 / 0.4 / 0.8`
- Clases: `stem` y `leaf`
- Normalización por nube
- Optimizer: Adam
- Learning rate: `5e-4`
- Scheduler: StepLR con `gamma=1.0` (LR efectivo fijo)
- Batch size: `2`
- Epochs: `50`
- Augmentation: desactivada
- Seeds de entrenamiento: `0,1,2,3,4`
- Eval seed fijo: `12345`
- Selección de checkpoint: mejor `val mIoU`
- Test permanece sellado

## Validación del evaluador

Se verificó `src/pipeline/S10_evaluate_pointnet2.py` sobre validación.

Checkpoint probado:

`data/results/training/tomato/random/16384/seed_0/no_aug/best_model.pt`

Resultado reproducido exactamente:

- Loss: `0.096017`
- IoU stem: `0.748702`
- IoU leaf: `0.970494`
- mIoU: `0.859598`
- Diferencia respecto al checkpoint: `0.0000000000`

Conclusión:

- carga de checkpoint correcta
- reconstrucción del dataset correcta
- `eval_seed=12345` respetado
- métricas reproducibles
- estructura `seed_X/no_aug/` compatible

## Baseline Random / 16384

### Tomato

Resultados por seed:

- seed 0: mIoU `0.859598`
- seed 1: mIoU `0.850216`
- seed 2: mIoU `0.868632`
- seed 3: mIoU `0.875289`
- seed 4: mIoU `0.843228`

Resumen aproximado:

- IoU stem: `0.7487 ± 0.0228`
- IoU leaf: `0.9700 ± 0.0035`
- mIoU: `0.8594 ± 0.0131`

### Maize

Resultados por seed:

- seed 0: mIoU `0.676543`
- seed 1: mIoU `0.707709`
- seed 2: mIoU `0.758374`
- seed 3: mIoU `0.760710`
- seed 4: mIoU `0.574763`

Resumen aproximado:

- IoU stem: `0.7094 ± 0.0703`
- IoU leaf: `0.6818 ± 0.0829`
- mIoU: `0.6956 ± 0.0763`

Observación:

- Tomato muestra alta estabilidad entre seeds.
- Maize presenta variabilidad considerable entre inicializaciones.
- No se harán más ajustes de arquitectura, radios, augmentation, loss, LR ni epochs antes del benchmark formal.

## Seed 0 de maize — limpieza de artefactos

Se detectó que existían dos ejecuciones bajo `seed_0`:

- baseline histórico correcto:
  `seed_0/run_summary.json`
  con mIoU `0.676543`
- piloto de radios:
  `seed_0/no_aug/run_summary.json`
  con mIoU `0.719398`

El piloto fue movido a:

`seed_0/diagnostic_radii/`

Después se reejecutó el baseline con el código restaurado y actual, obteniendo exactamente:

- Best epoch: `41`
- IoU stem: `0.684227`
- IoU leaf: `0.668858`
- mIoU: `0.676543`

La nueva corrida correcta quedó en:

`seed_0/no_aug/`

## Verificación de 10 runs oficiales del baseline

Se confirmaron exactamente 10 summaries oficiales:

- 5 seeds maize
- 5 seeds tomato

Todos bajo:

`data/results/training/{species}/random/16384/seed_X/no_aug/run_summary.json`

## Backup del proyecto

Snapshot:

`/home/seba/backups/q5/q5-project-protocol-frozen-2026-09-04.tar.zst`

SHA256:

`70ce3dd34af44a53e7837c425448063f5abf87035330e2cfead3af40bc5cf2b6`

## Backup Docker

Imagen:

`pheno-q5:pytorch`

Export:

`/home/seba/backups/q5/pheno-q5-pytorch-protocol-frozen-2026-09-04.tar`

SHA256:

`2fd6c1266d691f55a02d0a938d5fec922224f528c1114adcf1d2ee3ab1da6b00`

Este SHA256 coincide exactamente con el backup Docker anterior, confirmando que la imagen no cambió.

## Estado del protocolo

El protocolo base queda congelado en este checkpoint.

Pendiente inmediato:

1. Definir operacionalmente el control sin reducción.
2. Implementar y validar dicho control.
3. Lanzar benchmark formal:
   - Random
   - Voxel Grid
   - FPS
   - targets: 16384, 8192, 4096, 2048, 1024
   - species: maize, tomato
   - seeds: 0–4
4. Mantener test sellado hasta que el protocolo completo quede definitivamente cerrado.



## Revisión de protocolo — Control práctico de alta resolución (2026-09-06)

### Motivo de la revisión

El protocolo original contemplaba una condición control con la nube original sin reducción.

Se verificó experimentalmente que esta condición no es viable de forma uniforme con la arquitectura PointNet++ actual y la GPU disponible:

- una nube pequeña (~144k puntos) logró ejecutar forward y backward;
- una nube grande de validación (~4.45 M puntos post-soil) produjo OOM durante `query_ball_point`;
- el fallo requería una asignación del orden de decenas de GB, muy por encima de la VRAM disponible.

Para mantener:

- la misma arquitectura PointNet++;
- los mismos hiperparámetros;
- `batch_size=2`;
- el mismo pipeline de entrenamiento;
- un número fijo de puntos para todas las nubes;

se adoptó un **control práctico de alta resolución de 48,000 puntos**.

### Definición del control práctico

- condición: `control_48000`
- número de puntos: 48,000
- método de selección: Random Sampling sin reemplazo
- semilla de sampling: 42
- semilla de sampling independiente de las semillas de entrenamiento
- mismas particiones por planta
- misma normalización
- mismo PointNet++
- mismo batch size = 2
- mismo learning rate = 0.0005
- mismas 50 épocas
- sin augmentation

El valor 48,000 permite usar un tamaño fijo común inferior a la nube más pequeña observada en train/validation.

Este control debe describirse explícitamente como **control práctico de alta resolución** y no como “nube original sin reducción”.

La nube original post-soil continúa siendo la referencia para el cálculo de Chamfer Distance.

### Precomputación

Se creó:

`src/pipeline/S05_precompute_control_48000.py`

Resultado:

- 126 nubes control generadas
- `control_48000_metadata.csv`: 127 líneas
- `metadata.csv` factorial original permanece intacto con 1891 líneas

### Validación de memoria

Prueba histórica de memoria con:

- 48,000 puntos
- batch size = 2
- PointNet++ sin cambios

Resultado:

- ejecución correcta
- peak GPU memory ≈ 4902 MB

### Corridas del control

Se completaron las 10 corridas:

- maize: seeds 0, 1, 2, 3, 4
- tomato: seeds 0, 1, 2, 3, 4

#### Maize control_48000

mIoU por seed:

- seed 0: 0.612833
- seed 1: 0.756585
- seed 2: 0.686793
- seed 3: 0.688624
- seed 4: 0.645772

Resumen preliminar:

- IoU stem = 0.693603 ± 0.037736
- IoU leaf = 0.662640 ± 0.074963
- mIoU = 0.678121 ± 0.053970

#### Tomato control_48000

mIoU por seed:

- seed 0: 0.864156
- seed 1: 0.859770
- seed 2: 0.876010
- seed 3: 0.858951
- seed 4: 0.866241

Resumen preliminar:

- IoU stem = 0.758914 ± 0.011676
- IoU leaf = 0.971138 ± 0.002054
- mIoU = 0.865026 ± 0.006845

Los valores anteriores deberán extraerse nuevamente desde los `run_summary.json` al generar las tablas finales.

### Implementación del control

Se conserva:

- soporte de `control_48000` en el dataset;
- soporte de `target=48000` en entrenamiento;
- script formal de precomputación del control.

### Estado metodológico

El protocolo revisado reemplaza operacionalmente el control original no ejecutable por un control práctico de alta resolución.

El análisis estadístico de Friedman permanece restringido a los tres métodos experimentales:

- Random
- Voxel Grid
- FPS

en los cinco niveles oficiales de reducción.

El control práctico no forma parte del Friedman.

## Checkpoint 2026-09-07 — Cierre de análisis Q5

- Entrenamiento formal completado: 160/160 ejecuciones.
  - Factorial: 150/150.
  - Control práctico de alta resolución: 10/10.
- Evaluación formal en test completada: 160/160 checkpoints.
- Benchmark dedicado de inferencia completado.
- Resultados consolidados en tablas maestras y resúmenes.
- Análisis exploratorios históricos completados; fuera del pipeline final.
- Chamfer Distance consolidado para las 30 combinaciones factoriales.
- Trade-off integrado: mIoU, Chamfer, sampling, inferencia, entrenamiento y memoria GPU.
- Friedman por especie × target completado.
- Friedman global completado:
  - χ² = 24.04
  - p = 6.0225e-06
  - Kendall W = 0.2404
- Post-hoc global Wilcoxon + Holm:
  - Random vs Voxel: significativo.
  - Random vs FPS: significativo.
  - Voxel vs FPS: significativo.
- Tendencia global: FPS > Voxel > Random.
- Puntos de operación prácticos seleccionados:
  - Maíz: Random 8192.
  - Tomate: FPS 4096.
- Tablas del paper generadas:
  - desempeño de segmentación;
  - fidelidad y costo computacional;
  - análisis estadístico.
- Las figuras definitivas aún requieren una pasada estética final para publicación.

### Pendientes principales

- Revisar y compactar tablas para formato IEEE.
- Pulir figuras definitivas para paper.
- Redactar sección de Resultados de Q5.
- Revisar cumplimiento final del enunciado.
- Desarrollar/responder Q1–Q4 con soporte bibliográfico.