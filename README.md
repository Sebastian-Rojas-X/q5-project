# Q5 — Downsampling y segmentación semántica de nubes de puntos en Pheno4D

Repositorio experimental de la **Pregunta 5 (Q5)** del examen de calificación. El objetivo es estudiar cómo distintas estrategias de reducción de puntos afectan la segmentación semántica de órganos vegetales en nubes de puntos 3D, manteniendo fija la arquitectura de segmentación y evaluando conjuntamente desempeño, fidelidad geométrica y costo computacional.

> **Estado actual:** experimento congelado.
> No se deben reentrenar modelos ni cambiar arquitectura, split, samplers, número de puntos, semillas o hiperparámetros para modificar resultados. Las tareas posteriores son de análisis post-hoc, integración, visualización, documentación y redacción.

---

## Resumen del experimento

El benchmark ejecutado corresponde a:

| Factor | Configuración |
|---|---|
| Dataset | Pheno4D |
| Especies | Maíz y tomate |
| Segmentación | Tallo vs hoja |
| Arquitectura | PointNet++ SSG |
| Samplers | Random, Voxel Grid, FPS |
| Puntos objetivo | 1,024; 2,048; 4,096; 8,192; 16,384 |
| Semillas de entrenamiento | 0, 1, 2, 3, 4 |
| Control | `control_48000` |
| Ejecuciones factoriales | 150 |
| Ejecuciones de control | 10 |
| Total | **160** |

El identificador interno `control_48000` debe describirse en texto científico como **control práctico de alta resolución**. En las figuras se utiliza la etiqueta abreviada **Control 48k**. La nube original post-ground se mantiene como referencia geométrica independiente y no debe confundirse con el control.

---

## Dataset y partición por planta

Se utiliza el subconjunto útil etiquetado de Pheno4D:

- **Maíz:** 49 nubes.
- **Tomate:** 77 nubes.
- **Total:** 126 nubes post-ground etiquetadas.
- **Clase 0:** tallo.
- **Clase 1:** hoja.

La partición se realiza a nivel de planta para evitar fuga temporal entre entrenamiento, validación y prueba.

### Maíz

```text
Train: Maize01, Maize02, Maize04, Maize05
Validation: Maize03
Test: Maize06, Maize07
```

### Tomate

```text
Train: Tomato01, Tomato02, Tomato03, Tomato04
Validation: Tomato05
Test: Tomato06, Tomato07
```

La misma reducción externa se utiliza de forma coherente durante entrenamiento e inferencia para cada configuración.

---

## Configuración fija de segmentación

```text
Arquitectura: PointNet++ SSG
Entrada: XYZ
Número de clases: 2
Épocas: 50
Batch size: 2
Optimizador: Adam
Learning rate: 5e-4
Augmentation: no
Checkpoint: mejor mIoU de validación
sampling_seed: 42
eval_seed: 12345
Scheduler: StepLR(step_size=10, gamma=1.0)
```

La variable experimental principal es el preprocesamiento externo de puntos.

---

## Métricas

### Segmentación

Métricas principales:

- mIoU.
- IoU tallo.
- IoU hoja.

Los resultados se resumen como media y desviación estándar sobre cinco semillas de entrenamiento.

### Fidelidad geométrica

#### Chamfer Distance

La implementación congelada utiliza distancia Euclidiana L2 no cuadrática:

```text
CD(P, S) = 0.5 * [ mean_{p in P} d(p,S) + mean_{s in S} d(s,P) ]
```

con:

- `P`: nube original post-ground;
- `S`: nube reducida;
- `d(x,A)`: distancia Euclidiana al vecino más cercano en `A`.

Como la nube reducida es subconjunto de la nube original post-ground, el término reducido→original es cero en los resultados almacenados. Para este benchmark:

```text
CD = 0.5 * mean(original → reduced)
```

#### Robust Hausdorff H99

Como complemento se calcula:

```text
H99 = percentile_99( min_{s in S} ||p - s||_2,  p in P )
```

Definición usada:

- dirección: original → reducida;
- distancia: Euclidiana L2 no cuadrática;
- percentil: 99;
- método NumPy: `linear`;
- mismas coordenadas y unidades que Chamfer.

### Costo computacional

Por configuración se integran:

- tiempo de sampling por nube;
- tiempo de entrenamiento;
- tiempo de inferencia por nube;
- memoria GPU máxima de entrenamiento;
- memoria GPU máxima de inferencia.

La discusión final utiliza el concepto de **accuracy–efficiency trade-off** para relacionar desempeño de segmentación y costo computacional.

---

## Análisis estadístico

### Comparación entre samplers

Se utiliza:

- Friedman por especie × resolución;
- Friedman global;
- comparaciones post-hoc pareadas;
- corrección de Holm;
- tamaños de efecto.

Tamaños de efecto:

- Kendall's W para Friedman;
- rank-biserial correlation para comparaciones pareadas.

### Comparación contra el control

Cada una de las 15 configuraciones factoriales por especie se compara con `control_48000` emparejando por semilla.

```text
Wilcoxon signed-rank exacto, bilateral
Holm sobre 15 comparaciones dentro de cada especie
matched-pairs rank-biserial correlation
ΔmIoU = configuración - control
```

Resultado inferencial resumido:

```text
30 comparaciones configuración-control
0/30 significativas después de Holm
```

Con `n = 5` pares no nulos y sin empates, el menor valor-p bilateral exacto posible de Wilcoxon es `0.0625`. Por tanto:

> **Ausencia de significación estadística no implica equivalencia.**

El punto operativo debe discutirse usando conjuntamente ΔmIoU, tamaño de efecto y ahorro computacional, sin interpretar `p >= 0.05` como prueba de igualdad.

---

## Estructura del repositorio

```text
q5-project/
├── AGENTS.md
├── README.md
├── Dockerfile
├── .gitignore
├── run_experiment.sh
├── configs/
│   └── splits.yaml
├── docs/
│   ├── protocol.md
│   ├── reproducibility.md
│   ├── third_party.md
│   └── checkpoints/
├── src/
│   ├── pipeline/                # 25 ejecutables S01–S25
│   ├── data/                    # Loader y dataset
│   ├── sampling/                # Random, Voxel y FPS
│   ├── models/pointnet2/        # Arquitectura congelada
│   └── metrics/                 # Bibliotecas IoU y Chamfer
└── data/                        # Artefactos externos, ignorados por Git
    ├── raw/Pheno4D/
    ├── manifests/
    ├── processed/
    └── results/
        ├── training/
        ├── analysis/
        ├── paper_tables/
        └── figures/
            ├── tradeoff/
            └── tradeoff_publication/
```

Los ejecutables se invocan como módulos desde la raíz. Las bibliotecas conservan
sus ubicaciones para mantener imports, arquitectura y comportamiento numérico.
Los checkpoints e historiales permanecen bajo `data/results/training/`: los
manifiestos y CSV guardan referencias a esas rutas.

`/data/` está excluido de Git. Para reproducir el cierre se necesitan los
artefactos externos descritos en [reproducibility.md](docs/reproducibility.md).
El protocolo vigente está documentado en [protocol.md](docs/protocol.md) y la
atribución del modelo en [third_party.md](docs/third_party.md).

---

## Entorno de ejecución

El entorno original utiliza la imagen Docker:

```text
pheno-q5:pytorch
```

Patrón general:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  bash run_experiment.sh verify
```

Para los análisis CPU-only, el aviso de Docker indicando que no se detectó un driver NVIDIA es esperable y no afecta los resultados.

---

## Pipeline completo y ejecución por etapas

Todos los comandos se ejecutan desde la raíz en el entorno original. El
lanzador no cambia parámetros experimentales y evita escribir sobre las salidas
existentes de preparación, benchmark, consolidación, análisis y figuras.
Entrenamiento y evaluación reutilizan el comportamiento de sus lanzadores para
omitir las corridas ya completadas. La invocación sin argumentos verifica
integridad y no genera resultados:

```bash
bash run_experiment.sh
bash run_experiment.sh train --dry-run
bash run_experiment.sh evaluate --dry-run
bash run_experiment.sh benchmark --dry-run
```

Para ejecutar una etapa con CUDA dentro de la imagen original:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace" -w /workspace \
  pheno-q5:pytorch bash run_experiment.sh train --dry-run
```

Los paquetes requeridos incluyen PyTorch, NumPy, SciPy, pandas, PyYAML y
Matplotlib. El `Dockerfile` instala versiones sin fijar; reconstruirlo hoy no
identifica por sí solo el entorno histórico. La imagen respaldada y sus límites
de trazabilidad están descritos en [reproducibility.md](docs/reproducibility.md).

### Preparación desde Pheno4D

Obtener el archivo original de Pheno4D usado en el experimento y extraerlo en
`data/raw/Pheno4D/`, con las carpetas `Maize01`–`Maize07` y
`Tomato01`–`Tomato07`. Se utilizan las 126 nubes anotadas con sufijo `_a.txt`:
49 de maíz y 77 de tomate. Maíz utiliza Leaf Collar; el loader elimina suelo y
remapea tallo a 0 y hojas a 1. El split permanece en `configs/splits.yaml`.
La procedencia, versión y checksum del ZIP original requieren el registro del
archivo efectivamente empleado; este repositorio no inventa un reemplazo.

En una copia independiente que contenga código y originales, con los destinos
generados todavía ausentes:

```bash
bash run_experiment.sh prepare
```

Esta etapa ejecuta catálogo → validación de split → manifiestos → muestreo
factorial completo → control → Chamfer → resumen de Chamfer. Se esperan 223
archivos catalogados, seis manifiestos, 1,890 nubes factoriales y 126 de control.
La metadata factorial se genera completa en una única ejecución; ejecuciones
parciales reemplazan su CSV y no representan el benchmark completo.

### Entrenamiento, evaluación y consolidación

La reconstrucción de la campaña se realiza exclusivamente en esa copia:

```bash
bash run_experiment.sh train
bash run_experiment.sh evaluate
bash run_experiment.sh benchmark
bash run_experiment.sh collect
```

`train` llama a S09 para Random, Voxel y FPS, preservando el orden interno de
resoluciones y semillas, y a S08 para los diez controles de 48,000 puntos.
S11 evalúa los 160 checkpoints en test. S12 mide el forward dedicado con batch
2, cinco warmups y veinte repeticiones, incluyendo el control. S13/S14 producen
160 filas maestras y 32 grupos. Los tiempos nuevos pertenecen a esa ejecución y
no reemplazan los tiempos históricos del examen.

### Análisis y entrega final

Con las fuentes de segmentación, Chamfer, benchmark, originales, NPZ y metadata
disponibles, y destinos derivados ausentes:

```bash
bash run_experiment.sh posthoc
bash run_experiment.sh figures
```

Para regenerar solamente el análisis se pueden omitir preparación y
entrenamiento restaurando las fuentes congeladas en una copia independiente.
El análisis de H99 necesita los originales y NPZ; tablas y figuras, una vez
producido el tradeoff, consumen sus CSV. Los comandos individuales se detallan
abajo y el orden completo aparece en la tabla siguiente.

| Etapa | Ejecutable en `src/pipeline/` | Entrada principal → salida |
|---|---|---|
| S01 | `S01_catalog_pheno4d.py` | Originales → catálogo |
| S02 | `S02_validate_splits.py` | Split YAML → validación |
| S03 | `S03_create_manifests.py` | Catálogo y YAML → seis manifiestos |
| S04 | `S04_precompute_downsampling.py` | Manifiestos y originales → 1,890 NPZ y metadata |
| S05 | `S05_precompute_control_48000.py` | Manifiestos y originales → 126 NPZ de control |
| S06 | `S06_compute_chamfer.py` | Originales y reducidas → Chamfer individual |
| S07 | `S07_summarize_chamfer.py` | Chamfer individual → 30 grupos |
| S08 | `S08_train_pointnet2.py` | NPZ → checkpoint e historial de una corrida |
| S09 | `S09_run_training_batch.py` | S08 → 150 corridas factoriales |
| S10 | `S10_evaluate_pointnet2.py` | Checkpoint y NPZ → evaluación individual |
| S11 | `S11_run_evaluation_batch.py` | S10 → 160 evaluaciones test |
| S12 | `S12_benchmark_inference.py` | Checkpoints y test → benchmark de 160 corridas / 32 grupos |
| S13 | `S13_collect_results.py` | JSON train/test → master de 160 filas |
| S14 | `S14_summarize_results.py` | Master → resumen de 32 grupos |
| S15 | `S15_compute_hausdorff_h99.py` | Originales, reducidas y Chamfer → H99 |
| S16 | `S16_integrate_results.py` | Resúmenes finales → 32 grupos integrados |
| S17 | `S17_build_tradeoff_table.py` | Integrado y recursos → tradeoff |
| S18 | `S18_run_friedman.py` | Master → Friedman por especie y resolución |
| S19 | `S19_run_posthoc_and_global.py` | Master y Friedman → post-hoc local / Friedman global |
| S20 | `S20_run_global_posthoc.py` | Master → post-hoc global |
| S21 | `S21_analyze_operating_points.py` | Master y tradeoff → 30 comparaciones contra control |
| S22 | `S22_build_statistical_summary.py` | CSV estadísticos y control → 44 filas |
| S23 | `S23_build_paper_tables.py` | Tradeoff y estadística → cuatro tablas CSV/TEX |
| S24 | `S24_generate_tradeoff_figures.py` | Tradeoff → 16 figuras técnicas |
| S25 | `S25_generate_publication_figures.py` | Tradeoff → cinco figuras PNG/PDF |

S08 y S10 son motores individuales llamados por S09/S11; no se ejecutan además
para duplicar las mismas corridas. Las bibliotecas de datos, samplers, modelo y
métricas se importan desde estos ejecutables.

---

## Reproducción del análisis post-hoc

Ejecutar desde la raíz del repositorio con las fuentes experimentales congeladas y el benchmark de inferencia disponibles. La integración precede a la tabla de trade-off; ambas preceden a las comparaciones contra el control.

### Fuentes estadísticas requeridas

El resumen estadístico consume los CSV generados por estos tres productores, además de las comparaciones contra el control. Se ejecutan en este orden, con `results_master.csv` congelado disponible. Los nombres y procedimientos estadísticos existentes se conservan.

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S18_run_friedman

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S19_run_posthoc_and_global

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S20_run_global_posthoc
```

Salidas bajo `data/results/analysis/`:

```text
friedman_test_miou.csv
posthoc_test_miou.csv
friedman_global_test_miou.csv
posthoc_global_test_miou.csv
```

### 1. Robust Hausdorff H99

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S15_compute_hausdorff_h99
```

Salidas:

```text
data/results/analysis/hausdorff_h99_results.csv
data/results/analysis/hausdorff_h99_summary.csv
```

Validación realizada:

```text
1890 comparaciones geométricas
error máximo de recomputación Chamfer ≈ 5e-13 a nivel de nube
```

### 2. Integración de resultados

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S16_integrate_results
```

Validación esperada:

```text
Segmentation groups: 32 / 32
Chamfer groups: 30 / 30
H99 groups: 30 / 30
Factorial groups: 30 / 30
Control groups: 2 / 2
Total integrated groups: 32 / 32
```

Salida:

```text
data/results/analysis/results_integrated.csv
```

### 3. Tabla integrada accuracy–efficiency

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S17_build_tradeoff_table
```

Salida:

```text
data/results/analysis/results_tradeoff.csv
```

### 4. Comparaciones contra el control

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S21_analyze_operating_points
```

Salida:

```text
data/results/analysis/operating_points.csv
```

Debe contener 30 comparaciones: 15 por especie.

### 5. Resumen estadístico reproducible

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S22_build_statistical_summary
```

Salidas:

```text
data/results/analysis/statistical_summary.csv
data/results/analysis/statistical_summary.txt
```

Validación esperada:

```text
Species-target Friedman rows: 10 / 10
Global Friedman rows: 1 / 1
Global post-hoc rows: 3 / 3
Control-comparison rows: 30 / 30
Summary rows: 44 / 44
```

### 6. Tablas reproducibles para el paper

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S23_build_paper_tables
```

Salidas:

```text
data/results/paper_tables/table_1_segmentation_performance.csv/.tex
data/results/paper_tables/table_2_cost_fidelity.csv/.tex
data/results/paper_tables/table_3_statistics.csv/.tex
data/results/paper_tables/table_4_control_comparisons.csv/.tex
```

Filas esperadas:

```text
Tabla 1: 32
Tabla 2: 32
Tabla 3: 14
Tabla 4: 30
```

### 7. Figuras técnicas

Este generador forma parte de la entrega final: produce las curvas de tiempo
de sampling e inferencia frente al número de puntos para ambas especies.
Estas relaciones no están incluidas en las cinco figuras de publicación.

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S24_generate_tradeoff_figures
```

Salida:

```text
data/results/figures/tradeoff/
```

### 8. Figuras publication-ready

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  pheno-q5:pytorch \
  python3 -B -m src.pipeline.S25_generate_publication_figures
```

Salida:

```text
data/results/figures/tradeoff_publication/
```

Se generan cinco figuras en PNG y PDF:

```text
figure_1_performance_vs_points
figure_2_geometry_vs_points
figure_3_accuracy_vs_inference_time
figure_4_accuracy_vs_sampling_time
figure_5_resource_profile
```

---

## Resultados experimentales congelados

Los siguientes cuatro archivos son las fuentes congeladas principales del experimento:

```text
6360a77f7b995a2e9acc2639e857d9dace1a0f83401381871835df5f0da78dd1  data/results/analysis/results_master.csv
95b2c42ac8d832ddfb70909881d4cc186e4d776422d121d093dfb652f811f63d  data/results/analysis/results_summary.csv
5521dbe04e789d762ffa3d72046f7ec046e6bad8fd18185d6d802c8e0c44750c  data/results/chamfer_results.csv
22eb24151e02d354bb8b171e5a546efba3607dd98c72bb6a4b76d8d515a8482f  data/results/chamfer_summary.csv
```

Comprobación:

```bash
sha256sum \
  data/results/analysis/results_master.csv \
  data/results/analysis/results_summary.csv \
  data/results/chamfer_results.csv \
  data/results/chamfer_summary.csv
```

Si cualquiera de estos hashes cambia, debe detenerse el análisis y revisarse la causa antes de continuar.

---

## Artefactos post-hoc derivados

Hashes registrados durante el cierre experimental para los archivos existentes.
La proyección final de columnas de integración, tradeoff y tabla 1 cambia su
esquema al regenerarlos; sus métricas retenidas deben permanecer idénticas.
Estos hashes documentan los originales, que se mantienen intactos:

```text
operating_points.csv
edfab924acd000483429da2a5f6eac530bc6689e0b3b183f68d1eb4bbe7fe618

hausdorff_h99_results.csv
40829e665fbff34bddc5536e12f85cba779635b1c7055f9c16bda2c9172c12f4

hausdorff_h99_summary.csv
d1f437275eb34f6db1685fd944d8e07a35520bbcea7f23438b2c84cbe00d5b1f

results_integrated.csv
a75cf609e36354c58e2de55e79dfb219d6b53dae48237537861c8edc4f5bb5f7

results_tradeoff.csv
3192a2fa4cac8c6492873804af0fbf254f7fc921c7d85f57f1e23bf243c6121f

statistical_summary.csv
acf837ce3dee6b968b98a6cc50578deab948e17dc9f7f4eb8ae1d18802b10b09
```

Tablas derivadas registradas:

```text
table_1_segmentation_performance.csv
763065a76c75ddb20cac6c37604c9b4f103066e2fd040e40f04b457bd36f1f4b

table_2_cost_fidelity.csv
13a732ae28efe95e28909523176fcfa1bd67582a29f91f6c2e0ecbdc22b27df3

table_3_statistics.csv
cf0e7deb898a7a76950f9782838402d91ef5d18011dc107a8e0895e8f52c1b2c

table_4_control_comparisons.csv
b20a56cf3940e95dfd842060c0b3f634ca2ca069b1e544eeb6a1001bdb78a8c1
```

> Los hashes de integración, tradeoff y tabla 1 corresponden al esquema de cierre conservado. La regeneración del esquema final se verifica comparando las columnas retenidas; no reemplaza los originales. Los cuatro hashes experimentales congelados de la sección anterior no deben cambiar.

---

## Resultados estadísticos globales congelados

Friedman global sobre mIoU:

```text
χ² = 24.04
p = 0.0000060225
Kendall W = 0.2404
```

Post-hoc global:

```text
Random vs Voxel
Holm p = 0.0122623605
r_rb = -0.440
Δ = -0.0144

Random vs FPS
Holm p = 0.0006083902
r_rb = -0.584
Δ = -0.0273

Voxel vs FPS
Holm p = 0.0289842763
r_rb = -0.354
Δ = -0.0129
```

Estos resultados describen el comportamiento agregado y no implican que un sampler sea siempre superior para toda especie y resolución.

---

## Resumen geométrico H99

En las 30 combinaciones especie × sampler × target se observa el orden geométrico:

```text
FPS < Voxel < Random
```

para H99, donde menor es mejor.

H99 disminuye monotónicamente al aumentar el número de puntos para cada sampler y especie.

Ejemplo a 4,096 puntos:

```text
Maíz:
FPS    0.9426
Voxel  1.0404
Random 1.8364

Tomate:
FPS    0.9221
Voxel  1.0393
Random 1.7929
```

Este patrón geométrico no implica automáticamente el mismo orden en desempeño de segmentación.

---

## Verificación rápida

### Estado del repositorio

```bash
git status --short
git rev-parse HEAD
```

### Hashes congelados

```bash
sha256sum \
  data/results/analysis/results_master.csv \
  data/results/analysis/results_summary.csv \
  data/results/chamfer_results.csv \
  data/results/chamfer_summary.csv
```

### Conteos principales

```bash
wc -l \
  data/results/analysis/results_master.csv \
  data/results/analysis/results_summary.csv \
  data/results/chamfer_results.csv \
  data/results/chamfer_summary.csv
```

Esperado, incluyendo encabezado:

```text
results_master.csv      161 líneas
results_summary.csv      33 líneas
chamfer_results.csv    1891 líneas
chamfer_summary.csv      31 líneas
```

---

## Hardware

La bitácora registra CPU Intel Core i9-14900HX, NVIDIA GeForce RTX 4060 Laptop
GPU, aproximadamente 8 GB de VRAM y 32 GB de RAM. Estos son registros históricos
del experimento, no mediciones de la máquina que ejecute el repositorio después.
El driver exacto y el ID/digest de la imagen no están registrados en las fuentes
auditadas. Véase [reproducibility.md](docs/reproducibility.md).

Las mediciones de costo deben conservar el hardware y el procedimiento
originales; una nueva medición debe identificarse separadamente.

---

## Política de Git y resultados

La política actual del repositorio mantiene `/data/` fuera de Git. Por tanto:

- datasets no se suben;
- nubes procesadas no se suben;
- checkpoints no se suben;
- CSV derivados no se suben;
- tablas y figuras bajo `/data/` no se suben;
- el código fuente sí se versiona.

Para la entrega del examen, los artefactos reproducibles pueden empaquetarse por separado sin cambiar esta política.

---

## Criterios para la redacción final de Q5

1. Usar **accuracy–efficiency trade-off** como marco principal para discutir desempeño frente a costo.
2. No presentar ausencia de significación como equivalencia.
3. Reportar ΔmIoU y tamaño de efecto junto con los valores-p.
4. Mantener diferenciada la fidelidad geométrica de la segmentación: menor Chamfer/H99 no implica necesariamente mayor mIoU.
5. Describir `control_48000` como **control práctico de alta resolución**; en figuras se usa **Control 48k**.
6. Mantener explícito que la nube original post-ground es la referencia geométrica.
7. No introducir nuevos resultados experimentales durante la escritura.
8. Explicar cualquier configuración práctica a partir de la evidencia conjunta de desempeño y costo, no mediante una afirmación de equivalencia no demostrada.

---

## Referencias base

- Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*. NeurIPS 30, 5099–5108.
- Schunck, D., Magistri, F., Roscher, R., Rosu, R. A., Cornelißen, A., Chebrolu, N., Paulus, S., Léon, J., Behnke, S., Stachniss, C., Kuhlmann, H., & Klingbeil, L. (2021). *Pheno4D: A spatio-temporal dataset of maize and tomato plant point clouds for phenotyping and advanced plant analysis*. PLoS ONE, 16(8), e0256340.

---

## Estado de cierre

| Componente | Estado |
|---|---|
| Diseño factorial | ✅ Cerrado |
| 160 ejecuciones | ✅ Cerrado |
| mIoU e IoU por clase | ✅ Cerrado |
| Chamfer | ✅ Cerrado |
| H99 robusto | ✅ Cerrado |
| Tiempos y memoria | ✅ Cerrado |
| Friedman + post-hoc | ✅ Cerrado |
| Corrección Holm | ✅ Cerrado |
| Tamaños de efecto | ✅ Cerrado |
| Comparación contra control | ✅ Cerrado |
| Tablas reproducibles | ✅ Cerrado |
| Figuras técnicas | ✅ Cerrado |
| Figuras publication-ready | ✅ Cerrado |
| Hardware histórico | Documentado; driver e ID de imagen sin registro |
| Paper IEEE | ⚠️ Pendiente |

La estructura final y la integridad de los artefactos se verifican con
`bash run_experiment.sh verify`. El protocolo experimental y los resultados
numéricos originales permanecen congelados.
