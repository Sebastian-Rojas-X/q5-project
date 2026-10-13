# Reproducibilidad y artefactos externos

Autor: Sebastián Rojas · srojas.hz@gmail.com · 2026.

El código y la documentación se versionan. `data/` permanece fuera de Git;
conserva originales, NPZ, metadata, checkpoints, historiales y resultados. Las
rutas relativas guardadas en manifiestos y CSV requieren preservar ese árbol.

## Entorno histórico

La [bitácora](checkpoints/Bitacora_Checkpoint_2026-09-07.md) registra:

| Elemento | Registro histórico |
|---|---|
| Imagen | `pheno-q5:pytorch` |
| Base Docker | `nvidia/cuda:13.0.2-runtime-ubuntu24.04` |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| CPU | Intel Core i9-14900HX |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, aproximadamente 8 GB VRAM |
| RAM | 32 GB |
| Sistema / Docker / Git | Ubuntu 26 / 29.7.2 / 2.53.0 |

Son registros del experimento; no identifican automáticamente el entorno actual.
No constan el driver NVIDIA exacto, el ID/digest de la imagen ni un inventario
completo de versiones de dependencias. El Dockerfile instala paquetes sin fijar
versiones y no declara directamente todos los paquetes usados. El pipeline
requiere PyTorch, NumPy, SciPy, pandas, PyYAML y Matplotlib. Un intérprete anfitrión
sin PyTorch no puede importar los motores de entrenamiento o benchmark.

La bitácora registra un respaldo de imagen:

```text
/home/seba/backups/q5/pheno-q5-pytorch.tar
SHA-256: 2fd6c1266d691f55a02d0a938d5fec922224f528c1114adcf1d2ee3ab1da6b00
```

Verificar ese checksum antes de restaurar con `docker load -i <archivo>`.
La existencia actual y el contenido del respaldo no fueron verificados durante
la reorganización. Para conservar el entorno efectivo se debe recuperar esa
imagen y registrar su ID y sus versiones; no sustituirlos por versiones nuevas.

## Entradas necesarias

| Reproducción | Artefactos externos necesarios |
|---|---|
| Preparación | Distribución original Pheno4D extraída en `data/raw/Pheno4D/` |
| Entrenamiento/evaluación | Manifiestos, NPZ y metadata; checkpoints para evaluación |
| H99 | Originales, NPZ, metadata factorial y Chamfer individual congelado |
| Integración y estadística | Master/resumen de segmentación, resúmenes geométricos y benchmark |
| Tradeoff | Integrado, benchmark y metadata factorial/control con tiempos históricos |
| Tablas/figuras | CSV finales integrados y estadísticos |

El ZIP original se registra como `data/raw/Pheno4D.zip`. Su versión/release,
procedencia exacta y checksum no están registrados en las fuentes auditadas.
Recuperar el archivo efectivamente empleado y documentar su identidad. Los
respaldo(s) de proyecto y sus hashes históricos están en la bitácora.

## Trabajo sobre una copia independiente

Mantener intacto el proyecto que contiene los resultados congelados. Copiar el
código reorganizado y restaurar los artefactos de entrada necesarios dentro de
una carpeta de trabajo diferente, conservando sus rutas relativas. Para repetir
preparación, esa carpeta debe contener los originales y carecer de los destinos
generados. Para repetir post-hoc, restaurar las fuentes congeladas y omitir los
destinos de S15–S23; para figuras, utilizar el tradeoff final y destinos nuevos.

El lanzador rechaza destinos ya existentes en las etapas que generan archivos.
Los comandos Python individuales conservan su comportamiento original de
escritura; ejecutarlos también requiere una copia independiente.

```bash
bash run_experiment.sh verify
bash run_experiment.sh train --dry-run
bash run_experiment.sh evaluate --dry-run
bash run_experiment.sh benchmark --dry-run
```

`verify` comprueba los cuatro SHA-256 congelados, conteos y referencias de los
artefactos de cierre. No certifica que un reentrenamiento nuevo produzca los
mismos bytes ni tiempos: valida los archivos históricos disponibles. La
identidad de resultados nuevos requiere comparar las métricas y registrar el
entorno y hardware utilizados.

## Integridad tras la reorganización

Los cuatro hashes del README identifican master, resumen y los dos CSV Chamfer
originales. Las proyecciones finales de integración, tradeoff y tabla 1 conservan
los valores y orden de las columnas retenidas; su esquema de columnas produce
bytes diferentes al regenerarlas. Los originales permanecen intactos.

Las comprobaciones de equivalencia ejecutan análisis en carpetas temporales y
comparan las salidas contra las fuentes previas. La reorganización de scripts
solo cambia nombres de módulos y resolución de rutas. No reentrena modelos,
regenera geometría ni reemplaza las mediciones históricas durante la limpieza.

No se deben versionar datos, checkpoints, cachés ni resultados pesados. Todos
los Python contienen el encabezado del proyecto; la atribución externa del
modelo se conserva por separado en [third_party.md](third_party.md).
