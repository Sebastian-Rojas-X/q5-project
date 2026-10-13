# Procedencia de PointNet++

La arquitectura y los operadores conservados en `src/models/pointnet2/`
contienen una adaptación de `yanx27/Pointnet_Pointnet2_pytorch`.
`pointnet2_utils.py` mantiene su atribución original y referencia a MIT:

<https://github.com/yanx27/Pointnet_Pointnet2_pytorch>

El docstring de `pointnet2_sem_seg.py` documenta la adaptación para entradas XYZ
y dos clases del benchmark. Los encabezados de Sebastián Rojas identifican el
mantenimiento del proyecto; la atribución externa permanece en los fuentes.

No se registra un commit/release de la copia original ni un archivo de licencia
completo en el repositorio auditado. Este documento registra la atribución
existente; no reconstruye esos datos de procedencia ni asigna una licencia nueva
al proyecto. La reorganización conserva los operadores y la arquitectura sin
reemplazarlos por una versión externa actual.
