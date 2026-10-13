"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Define the frozen PointNet++ SSG semantic segmentation model and loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)


class PointNet2SemanticSegmentation(nn.Module):
    """
    PointNet++ SSG para segmentación semántica.

    Adaptación mínima del modelo pointnet2_sem_seg.py
    de yanx27/Pointnet_Pointnet2_pytorch.

    Diferencia principal respecto al modelo S3DIS:
    ------------------------------------------------
    El modelo original recibe 9 canales por punto.

    Nuestro benchmark Pheno4D utiliza únicamente:

        X, Y, Z

    por lo que la entrada tiene forma:

        [B, 3, N]

    En la primera Set Abstraction, PointNet++ concatena:

        XYZ relativo: 3 canales
        features XYZ: 3 canales

    Total:

        6 canales
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # --------------------------------------------------
        # SET ABSTRACTION
        # --------------------------------------------------

        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.1,
            nsample=32,
            in_channel=6,       # 3 XYZ relativos + 3 XYZ features
            mlp=[32, 32, 64],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.sa4 = PointNetSetAbstraction(
            npoint=16,
            radius=0.8,
            nsample=32,
            in_channel=256 + 3,
            mlp=[256, 256, 512],
            group_all=False,
        )

        # --------------------------------------------------
        # FEATURE PROPAGATION
        # --------------------------------------------------

        self.fp4 = PointNetFeaturePropagation(
            768,
            [256, 256],
        )

        self.fp3 = PointNetFeaturePropagation(
            384,
            [256, 256],
        )

        self.fp2 = PointNetFeaturePropagation(
            320,
            [256, 128],
        )

        self.fp1 = PointNetFeaturePropagation(
            128,
            [128, 128, 128],
        )

        # --------------------------------------------------
        # CLASIFICADOR POR PUNTO
        # --------------------------------------------------

        self.conv1 = nn.Conv1d(
            128,
            128,
            1,
        )

        self.bn1 = nn.BatchNorm1d(
            128
        )

        self.drop1 = nn.Dropout(
            0.5
        )

        self.conv2 = nn.Conv1d(
            128,
            num_classes,
            1,
        )

    def forward(self, xyz):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            Forma:
                [B, 3, N]

        Returns
        -------
        x : torch.Tensor
            Log-probabilidades por punto.

            Forma:
                [B, N, num_classes]

        l4_points : torch.Tensor
            Features profundas de PointNet++.
        """

        # XYZ también se utiliza como feature inicial.
        l0_points = xyz

        # Coordenadas geométricas.
        l0_xyz = xyz[:, :3, :]

        # --------------------------------------------------
        # ENCODER / SET ABSTRACTION
        # --------------------------------------------------

        l1_xyz, l1_points = self.sa1(
            l0_xyz,
            l0_points,
        )

        l2_xyz, l2_points = self.sa2(
            l1_xyz,
            l1_points,
        )

        l3_xyz, l3_points = self.sa3(
            l2_xyz,
            l2_points,
        )

        l4_xyz, l4_points = self.sa4(
            l3_xyz,
            l3_points,
        )

        # --------------------------------------------------
        # DECODER / FEATURE PROPAGATION
        # --------------------------------------------------

        l3_points = self.fp4(
            l3_xyz,
            l4_xyz,
            l3_points,
            l4_points,
        )

        l2_points = self.fp3(
            l2_xyz,
            l3_xyz,
            l2_points,
            l3_points,
        )

        l1_points = self.fp2(
            l1_xyz,
            l2_xyz,
            l1_points,
            l2_points,
        )

        l0_points = self.fp1(
            l0_xyz,
            l1_xyz,
            None,
            l1_points,
        )

        # --------------------------------------------------
        # PREDICCIÓN POR PUNTO
        # --------------------------------------------------

        x = self.conv1(
            l0_points
        )

        x = self.bn1(
            x
        )

        x = F.relu(
            x
        )

        x = self.drop1(
            x
        )

        x = self.conv2(
            x
        )

        x = F.log_softmax(
            x,
            dim=1,
        )

        # [B, classes, N]
        # ->
        # [B, N, classes]

        x = x.permute(
            0,
            2,
            1,
        )

        return x, l4_points


class PointNet2Loss(nn.Module):
    """
    Negative Log Likelihood Loss
    para segmentación semántica.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions,
        targets,
        weight=None,
    ):

        loss = F.nll_loss(
            predictions,
            targets,
            weight=weight,
        )

        return loss