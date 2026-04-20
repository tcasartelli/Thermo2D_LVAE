"""LVAE autoencoder classes with validation-aware pruning.

Extends the base classes in engiopt.vanilla_lvae.aes to add validation
pruning checks. Before confirming the pruning of a latent dimension,
the std on the validation set is checked — dimensions that are still
active on val are protected from pruning (avoid over-pruning).

Classes:
    - LeastVolumeAE_DynamicPruning_ValPruning
    - ConstrainedLeastVolumeAE_DP_ValPruning
    - ConstrainedPerfLeastVolumeAE_DP_ValPruning
"""

from __future__ import annotations

import torch
from engiopt.vanilla_lvae.aes import (
    LeastVolumeAE_DynamicPruning,
    ConstrainedLeastVolumeAE_DP,
    ConstrainedPerfLeastVolumeAE_DP,
)


class LeastVolumeAE_DynamicPruning_ValPruning(LeastVolumeAE_DynamicPruning):  # noqa: N801
    """LeastVolumeAE_DynamicPruning with validation-aware pruning.

    Before pruning a dimension, checks that it also has low variance
    on the validation set. Dimensions with high val std are protected.

    Extra args (passed to epoch_report):
        val_z: Latent codes computed on the full validation set (N_val, latent_dim).
               If None, falls back to standard pruning.
    """

    @torch.no_grad()
    def _prune_step(self, epoch: int, val_z: torch.Tensor | None = None) -> None:
        """Pruning step with optional validation check."""
        if self._zstd is None or self._zmean is None:
            return

        z_std_active = self._zstd[~self._p]
        if len(z_std_active) == 0:
            return

        # Compute pruning candidates on train (same as original)
        if self.pruning_strategy == "lognorm":
            cand_active = self._lognorm_prune(z_std_active)
        else:
            cand_active = self._plummet_prune(z_std_active)

        # Map back to full latent space
        cand = torch.zeros_like(self._p, dtype=torch.bool)
        cand[~self._p] = cand_active

        # --- VALIDATION CHECK ---
        # Proteggi le dimensioni che hanno ancora varianza alta sul val
        if val_z is not None:
            val_std = val_z.std(0)  # (latent_dim,)
            # Threshold: stessa proporzione usata sul train
            val_threshold = self.pruning_threshold * val_std.max()
            val_important = val_std > val_threshold
            # Rimuovi dai candidati le dimensioni ancora importanti sul val
            cand = cand & (~val_important)

        prune_idx = torch.where(cand & (~self._p))[0]
        if len(prune_idx) == 0:
            return

        # Freeze std e commit pruning (identico all'originale)
        self._frozen_std[prune_idx] = self._zstd[prune_idx].clone()
        self._p[prune_idx] = True
        self._z[prune_idx] = self._zmean[prune_idx]

    def epoch_report(self, epoch: int, callbacks, val_z: torch.Tensor | None = None, **kwargs) -> None:
        """Epoch report with val_z passed to pruning step."""
        if epoch == self.pruning_epoch and self.pruning_strategy == "lognorm" and self._zstd is not None:
            self._set_lognorm_reference(self._zstd)

        if epoch >= self.pruning_epoch:
            self._prune_step(epoch, val_z=val_z)

        # Skip parent's epoch_report pruning (already done above)
        # Call grandparent directly
        from engiopt.vanilla_lvae.aes import LeastVolumeAE
        LeastVolumeAE.epoch_report(self, epoch=epoch, callbacks=callbacks, **kwargs)


class ConstrainedLeastVolumeAE_DP_ValPruning(ConstrainedLeastVolumeAE_DP):  # noqa: N801
    """ConstrainedLeastVolumeAE_DP with validation-aware pruning."""

    @torch.no_grad()
    def _prune_step(self, epoch: int, val_z: torch.Tensor | None = None) -> None:
        """Pruning step with optional validation check."""
        if self._zstd is None or self._zmean is None:
            return

        z_std_active = self._zstd[~self._p]
        if len(z_std_active) == 0:
            return

        if self.pruning_strategy == "lognorm":
            cand_active = self._lognorm_prune(z_std_active)
        else:
            cand_active = self._plummet_prune(z_std_active)

        cand = torch.zeros_like(self._p, dtype=torch.bool)
        cand[~self._p] = cand_active

        if val_z is not None:
            val_std = val_z.std(0)
            val_threshold = self.pruning_threshold * val_std.max()
            val_important = val_std > val_threshold
            cand = cand & (~val_important)

        prune_idx = torch.where(cand & (~self._p))[0]
        if len(prune_idx) == 0:
            return

        self._frozen_std[prune_idx] = self._zstd[prune_idx].clone()
        self._p[prune_idx] = True
        self._z[prune_idx] = self._zmean[prune_idx]

    def epoch_report(self, epoch: int, callbacks, val_z: torch.Tensor | None = None, **kwargs) -> None:
        """Epoch report with val_z passed to pruning step."""
        if epoch == self.pruning_epoch and self.pruning_strategy == "lognorm" and self._zstd is not None:
            self._set_lognorm_reference(self._zstd)

        if epoch >= self.pruning_epoch:
            self._prune_step(epoch, val_z=val_z)

        from engiopt.vanilla_lvae.aes import LeastVolumeAE
        LeastVolumeAE.epoch_report(self, epoch=epoch, callbacks=callbacks, **kwargs)


class ConstrainedPerfLeastVolumeAE_DP_ValPruning(ConstrainedPerfLeastVolumeAE_DP):  # noqa: N801
    """ConstrainedPerfLeastVolumeAE_DP with validation-aware pruning."""

    @torch.no_grad()
    def _prune_step(self, epoch: int, val_z: torch.Tensor | None = None) -> None:
        """Pruning step with optional validation check."""
        if self._zstd is None or self._zmean is None:
            return

        z_std_active = self._zstd[~self._p]
        if len(z_std_active) == 0:
            return

        if self.pruning_strategy == "lognorm":
            cand_active = self._lognorm_prune(z_std_active)
        else:
            cand_active = self._plummet_prune(z_std_active)

        cand = torch.zeros_like(self._p, dtype=torch.bool)
        cand[~self._p] = cand_active

        if val_z is not None:
            val_std = val_z.std(0)
            val_threshold = self.pruning_threshold * val_std.max()
            val_important = val_std > val_threshold
            cand = cand & (~val_important)

        prune_idx = torch.where(cand & (~self._p))[0]
        if len(prune_idx) == 0:
            return

        self._frozen_std[prune_idx] = self._zstd[prune_idx].clone()
        self._p[prune_idx] = True
        self._z[prune_idx] = self._zmean[prune_idx]

    def epoch_report(self, epoch: int, callbacks, val_z: torch.Tensor | None = None, **kwargs) -> None:
        """Epoch report with val_z passed to pruning step."""
        if epoch == self.pruning_epoch and self.pruning_strategy == "lognorm" and self._zstd is not None:
            self._set_lognorm_reference(self._zstd)

        if epoch >= self.pruning_epoch:
            self._prune_step(epoch, val_z=val_z)

        from engiopt.vanilla_lvae.aes import LeastVolumeAE
        LeastVolumeAE.epoch_report(self, epoch=epoch, callbacks=callbacks, **kwargs)


__all__ = [
    "LeastVolumeAE_DynamicPruning_ValPruning",
    "ConstrainedLeastVolumeAE_DP_ValPruning",
    "ConstrainedPerfLeastVolumeAE_DP_ValPruning",
]


# ============================================================
# PERCHE' QUESTO FILE ESISTE
# ============================================================
# Nel pruning standard (aes.py), una dimensione latente viene
# eliminata se ha varianza bassa sul TRAINING set.
#
# Il problema: quella dimensione potrebbe avere varianza bassa
# sul train ma essere ancora utile sul VALIDATION set
# (che ha configurazioni diverse di boundary conditions, carichi, ecc.)
# -> risultato: over-pruning, il modello perde dimensioni utili
#    per generalizzare e le performance sul val peggiorano.
#
# COSA CAMBIA QUI:
# Prima di eliminare una dimensione, controlliamo anche la sua
# varianza sul validation set. Se sul val quella dimensione e'
# ancora "attiva" (varianza alta), la proteggiamo e non la
# eliminiamo. Il pruning avviene solo se la dimensione e'
# inutile SIA sul train CHE sul val.
#
# In pratica: passiamo val_z (i latenti del val set) a
# epoch_report() -> _prune_step(), che usa val_z per
# decidere quali dimensioni e' sicuro eliminare.
# ============================================================