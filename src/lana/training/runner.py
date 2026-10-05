import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from lana.codec.artifact import FORMAT_VERSION, MODEL_TYPE, signal_signature
from lana.codec.model import ScalableCodec
from lana.config import Config
from lana.training.data import MelDataModule, TrainingBatch


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average channel errors over valid target frames."""

    denominator = mask.sum() * values.shape[1]

    return (values * mask).sum() / denominator.clamp_min(1.0)


class CodecTrainingModule(L.LightningModule):
    """Train every residual prefix using the deployed block-conditioning rules.

    A uniformly sampled prefix supplies the primary training objective.
    Non-base prefixes optionally receive an additional base-only objective.
    Validation evaluates every supported prefix independently.

    Previous smoothed embeddings are detached between blocks. This preserves
    forward conditioning while truncating its cross-block gradient history.
    """

    def __init__(self, configuration: dict[str, Any]) -> None:
        """Restore validated configuration from checkpoint-serializable values."""

        super().__init__()
        self.config = Config.model_validate(configuration)
        self.settings = self.config.validate_training()
        self.codec = ScalableCodec(
            self.config.audio.n_mels,
            self.settings.architecture,
        )
        self.save_hyperparameters(
            {"configuration": self.config.model_dump(mode="json")}
        )

    def forward(self, mel: Tensor, layers: int) -> Tensor:
        """Reconstruct consecutive blocks using bounded prompt and speaker state."""

        settings = self.config.streaming
        prompt: Tensor | None = None
        embedding: Tensor | None = None
        reconstructed: list[Tensor] = []

        for chunk in mel.split(settings.chunk_frames, dim=-1):
            prompt = chunk if prompt is None else torch.cat((prompt, chunk), dim=-1)
            prompt = prompt[:, :, -settings.prompt_frames :]
            current = self.codec.ref_encoder(prompt)
            if embedding is None:
                embedding = current
            else:
                weight = settings.new_embedding_weight
                embedding = embedding.detach() * (1.0 - weight) + current * weight

            reconstructed.append(self.codec.reconstruct(chunk, embedding, layers))

        return torch.cat(reconstructed, dim=-1)

    def _losses(self, batch: TrainingBatch, layers: int) -> dict[str, Tensor]:
        """Compute masked Mel and multi-scale temporal losses in float32."""

        prediction = self(batch["mel"], layers).float() * 2.0 - 1.0
        target = batch["mel"].float() * 2.0 - 1.0
        mask = batch["mask"].float()
        reconstruction = masked_mean((prediction - target).abs(), mask)

        differences: list[Tensor] = []
        for scale in self.settings.optimization.temporal_scales:
            predicted = prediction[:, :, scale:] - prediction[:, :, :-scale]
            expected = target[:, :, scale:] - target[:, :, :-scale]
            valid = mask[:, :, scale:] * mask[:, :, :-scale]
            differences.append(masked_mean((predicted - expected).abs(), valid))
        temporal = torch.stack(differences).mean()
        loss = reconstruction + self.settings.optimization.temporal_weight * temporal

        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "temporal": temporal,
        }

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Sample one bitrate prefix and optionally reinforce the base layer."""

        maximum = self.settings.architecture.num_quantizers
        layers = int(torch.randint(1, maximum + 1, ()).item())
        losses = self._losses(batch, layers)
        total = losses["loss"]
        base_weight = self.settings.optimization.base_loss_weight

        if layers > 1 and base_weight > 0:
            base = self._losses(batch, 1)
            total = total + base_weight * base["loss"]

        values = {f"train/{name}": value for name, value in losses.items()}
        values["train/objective"] = total
        self.log_dict(
            values,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["mel"].shape[0],
        )

        return total

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> None:
        """Evaluate all prefixes and select checkpoints by their mean loss."""

        losses: list[Tensor] = []
        metrics: dict[str, Tensor] = {}

        for layers in range(1, self.settings.architecture.num_quantizers + 1):
            values = self._losses(batch, layers)
            losses.append(values["loss"])
            for name, value in values.items():
                metrics[f"val/q{layers}/{name}"] = value

        metrics["val/loss"] = torch.stack(losses).mean()
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["mel"].shape[0],
            prog_bar=False,
        )
        self.log(
            "val/selection",
            metrics["val/loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["mel"].shape[0],
            prog_bar=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Schedule AdamW by optimizer steps, including gradient accumulation."""

        settings = self.settings.optimization
        total_steps = self.settings.runtime.max_steps
        optimizer = AdamW(
            self.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

        def multiplier(step: int) -> float:
            """Return warmup followed by bounded cosine decay."""

            if step < settings.warmup_steps:
                return (step + 1) / settings.warmup_steps

            progress = min(
                1.0,
                max(
                    0.0,
                    (step - settings.warmup_steps)
                    / max(1, total_steps - settings.warmup_steps),
                ),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return (
                settings.minimum_lr_ratio + (1.0 - settings.minimum_lr_ratio) * cosine
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda=multiplier),
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject resume changes affecting model semantics or schedule position."""

        previous = Config.model_validate(
            checkpoint["hyper_parameters"]["configuration"]
        )
        settings = previous.validate_training()
        if signal_signature(previous) != signal_signature(self.config):
            raise ValueError("Checkpoint signal configuration differs.")
        if settings.architecture != self.settings.architecture:
            raise ValueError("Checkpoint architecture differs.")
        if settings.optimization != self.settings.optimization:
            raise ValueError("Checkpoint optimization configuration differs.")
        if settings.data != self.settings.data or settings.seed != self.settings.seed:
            raise ValueError("Checkpoint dataset configuration differs.")
        if settings.runtime.max_steps != self.settings.runtime.max_steps:
            raise ValueError("Checkpoint learning-rate schedule length differs.")

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Reject non-finite full-precision or BF16 gradients before updating."""

        if self.settings.runtime.precision == "16-mixed":
            return

        checks = [
            torch.isfinite(parameter.grad).all()
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if checks and not bool(torch.stack(checks).all()):
            raise FloatingPointError("Training produced non-finite gradients.")


def write_json(path: Path, values: dict[str, Any]) -> None:
    """Atomically replace a JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(values, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_codec(module: CodecTrainingModule, directory: Path) -> Path:
    """Export all residual prefixes in one CPU TorchScript artifact.

    Export creates a separate model and does not mutate trainer parameters.
    The companion application configuration selects the desired runtime prefix.
    """

    directory.mkdir(parents=True, exist_ok=True)
    config = module.config
    architecture = module.settings.architecture
    codec = ScalableCodec(config.audio.n_mels, architecture)
    codec.load_state_dict(
        {
            name: value.detach().cpu()
            for name, value in module.codec.state_dict().items()
        },
        strict=True,
    )
    codec.eval()
    codec.set_num_quantizers(config.codec.num_quantizers)
    scripted = torch.jit.script(codec)

    metadata = {
        "format_version": FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "architecture": architecture.model_dump(mode="json"),
        "signal": signal_signature(config),
        "content_index_bits": architecture.content_bits,
        "content_tokens_per_second_per_stage": (
            config.audio.sample_rate / config.audio.hop_size / architecture.frame_stride
        ),
        "content_layout": "batch,positions,quantizers",
        "speaker_layout": "batch,1",
        "index_dtype": "int64",
    }

    destination = directory / "codec.pt"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=directory,
            prefix=".codec.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.jit.save(
            scripted,
            str(temporary),
            _extra_files={"lana.json": json.dumps(metadata, sort_keys=True)},
        )
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    exported = config.model_copy(
        update={
            "models": config.models.model_copy(update={"codec": destination.resolve()}),
            "training": None,
        }
    )
    write_json(directory / "config.json", exported.model_dump(mode="json"))
    write_json(directory / "codec.json", metadata)

    return destination


def train(
    config_path: str | Path = "config.json",
    *,
    resume: str | Path | None = None,
) -> None:
    """Train all bitrate prefixes and export the best validation checkpoint."""

    config = Config.from_file(config_path)
    settings = config.validate_training()
    runtime = settings.runtime
    resume_path = None if resume is None else Path(resume).expanduser().resolve()
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    L.seed_everything(settings.seed, workers=True)
    torch.set_float32_matmul_precision(runtime.matmul_precision)

    module = CodecTrainingModule(config.model_dump(mode="json"))
    data = MelDataModule(config)
    checkpoint = ModelCheckpoint(
        dirpath=settings.output_directory / "checkpoints",
        filename="epoch={epoch:04d}-step={step}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=runtime.save_top_k,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    device_count = (
        runtime.devices if isinstance(runtime.devices, int) else len(runtime.devices)
    )
    trainer = L.Trainer(
        default_root_dir=settings.output_directory,
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        strategy=(
            DDPStrategy(find_unused_parameters=True) if device_count > 1 else "auto"
        ),
        precision=runtime.precision,
        max_steps=runtime.max_steps,
        max_epochs=-1,
        accumulate_grad_batches=runtime.accumulate_grad_batches,
        gradient_clip_val=runtime.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=runtime.log_every_n_steps,
        check_val_every_n_epoch=runtime.validation_every_n_epochs,
        deterministic=runtime.deterministic,
        benchmark=False,
        logger=CSVLogger(str(settings.output_directory), name="logs"),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        num_sanity_val_steps=2,
    )

    if trainer.is_global_zero:
        write_json(
            settings.output_directory / "config.json",
            config.model_dump(mode="json"),
        )

    trainer.fit(
        module,
        datamodule=data,
        ckpt_path=None if resume_path is None else str(resume_path),
    )

    if trainer.is_global_zero and not trainer.interrupted:
        if checkpoint.best_model_path:
            best = CodecTrainingModule.load_from_checkpoint(
                checkpoint.best_model_path,
                map_location="cpu",
                weights_only=False,
            )
            destination = export_codec(
                best,
                settings.output_directory / "export",
            )
            print(f"Exported codec: {destination}")
        else:
            print("No validated checkpoint is available for export.")

    trainer.strategy.barrier()
