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
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from lana.config import AudioConfig, Config
from lana.training_codec import TrainableCodec
from lana.training_config import TrainingConfig
from lana.training_data import MelDataModule, TrainingBatch


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average channel-wise errors over valid frames, excluding padded targets."""

    numerator = (values * mask).sum()
    denominator = mask.sum() * values.shape[1]

    return numerator / denominator.clamp_min(1.0)


class CodecTrainingModule(L.LightningModule):
    """Train the codec with quantized Mel reconstruction and temporal losses.

    Checkpoints contain architecture, optimizer settings, and frontend metadata.
    The external DSPGAN models are neither loaded nor optimized during training.
    """

    def __init__(
        self,
        configuration: dict[str, Any],
        audio: dict[str, Any],
    ) -> None:
        """Reconstruct validated settings from checkpoint-serializable mappings."""

        super().__init__()
        self.config = TrainingConfig.model_validate(configuration)
        self.audio = AudioConfig.model_validate(audio)
        self.codec = TrainableCodec(self.config.architecture)

        self.save_hyperparameters(
            {
                "configuration": self.config.model_dump(mode="json"),
                "audio": self.audio.model_dump(mode="json"),
            }
        )

    def forward(self, mel: Tensor, prompt: Tensor) -> Tensor:
        """Reconstruct normalized Mel segments through straight-through FSQ."""

        return self.codec(mel, prompt)

    def _losses(self, batch: TrainingBatch) -> dict[str, Tensor]:
        """Calculate float32 losses in the frontend's normalized [-1, 1] domain."""

        prediction = self(batch["mel"], batch["prompt"]).float() * 2 - 1
        target = batch["mel"].float() * 2 - 1
        mask = batch["mask"].float()

        reconstruction = _masked_mean((prediction - target).abs(), mask)

        predicted_difference = prediction[:, :, 1:] - prediction[:, :, :-1]
        target_difference = target[:, :, 1:] - target[:, :, :-1]
        difference_mask = mask[:, :, 1:] * mask[:, :, :-1]
        temporal = _masked_mean(
            (predicted_difference - target_difference).abs(),
            difference_mask,
        )

        config = self.config.optimization
        total = (
            config.reconstruction_weight * reconstruction
            + config.temporal_weight * temporal
        )

        return {
            "loss": total,
            "reconstruction": reconstruction,
            "temporal": temporal,
        }

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Optimize one crop batch and log local steps plus distributed epoch means."""

        losses = self._losses(batch)
        self.log_dict(
            {f"train/{name}": value for name, value in losses.items()},
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["mel"].shape[0],
            prog_bar=False,
        )

        return losses["loss"]

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> None:
        """Evaluate fixed crops and synchronize checkpoint-selection metrics."""

        losses = self._losses(batch)
        self.log_dict(
            {f"val/{name}": value for name, value in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["mel"].shape[0],
            prog_bar=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Use AdamW with optimizer-step warm-up followed by cosine decay.

        Lightning's estimated step count includes gradient accumulation and the
        configured training limit. Scheduling advances only on optimizer steps.
        """

        config = self.config.optimization
        optimizer = AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(config.warmup_steps, max(0, total_steps - 1))

        def multiplier(step: int) -> float:
            """Return the learning-rate multiplier for an optimizer step."""

            if step < warmup_steps:
                return (step + 1) / warmup_steps

            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return config.minimum_lr_ratio + (1 - config.minimum_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=multiplier)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject resume settings that would silently alter the training problem.

        Runtime and output locations may change. The data split, architecture,
        frontend, and optimizer definition must match the saved run.
        """

        saved = checkpoint["hyper_parameters"]
        previous = TrainingConfig.model_validate(saved["configuration"])
        previous_audio = AudioConfig.model_validate(saved["audio"])

        if previous.architecture != self.config.architecture:
            raise ValueError("Checkpoint architecture differs from configuration.")
        if previous.optimization != self.config.optimization:
            raise ValueError(
                "Checkpoint optimization settings differ from configuration."
            )
        if previous.data != self.config.data or previous.seed != self.config.seed:
            raise ValueError("Checkpoint dataset or seed differs from configuration.")
        if previous_audio != self.audio:
            raise ValueError("Checkpoint frontend differs from configuration.")

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Stop before applying non-finite gradients instead of corrupting weights.

        Mixed-precision FP16 overflow is managed by the precision plugin's
        gradient scaler. Full precision and BF16 failures are surfaced directly.
        """

        if self.config.runtime.precision == "16-mixed":
            return

        gradients = [
            parameter.grad
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            finite = torch.stack(
                [torch.isfinite(gradient).all() for gradient in gradients]
            ).all()
            if not bool(finite):
                raise FloatingPointError("Training produced non-finite gradients.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON artifact after serializing its complete contents."""

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
            json.dump(payload, stream, indent=2, ensure_ascii=True)
            stream.write("\n")

        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_codec(
    module: CodecTrainingModule,
    inference: Config,
    directory: Path,
) -> Path:
    """Export an inference-compatible CPU TorchScript artifact and configuration.

    A fresh module avoids changing the device or dtype of an active trainer.
    Quantizer metadata describes fixed-width content coding, not NPY storage or
    transport overhead. Speaker conditioning is still sent separately.
    """

    directory.mkdir(parents=True, exist_ok=True)
    architecture = module.config.architecture

    codec = TrainableCodec(architecture)
    state = {
        name: value.detach().cpu() for name, value in module.codec.state_dict().items()
    }
    codec.load_state_dict(state, strict=True)
    codec.eval()
    scripted = torch.jit.script(codec)

    token_rate = (
        inference.audio.sample_rate
        / inference.audio.hop_size
        / architecture.frame_stride
    )
    metadata = {
        "format_version": 1,
        "model_type": "lana-fsq-mel",
        "architecture": architecture.model_dump(mode="json"),
        "audio": inference.audio.model_dump(mode="json"),
        "content_codebook_size": architecture.content_codebook_size,
        "speaker_codebook_size": architecture.speaker_codebook_size,
        "content_index_bits": architecture.content_bits,
        "content_tokens_per_second": token_rate,
        "fixed_width_content_bps": token_rate * architecture.content_bits,
        "content_dtype": "int64",
        "speaker_dtype": "int64",
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

    models = inference.models.model_copy(update={"codec": destination.resolve()})
    encoding = inference.encoding.model_copy(
        update={"prompt_length": module.config.data.prompt_frames}
    )
    chunking = inference.inference.model_copy(
        update={"prompt_length": module.config.data.prompt_frames}
    )
    exported_config = inference.model_copy(
        update={
            "models": models,
            "encoding": encoding,
            "inference": chunking,
        }
    )

    _write_json(directory / "codec.json", metadata)
    _write_json(
        directory / "inference.json",
        exported_config.model_dump(mode="json"),
    )

    return destination


def train(
    config_path: str | Path = "training.json",
    *,
    resume: str | Path | None = None,
) -> None:
    """Train a new codec or resume a trusted Lightning checkpoint.

    Successful training exports the best validation checkpoint. Optimizer,
    scheduler, scaler, and loop state are restored by Lightning when resuming;
    random crop replay across interruptions is not guaranteed to be bitwise
    identical.
    """

    config = TrainingConfig.from_file(config_path)
    inference = config.load_inference_config()
    runtime = config.runtime

    resume_path = None if resume is None else Path(resume).expanduser().resolve()
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision(runtime.matmul_precision)

    data = MelDataModule(config, inference.audio)
    module = CodecTrainingModule(
        configuration=config.model_dump(mode="json"),
        audio=inference.audio.model_dump(mode="json"),
    )

    logger = CSVLogger(
        save_dir=str(config.output_directory),
        name="logs",
    )
    checkpoint = ModelCheckpoint(
        dirpath=config.output_directory / "checkpoints",
        filename="epoch={epoch:04d}-step={step}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=runtime.save_top_k,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    trainer = L.Trainer(
        default_root_dir=config.output_directory,
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        strategy="auto",
        precision=runtime.precision,
        max_epochs=runtime.max_epochs,
        max_steps=runtime.max_steps,
        accumulate_grad_batches=runtime.accumulate_grad_batches,
        gradient_clip_val=runtime.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=runtime.log_every_n_steps,
        deterministic=runtime.deterministic,
        benchmark=False,
        logger=logger,
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        num_sanity_val_steps=2,
    )

    if trainer.is_global_zero:
        config.output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            config.output_directory / "training.json",
            config.model_dump(mode="json"),
        )
        architecture = config.architecture
        content_bps = (
            inference.audio.sample_rate
            / inference.audio.hop_size
            / architecture.frame_stride
            * architecture.content_bits
        )
        print(
            f"Content codebook: {architecture.content_codebook_size} entries, "
            f"{architecture.content_bits} bits/index, "
            f"{content_bps:.2f} fixed-width content bps."
        )

    trainer.fit(
        module,
        datamodule=data,
        ckpt_path=None if resume_path is None else str(resume_path),
    )

    if trainer.is_global_zero and not trainer.interrupted:
        best_path = checkpoint.best_model_path
        if best_path:
            best = CodecTrainingModule.load_from_checkpoint(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
            destination = export_codec(
                best,
                inference,
                config.output_directory / "export",
            )
            print(f"Exported codec: {destination}")
        else:
            print("No validated checkpoint is available; export was skipped.")

    trainer.strategy.barrier()
