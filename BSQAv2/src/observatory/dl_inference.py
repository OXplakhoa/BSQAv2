"""Deep Learning inference helpers for the Pipeline Observatory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from src.config import STROKE_TYPES
from src.models.gcn_bilstm_attn import GCNBiLSTMAttention

from .schema import ArtifactValidationError, PredictionResult


@dataclass
class DLModelBundle:
    """Loaded GCN+BiLSTM+Attention model and checkpoint metadata."""

    model: GCNBiLSTMAttention
    device: torch.device
    checkpoint_path: Path
    metadata: Dict


@dataclass
class DLInferenceResult:
    """Deep Learning inference output for one skeleton sequence."""

    prediction: PredictionResult
    attention_weights: np.ndarray
    quality_score: Optional[float]
    shape_metadata: Dict[str, list]


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ArtifactValidationError(
        "DL checkpoint must be a state_dict or contain 'model_state_dict'"
    )


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle DataParallel and training wrapper prefixes."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "inner."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _checkpoint_metadata(checkpoint) -> Dict:
    if not isinstance(checkpoint, dict):
        return {}
    metadata = {}
    for key in ("epoch", "val_loss", "val_metrics"):
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, np.generic):
                value = value.item()
            metadata[key] = value
    return metadata


def load_dl_model(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> DLModelBundle:
    """Load the GCN + BiLSTM + Attention checkpoint for inference."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DL checkpoint not found: {checkpoint_path}")

    device = torch.device(device)

    # Local project checkpoints are trusted artifacts. weights_only=False is
    # required for PyTorch 2.6+ when older checkpoints include numpy objects in
    # metric dictionaries.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _strip_known_prefixes(_extract_state_dict(checkpoint))

    model = GCNBiLSTMAttention().to(device)
    missing_keys = []
    unexpected_keys = []
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
    except RuntimeError as exc:
        raise ArtifactValidationError(
            f"DL checkpoint is incompatible with GCNBiLSTMAttention: {exc}"
        ) from exc

    allowed_missing = {"gcn.joint_attn_proj.weight", "gcn.joint_attn_proj.bias"}
    missing_set = set(missing_keys)
    if missing_set == allowed_missing:
        # Older checkpoints were trained before joint-attention pooling was added.
        # Switch the runtime GCN back to mean pooling so unused random joint-attn
        # parameters cannot affect predictions.
        model.gcn.pool = "mean"
    elif missing_keys:
        raise ArtifactValidationError(
            "DL checkpoint is missing required model weights: "
            f"{', '.join(missing_keys[:10])}"
        )
    if unexpected_keys:
        raise ArtifactValidationError(
            "DL checkpoint contains unexpected model weights: "
            f"{', '.join(unexpected_keys[:10])}"
        )
    model.eval()

    metadata = _checkpoint_metadata(checkpoint)
    metadata["gcn_pool"] = model.gcn.pool

    return DLModelBundle(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )


@torch.no_grad()
def run_dl_inference(
    bundle: DLModelBundle,
    normalized_keypoints: np.ndarray,
) -> DLInferenceResult:
    """Run DL inference on one normalized skeleton sequence."""
    normalized_keypoints = np.asarray(normalized_keypoints, dtype=np.float32)
    if normalized_keypoints.shape != (64, 17, 2):
        raise ArtifactValidationError(
            "DL inference expects normalized_keypoints shape (64, 17, 2); "
            f"got {normalized_keypoints.shape}"
        )

    x = torch.from_numpy(normalized_keypoints).unsqueeze(0).to(bundle.device)
    logits, quality, attention = bundle.model(x, return_attention=True)
    probabilities_array = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    predicted_index = int(np.argmax(probabilities_array))
    probabilities = {
        class_name: float(probabilities_array[idx])
        for idx, class_name in enumerate(STROKE_TYPES)
    }

    attention_array = np.empty((0, 0), dtype=np.float32)
    if attention is not None:
        attention_array = attention.squeeze(0).detach().cpu().numpy().astype(np.float32)

    quality_value = None
    if quality is not None:
        quality_value = float(quality.squeeze(0).detach().cpu().item())

    return DLInferenceResult(
        prediction=PredictionResult.from_probabilities(
            probabilities,
            predicted_index=predicted_index,
        ),
        attention_weights=attention_array,
        quality_score=quality_value,
        shape_metadata={
            "input": list(x.shape),
            "logits": list(logits.shape),
            "attention_weights": list(attention.shape) if attention is not None else [],
        },
    )
