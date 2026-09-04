"""
Export script for FoundationPose models (RefineNet and ScoreNet) to ONNX format.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from learning.models.refine_network import RefineNet
from learning.models.score_network import ScoreNetMultiPair

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Disable PyTorch multi-head attention fastpath for deterministic ONNX tracing
torch.backends.mha.set_fastpath_enabled(False)


class ScoreNetONNXWrapper(nn.Module):
    """
    PyTorch Module wrapper for exporting ScoreNetMultiPair pose candidate scoring network to ONNX format.

    Replaces PyTorch native nn.MultiheadAttention cross-candidate attention (att_cross) with an
    explicit linear projection and scaled dot-product self-attention implementation (_dynamic_attention).

    Attributes:
        model (ScoreNetMultiPair): Pre-trained underlying ScoreNet network instance.
        embed_dim (int): Cross-attention embedding dimension (512).
        num_heads (int): Number of multi-head attention heads (4).
        head_dim (int): Dimensionality per attention head (embed_dim // num_heads = 128).
        scale (float): Scaled dot-product normalization factor (1.0 / sqrt(head_dim)).
        qkv_weight (torch.Tensor): Weight tensor for combined Query, Key, and Value projections.
        qkv_bias (torch.Tensor): Bias tensor for combined Query, Key, and Value projections.
        out_weight (torch.Tensor): Output linear projection weight tensor.
        out_bias (torch.Tensor): Output linear projection bias tensor.
    """

    def __init__(self, model: ScoreNetMultiPair):
        """
        Initialize the ONNX wrapper, extracting weight and bias tensors from ScoreNetMultiPair.att_cross.

        Args:
            model: Pre-trained ScoreNetMultiPair model instance.
        """
        super().__init__()
        self.model = model

        # Extract attention dimensions from the underlying att_cross multi-head attention module
        mha = self.model.att_cross
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Extract QKV in-projection and output projection weights and biases
        self.qkv_weight = mha.in_proj_weight
        self.qkv_bias = mha.in_proj_bias
        self.out_weight = mha.out_proj.weight
        self.out_bias = mha.out_proj.bias

    def _dynamic_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head self-attention over candidate pose features.

        Args:
            x: Input candidate feature tensor of shape (L, 512).

        Returns:
            Attended candidate feature tensor of shape (L, 512).
        """

        # Linear in-projection for Query, Key, and Value: (L, 512) -> (L, 1536)
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape and permute into multi-head layouts: (num_heads, L, head_dim)
        q = q.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = k.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = v.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Scaled dot-product attention scores: (num_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted combination of Value vectors: (num_heads, L, head_dim) -> (L, 512)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.permute(1, 0, 2).reshape(-1, self.embed_dim)

        # Final linear out-projection: (L, 512) -> (L, 512)
        out = F.linear(attn_out, self.out_weight, self.out_bias)
        return out

    def forward(self, input_render_A: torch.Tensor, input_real_B: torch.Tensor) -> torch.Tensor:
        """
        Execute ScoreNet forward pass over rendered template and real observed crops.

        Args:
            input_render_A: Rendered candidate crop tensor of shape (L, C, H, W).
            input_real_B: Real observed crop tensor of shape (L, C, H, W).

        Returns:
            1D tensor of candidate score logits of shape (L,).
        """
        # Feature extraction via ScoreNetMultiPair.extract_feat: returns (L, 512)
        feats = self.model.extract_feat(input_render_A, input_real_B)

        # Self-attention over candidates using explicit matrix math
        x = self._dynamic_attention(feats)

        # Final linear projection: (L, 512) -> (L,)
        scores = self.model.linear(x).reshape(-1)

        return scores


def load_refine_net(cfg_path: Union[str, Path], ckpt_dir: Union[str, Path]) -> RefineNet:
    """
    Load a pre-trained RefineNet model from configuration and checkpoint files.

    Args:
        cfg_path: Path to YAML configuration file.
        ckpt_dir: Path to PyTorch model checkpoint (.pth).

    Returns:
        RefineNet: Loaded RefineNet model instance.
    """
    cfg = OmegaConf.load(str(cfg_path))
    model = RefineNet(cfg=cfg, c_in=cfg.get("c_in", 6))

    ckpt = torch.load(str(ckpt_dir), map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    return model


def export_refinenet(
    model: Union[RefineNet, nn.Module],
    onnx_path: Union[str, Path],
    device: str = "cpu",
    opset_version: int = 18,
) -> Path:
    """
    Export RefineNet model to ONNX format.

    Export Specifications:
        - ONNX Inputs:
            * "input_render_A": (B, 6, H, W) float32
            * "input_real_B": (B, 6, H, W) float32
        - ONNX Outputs:
            * "trans": (B, 3) float32
            * "rot": (B, 3) float32

    Args:
        model: RefineNet model instance.
        onnx_path: Output path for exported .onnx file.
        device: Target hardware device ('cpu' or 'cuda').
        opset_version: Target ONNX operator set version.

    Returns:
        Path: Path to exported ONNX model file.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    c_in = getattr(model, "c_in", 6)
    if hasattr(model, "cfg") and isinstance(model.cfg, dict):
        c_in = model.cfg.get("c_in", c_in)
        height, width = model.cfg.get("input_resize", [160, 160])
    else:
        height, width = 160, 160

    dummy_input_render_A = torch.randn(1, c_in, height, width, device=device, dtype=torch.float32)
    dummy_input_real_B = torch.randn(1, c_in, height, width, device=device, dtype=torch.float32)

    input_names = ["input_render_A", "input_real_B"]
    output_names = ["trans", "rot"]

    dynamic_axes = {
        "input_render_A": {0: "batch_size"},
        "input_real_B": {0: "batch_size"},
        "trans": {0: "batch_size"},
        "rot": {0: "batch_size"},
    }

    logger.info("Exporting RefineNet to %s (opset %d)...", onnx_path, opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_render_A, dummy_input_real_B),
            str(onnx_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    logger.info("RefineNet ONNX export successful: %s", onnx_path)
    return onnx_path


def load_score_net(cfg_path: Union[str, Path], ckpt_dir: Union[str, Path]) -> ScoreNetMultiPair:
    """
    Load a pre-trained ScoreNetMultiPair model from configuration and checkpoint files.

    Args:
        cfg_path: Path to YAML configuration file.
        ckpt_dir: Path to PyTorch model checkpoint (.pth).

    Returns:
        ScoreNetMultiPair: Loaded ScoreNetMultiPair model instance.
    """
    cfg = OmegaConf.load(str(cfg_path))
    model = ScoreNetMultiPair(cfg=cfg, c_in=cfg.get("c_in", 6))

    ckpt = torch.load(str(ckpt_dir), map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    return model


def export_scorenet(
    model: ScoreNetMultiPair,
    onnx_path: Union[str, Path],
    device: str = "cpu",
    opset_version: int = 18,
) -> Path:
    """
    Export ScoreNet model to ONNX format.

    Export Specifications:
        - ONNX Inputs:
            * "input_render_A": (B, 6, H, W) float32
            * "input_real_B": (B, 6, H, W) float32
        - ONNX Output:
            * "score_logit": (B,) float32

    Args:
        model: ScoreNetMultiPair model instance.
        onnx_path: Output path for exported .onnx file.
        device: Target hardware device ('cpu' or 'cuda').
        opset_version: Target ONNX operator set version.

    Returns:
        Path: Path to exported ONNX model file.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped_model = ScoreNetONNXWrapper(model).to(device)
    wrapped_model.eval()

    c_in = getattr(model, "c_in", 6)
    if hasattr(model, "cfg") and isinstance(model.cfg, dict):
        c_in = model.cfg.get("c_in", c_in)
        height, width = model.cfg.get("input_resize", [160, 160])
    else:
        height, width = 160, 160

    num_poses = 5
    dummy_input_render_A = torch.randn(num_poses, c_in, height, width, device=device, dtype=torch.float32)
    dummy_input_real_B = torch.randn(num_poses, c_in, height, width, device=device, dtype=torch.float32)

    input_names = ["input_render_A", "input_real_B"]
    output_names = ["score_logit"]

    dynamic_axes = {
        "input_render_A": {0: "batch_size"},
        "input_real_B": {0: "batch_size"},
        "score_logit": {0: "batch_size"},
    }

    logger.info("Exporting ScoreNet to %s (opset %d)...", onnx_path, opset_version)
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_input_render_A, dummy_input_real_B),
            str(onnx_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    logger.info("ScoreNet ONNX export successful: %s", onnx_path)
    return onnx_path


def validate_onnx(onnx_path: Union[str, Path]) -> bool:
    """
    Validate exported ONNX model graph structure.

    Args:
        onnx_path: Path to exported .onnx model file.

    Returns:
        bool: True if graph validation passes.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")

    logger.info("Checking ONNX model integrity at %s...", onnx_path)
    model = onnx.load(str(onnx_path))

    try:
        onnx.checker.check_model(model)
        logger.info("ONNX graph validation passed cleanly: %s", onnx_path.name)
        return True
    except onnx.checker.ValidationError as err:
        logger.error("ONNX graph validation failed for %s: %s", onnx_path.name, err)
        return False


def main() -> None:
    """CLI entrypoint for exporting RefineNet and ScoreNet models to ONNX."""
    parser = argparse.ArgumentParser(description="Export FoundationPose (RefineNet & ScoreNet) models to ONNX format.")
    parser.add_argument("--weights-dir", type=str, default="./weights", help="Directory containing model checkpoint folders.")
    parser.add_argument("--refine-run-name", type=str, default="2023-10-28-18-33-37", help="RefineNet checkpoint directory name.")
    parser.add_argument("--score-run-name", type=str, default="2024-01-11-20-02-45", help="ScoreNet checkpoint directory name.")
    parser.add_argument("--output-folder", type=str, default="./weights/onnx_output", help="Destination folder for exported .onnx files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target hardware device ('cpu' or 'cuda').")
    parser.add_argument("--opset", type=int, default=18, help="Target ONNX operator set version (default: 18).")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_folder)

    # Export & Validate RefineNet
    refinenet_model_path = weights_dir / args.refine_run_name / "model_best.pth"
    refinenet_config_path = weights_dir / args.refine_run_name / "config.yml"
    refinenet_onnx_path = output_dir / "refine_net.onnx"

    if refinenet_model_path.exists() and refinenet_config_path.exists():
        logger.info("Loading RefineNet model...")
        refine_model = load_refine_net(refinenet_config_path, refinenet_model_path)
        refine_onnx = export_refinenet(model=refine_model, onnx_path=refinenet_onnx_path, device=args.device, opset_version=args.opset)
        validate_onnx(refine_onnx)
    else:
        logger.warning("RefineNet weights or config not found at %s. Skipping RefineNet export.", refinenet_model_path)

    # Export & Validate ScoreNet
    scorenet_model_path = weights_dir / args.score_run_name / "model_best.pth"
    scorenet_config_path = weights_dir / args.score_run_name / "config.yml"
    scorenet_onnx_path = output_dir / "score_net.onnx"

    if scorenet_model_path.exists() and scorenet_config_path.exists():
        logger.info("Loading ScoreNet model...")
        score_model = load_score_net(scorenet_config_path, scorenet_model_path)
        score_onnx = export_scorenet(model=score_model, onnx_path=scorenet_onnx_path, device=args.device, opset_version=args.opset)
        validate_onnx(score_onnx)
    else:
        logger.warning("ScoreNet weights or config not found at %s. Skipping ScoreNet export.", scorenet_model_path)


if __name__ == "__main__":
    main()
