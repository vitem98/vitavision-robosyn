"""PhaseLoRA layers for ACT.

Official PhaseLoRA (arXiv:2608.15285, github.com/Grinffin/PhaseLoRA) is a
π₀.₅ adapter. The parameterization here follows Eqs. (3)–(4):

    ΔW_t = B_t A,   B_t = B0 + P B_P + E B_E + P E B_PE

applied only to the ACT action expert (transformer + action head), with the
pretrained weights frozen. B_P / B_E / B_PE start at zero, so the policy is
identical to official ACT until training moves those factors.
"""

from __future__ import annotations

import torch
from torch import nn

_ROUTE_PE: torch.Tensor | None = None


def set_phase_route(p: torch.Tensor, e: torch.Tensor) -> None:
    global _ROUTE_PE
    _ROUTE_PE = torch.stack([p.reshape(-1), e.reshape(-1)], dim=-1)


def clear_phase_route() -> None:
    global _ROUTE_PE
    _ROUTE_PE = None


class PhaseLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_b0 = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_bp = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_be = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_bpe = nn.Linear(rank, base_linear.out_features, bias=False)
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_b0.weight)
        nn.init.zeros_(self.lora_bp.weight)
        nn.init.zeros_(self.lora_be.weight)
        nn.init.zeros_(self.lora_bpe.weight)

    def _broadcast(self, coeff: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        batch = coeff.shape[0]
        if reference.ndim == 2:
            return coeff[:, None]
        if reference.ndim == 3:
            if reference.shape[0] == batch:
                return coeff[:, None, None]
            if reference.shape[1] == batch:
                return coeff[None, :, None]
        raise ValueError(
            f"cannot broadcast PhaseLoRA coeffs of batch {batch} onto {tuple(reference.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        if _ROUTE_PE is None:
            return base
        route = _ROUTE_PE.to(device=x.device, dtype=base.dtype)
        p = route[:, 0]
        e = route[:, 1]
        pe = p * e
        a_out = self.lora_a(x)
        lora = self.lora_b0(a_out)
        lora = (
            lora
            + self._broadcast(p, lora) * self.lora_bp(a_out)
            + self._broadcast(e, lora) * self.lora_be(a_out)
            + self._broadcast(pe, lora) * self.lora_bpe(a_out)
        )
        return base + lora * self.scaling

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias


def inject_phaselora(act_module: nn.Module, rank: int, alpha: float) -> list[str]:
    """Replace Linear layers in the ACT action expert with PhaseLoRALinear."""
    replaced: list[str] = []

    def _walk(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            child_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, PhaseLoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, PhaseLoRALinear(child, rank, alpha))
                replaced.append(child_name)
            else:
                _walk(child, child_name)

    _walk(act_module.encoder, "encoder")
    _walk(act_module.decoder, "decoder")
    for attr in ("action_head", "encoder_robot_state_input_proj", "encoder_latent_input_proj"):
        child = getattr(act_module, attr, None)
        if isinstance(child, nn.Linear):
            setattr(act_module, attr, PhaseLoRALinear(child, rank, alpha))
            replaced.append(attr)
    return replaced


def adapter_tensors(policy) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().contiguous()
        for name, param in policy.named_parameters()
        if param.requires_grad
    }
