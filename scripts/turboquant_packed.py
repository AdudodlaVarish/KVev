from dataclasses import dataclass

import torch
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde import AsyncSerdeProcessor, register_serde_factory
from lmcache.v1.distributed.serde.turboquant import (
    TurboQuantDeserializer,
    TurboQuantSerdeConfig,
    TurboQuantSerializer,
)


def kv_view(tensor):
    if tensor.ndim != 3 or tensor.shape[-1] % 2:
        raise ValueError(f"Expected packed [layers, tokens, 2*hidden] KV, got {tuple(tensor.shape)}")
    layers, tokens, packed = tensor.shape
    return tensor.reshape(layers, tokens, 2, packed // 2).permute(2, 0, 1, 3)


@dataclass
class TensorView:
    tensor: torch.Tensor


class PackedSerializer(TurboQuantSerializer):
    def serialize(self, src, dst, key):
        return super().serialize(TensorView(kv_view(src.tensor)), dst, key)

    def estimate_serialized_size(self, layout_desc):
        shapes = []
        for shape in layout_desc.shapes:
            if len(shape) != 3 or shape[-1] % 2:
                raise ValueError(f"Expected packed KV layout, got {tuple(shape)}")
            shapes.append(torch.Size((2, shape[0], shape[1], shape[2] // 2)))
        return super().estimate_serialized_size(MemoryLayoutDesc(shapes, layout_desc.dtypes))


class PackedDeserializer(TurboQuantDeserializer):
    def deserialize(self, src, dst, key):
        return super().deserialize(src, TensorView(kv_view(dst.tensor)), key)


def create_packed_serde(kwargs):
    cfg = TurboQuantSerdeConfig(
        preset=str(kwargs.get("preset", "turboquant_k8v4")),
        head_dim=int(kwargs.get("head_dim", 64)),
        block_size=int(kwargs.get("block_size", 16)),
        skip_first_layers=int(kwargs.get("skip_first_layers", 2)),
        skip_last_layers=int(kwargs.get("skip_last_layers", 2)),
    )
    return AsyncSerdeProcessor(
        PackedSerializer(cfg),
        PackedDeserializer(cfg),
        max_workers=int(kwargs.get("max_workers", 1)),
    )


register_serde_factory("turboquant_packed", create_packed_serde)


def self_test():
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_4bit_nc",
        head_dim=64,
        block_size=16,
        skip_first_layers=0,
        skip_last_layers=0,
    )
    serializer = PackedSerializer(cfg)
    deserializer = PackedDeserializer(cfg)
    source = torch.randn(8, 16, 128, dtype=torch.float16)
    size = serializer.estimate_serialized_size(
        MemoryLayoutDesc([source.shape], [source.dtype])
    )
    encoded = TensorView(torch.empty(size, dtype=torch.uint8))
    written = serializer.serialize(TensorView(source), encoded, None)
    decoded = TensorView(torch.empty_like(source))
    deserializer.deserialize(TensorView(encoded.tensor[:written]), decoded, None)
    assert torch.isfinite(decoded.tensor).all()
    assert (source - decoded.tensor).abs().float().mean() < 0.3
    print(f"TurboQuant packed round trip passed: {source.numel() * 2} -> {written} bytes")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["self-test"]:
        self_test()
        raise SystemExit
    from lmcache.cli.main import main

    main()
