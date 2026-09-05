"""自研 safetensors 加载器（读 header + 按偏移取字节 → torch.Tensor）

本文件从零编写，只依赖 struct/json/torch，不依赖 safetensors 库。
格式参考 safetensors 公开规范：8字节小端 header 长度 + JSON header + 数据区。
"""
import json
import struct

import numpy as np
import torch

_TORCH_MISSING = {"U16": np.uint16, "U32": np.uint32, "U64": np.uint64}

_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
}


def read_header(path: str) -> dict:
    """读文件头，返回 (header_dict, data_start_offset)。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]          # header 长度（8 字节小端）
        header = json.loads(f.read(n))                 # JSON header
    return header, 8 + n


def load_tensors(path: str, fp16_to_fp32: bool = True,
                 names: list | None = None) -> dict[str, torch.Tensor]:
    """把 safetensors 全部（或指定 names）张量载入内存。

    约定：
    - F16/BF16 默认转 F32 便于 CPU 计算（numpy/torch 无 fp16 加速）。
    - GPT 的 U32 量化权重原样保留（不解包，由 QuantLinear 层处理）。
    """
    header, data_start = read_header(path)
    out: dict[str, torch.Tensor] = {}
    keys = [k for k in header if k != "__metadata__"]
    if names is not None:
        keys = [k for k in keys if k in names]
    with open(path, "rb") as f:
        for name in keys:
            info = header[name]
            dtype_name = info["dtype"]
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            if dtype_name in _TORCH_MISSING:
                # torch 无 uint16/uint32/uint64 → numpy 中转
                arr = np.frombuffer(raw, dtype=_TORCH_MISSING[dtype_name]).reshape(info["shape"])
                if dtype_name == "U32":
                    # 打包 8bit：每 uint32 存 4 个 uint8（小端，低字节=第 0 值）
                    # → [*shape, 4] uint8 → reshape 展开成 [O, I]
                    t = torch.from_numpy(
                        arr.astype(np.int32, copy=False).view(np.uint8)
                    ).reshape(*info["shape"][:-1], info["shape"][-1] * 4)
                else:
                    t = torch.from_numpy(arr.astype(np.int64))
            else:
                dtype = _DTYPES[dtype_name]
                t = torch.frombuffer(raw, dtype=dtype).reshape(info["shape"])
                if fp16_to_fp32 and dtype in (torch.float16, torch.bfloat16):
                    t = t.float()
            out[name] = t
    return out


def summary(path: str) -> dict:
    """打印/返回文件张量摘要，便于人工核对结构。"""
    header, _ = read_header(path)
    meta = header.get("__metadata__", {})
    total = {k: header[k]["shape"] for k in header if k != "__metadata__"}
    return {"metadata": meta, "n_tensors": len(total), "shapes": total}


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit/bigvgan.safetensors"
    s = summary(p)
    print("metadata:", s["metadata"])
    print(f"共 {s['n_tensors']} 个张量，样例：")
    for i, (k, v) in enumerate(list(s["shapes"].items())[:8]):
        print(f"  {k}  {v}")
