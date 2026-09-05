#!/usr/bin/env python3
"""gen_model_manifest.py —— 从本地模型目录生成 App 端 ModelManifest.json

产出：ios/Sources/HuameiIndexTTS/Resources/ModelManifest.json
结构：
  {
    "modelscope": "ms-65b5df8b-d073-4c56-a508-7f24ccb2619c",
    "base_url": "https://modelscope.cn/models/{id}/resolve/master/",
    "groups": { "synthesis": [...], "clone": [...] },
    "files": [ {"path": "...", "group": "...", "size": N, "sha256": "hex"} ]
  }

用法：python scripts/gen_model_manifest.py
（models/ 下的大文件会逐个算 sha256，耗时数分钟，建议后台运行）
"""
import hashlib, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "mlx-indextts2-2.5-8bit")
HF_CACHE = os.path.join(ROOT, "models", "hf_cache")
GOLDEN = os.path.join(ROOT, "scripts", "golden")
OUT = os.path.join(ROOT, "ios", "Sources", "HuameiIndexTTS", "Resources", "ModelManifest.json")

MS_MODEL_ID = "ms-65b5df8b-d073-4c56-a508-7f24ccb2619c"

# (本地相对路径, 模型目录内目标路径, 组)
ENTRIES = [
    # ---- synthesis：基础 TTS 必需 ----
    ("gpt.safetensors", "gpt.safetensors", "synthesis"),
    ("codec.safetensors", "codec.safetensors", "synthesis"),
    ("s2mel.safetensors", "s2mel.safetensors", "synthesis"),
    ("bigvgan.safetensors", "bigvgan.safetensors", "synthesis"),
    ("multilingual_zh_ja_yue_char_del.tiktoken", "multilingual_zh_ja_yue_char_del.tiktoken", "synthesis"),
    ("../../../ios/Sources/MLXIndexTTS2Core/Resources/specials.json", "specials.json", "synthesis"),
    ("specials.json", "specials.json", "synthesis"),
    ("feat1.json", "feat1.json", "synthesis"),
    ("feat2.json", "feat2.json", "synthesis"),
    ("langs.json", "langs.json", "synthesis"),
    ("config.yaml", "config.yaml", "synthesis"),
    # ---- clone：A2 设备端克隆组件 ----
    ("w2v-bert-2.0/model.safetensors", "w2v-bert-2.0/model.safetensors", "clone"),
    ("campplus_cn_common.safetensors", "campplus_cn_common.safetensors", "clone"),
    ("wav2vec2bert_stats.json", "wav2vec2bert_stats.json", "clone"),
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    files, total = [], 0
    for local, dest, group in ENTRIES:
        # 定位本地源文件（四选一：MODEL_DIR / HF_CACHE / GOLDEN / ios core Resources）
        bases = (MODEL_DIR, HF_CACHE, GOLDEN,
                 os.path.join(ROOT, "ios", "Sources", "MLXIndexTTS2Core", "Resources"))
        for base in bases:
            p = os.path.join(base, local)
            if os.path.isfile(p):
                break
        else:
            print(f"[warn] 跳过缺失文件: {local}")
            continue
        size = os.path.getsize(p)
        total += size
        print(f"[{size/1e6:8.1f}MB] sha256 {local} ...")
        files.append({"path": dest, "group": group, "size": size, "sha256": sha256(p)})
        print(f"  -> {files[-1]['sha256'][:16]}...")

    manifest = {
        "modelscope": MS_MODEL_ID,
        "base_url": f"https://modelscope.cn/models/{MS_MODEL_ID}/resolve/master/",
        "total_bytes": total,
        "files": files,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"\nDONE -> {OUT}")
    print(f"共 {len(files)} 个文件 / {total/1e9:.2f} GB")


if __name__ == "__main__":
    main()