import os, shutil, time, sys

# 默认镜像，必须在 import 之前设好（切换端点时也会在循环里重设）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

ROOT = r"D:/indexTTS 2.5"
DST = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit"
os.makedirs(DST, exist_ok=True)

# 1) 把根目录已下的零散文件归位（已挪过的这次会自动跳过）
for f in ["bigvgan.safetensors", "codec.safetensors", "config.json", "config.yaml",
          "conversion_report.json", "feat1.pt", "feat2.pt"]:
    s = os.path.join(ROOT, f)
    if os.path.exists(s):
        shutil.move(s, os.path.join(DST, f))
        print("moved:", f)

# 2) 必需文件：这 4 个缺失了推理图就跑不起来
MUST = ["gpt.safetensors", "s2mel.safetensors",
        "multilingual_zh_ja_yue_char_del.tiktoken", "model_manifest.json"]

# 若设置了 token，则带鉴权下载（配额高很多，能绕过匿名限流）
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# 打印代理状态（Python 是否走了和浏览器一样的出口）
px = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
print("proxy:", px if px else "未设置(可能无法出网)")

# 先试国内镜像，再试官方 HF，最后试备用镜像
endpoints = ["https://hf-mirror.com", "https://huggingface.co", "https://hf-api.131.hk"]

for ep in endpoints:
    os.environ["HF_ENDPOINT"] = ep
    print("== 尝试端点:", ep, "==", "(token:", "yes" if token else "no", ")")
    ok = False
    for attempt in range(4):
        try:
            snapshot_download(
                repo_id="vanch007/mlx-indextts2-2.5-8bit",
                local_dir=DST,
                token=token,
            )
        except Exception as e:
            print(f"  尝试 {attempt+1} 失败: {e!r}")
            time.sleep(8)
            continue
        # 即使没抛异常也要校验——远端不可达时会原样返回本地目录
        missing = [m for m in MUST if not os.path.exists(os.path.join(DST, m))]
        if not missing:
            ok = True
            break
        else:
            print("  已返回但以下仍缺(换下一端点):", missing)
            break
    if ok:
        break

missing = [m for m in MUST if not os.path.exists(os.path.join(DST, m))]
if not missing:
    print("DONE: 全部必需文件已就位 ->", DST)
else:
    print("仍缺失必需文件:", missing)
    print("建议: 设置 HF_TOKEN 后重跑 ( $env:HF_TOKEN='hf_xxx' )，或检查网络/代理。")
    sys.exit(1)
