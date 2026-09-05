"""P3b：端到端合成（GPT → Codec → LR → CFM → BigVGAN）"""
import os
import sys
import time
import wave

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import IndexTTS25, EMO_NAMES, SR  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_e2e.wav")


def save_wav(path: str, wav: torch.Tensor, sr: int = SR):
    x = wav.squeeze().detach().cpu().float()
    x = x / max(1e-8, x.abs().max()) * 0.95
    pcm = (x.clamp(-1, 1) * 32767).short().numpy()
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path


def main():
    print("=" * 64)
    print("IndexTTS-2.5 端到端（全部自研模块 / 真实 MLX 权重）")
    print("=" * 64)
    t0 = time.time()
    tts = IndexTTS25()
    print(f"\n模型加载总耗时 {time.time() - t0:.1f}s\n")

    # 条件束：style 取真实预设（第 0 组），prompt/ref_mel 为占位
    spk = tts.make_preset_condition(row=0, prompt_len=86)
    print(f"条件束: {spk.name}  style{tuple(spk.style.shape)}  "
          f"prompt{tuple(spk.prompt_condition.shape)}  ref_mel{tuple(spk.ref_mel.shape)}")

    # 情绪向量：纯 happy（Σw=1 → 官方 w2v-bert 分量被抵消）
    emo = [1.0, 0, 0, 0, 0, 0, 0, 0]
    print(f"情绪权重: {dict(zip(EMO_NAMES, emo))}\n")

    texts = ["你好，欢迎使用语音合成。"]
    for i, text in enumerate(texts, 1):
        print(f"--- 合成 {i}/{len(texts)}: {text!r} ---")
        t = time.time()
        wav, st = tts.synth(text, spk, lang="zh", emo_vector=emo,
                            max_mel_tokens=250, diffusion_steps=25)
        print(f"    文本 token     : {st['text_tokens']}")
        print(f"    mel codes      : {st['mel_codes']}  (≈{st['audio_sec_est']:.2f}s @25fps)")
        print(f"    mel 帧         : {st['mel_frames']}  @86fps")
        print(f"    耗时 GPT/Codec/LR/CFM/Vocoder : "
              f"{st['gpt_sec']:.1f} / {st['codec_sec']:.2f} / {st['lr_sec']:.2f} / "
              f"{st['cfm_sec']:.1f} / {st['vocoder_sec']:.1f} s")
        print(f"    总计 {time.time() - t:.1f}s → 音频 {st['wav_sec']:.2f}s "
              f"(RTF {(time.time() - t) / max(1e-6, st['wav_sec']):.1f}×)")
        print(f"    wav {tuple(wav.shape)}  finite={bool(torch.isfinite(wav).all())}  "
              f"std={wav.std():.4f}  peak={wav.abs().max():.4f}")
        save_wav(OUT, wav)
        print(f"    ✅ 已保存 → {OUT}")


if __name__ == "__main__":
    main()
