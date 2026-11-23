import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print(f"torch.xpu.is_available(): {torch.xpu.is_available()}", flush=True)
device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"using device: {device}", flush=True)

model = ChatterboxMultilingualTTS.from_pretrained(device=device)

ref = "/work/ref_en_clip.wav"

lines = [
    ("pl", "Fry, doceniam poświęcenie, ale czy to nie przesada?", "line09"),
    ("pl", "Racja. Wybacz, że okazuję swą miłość.", "line10"),
]

for lang, text, name in lines:
    print(f"generating {name} ({lang}): {text}", flush=True)
    wav = model.generate(text, language_id=lang, audio_prompt_path=ref)
    out = f"/work/{name}_{lang}.wav"
    ta.save(out, wav, model.sr)
    print(f"wrote {out}", flush=True)

print("done", flush=True)
