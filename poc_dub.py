import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print(f"torch.xpu.is_available(): {torch.xpu.is_available()}", flush=True)
device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"using device: {device}", flush=True)

# chatterbox-tts's from_local() only special-cases map_location for "cpu"/"mps";
# for any other device string it passes map_location=None, so torch.load tries
# to deserialize the CUDA-pickled checkpoint straight onto a literal CUDA
# device and fails with no GPU. Load on CPU (which chatterbox handles
# correctly), then move submodules onto the real target device ourselves.
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
if device != "cpu":
    model.ve.to(device)
    model.t3.to(device)
    model.s3gen.to(device)
    model.device = device
    if model.conds is not None:
        model.conds = model.conds.to(device)

ref = "/work/ref_en_clip.wav"

lines = [
    ("pl", "Fry, doceniam poświęcenie, ale czy to nie przesada?", "line09"),
    ("pl", "Racja. Wybacz, że okazuję swą miłość.", "line10"),
]

for lang, text, name in lines:
    print(f"generating {name} ({lang}): {text}", flush=True)
    wav = model.generate(text, language_id=lang, audio_prompt_path=ref)
    out = f"/work/{name}_{lang}.wav"
    sf.write(out, wav.squeeze().cpu().numpy(), model.sr)
    print(f"wrote {out}", flush=True)

print("done", flush=True)
