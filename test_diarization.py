from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

pipeline.to(torch.device("cuda"))

print("SUCCESS: diarization pipeline loaded")