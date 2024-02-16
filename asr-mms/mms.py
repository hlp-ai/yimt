import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from datasets import load_dataset, Audio
#
# # English
# stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
# stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
# en_sample = next(iter(stream_data))["audio"]["array"]
#
# # French
# stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "fr", split="test", streaming=True)
# stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
# fr_sample = next(iter(stream_data))["audio"]["array"]


from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# 'joe keton disapproved of films and buster also had reservations about the media'


processor.tokenizer.set_target_lang("fra")
model.load_adapter("fra")

inputs = processor(fr_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# "ce dernier est volé tout au long de l'histoire romaine"


