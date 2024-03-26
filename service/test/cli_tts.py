import os
import sys
import tempfile

import requests, json
import base64
import numpy as np

from tts.interface import save_wav, play_wav

END_POINT = "http://127.0.0.1:5555"  # for edit
token = "api_key"  # api_key
lang = "en"
text = "This is a test for text to speech."

# print(encoded_image) # for test
headers1 = {"Content-Type": "application/json"}
json1 = {"text": text,
         "token": token,
         "lang": lang}

try:
    response1 = requests.post(url=END_POINT+"/text2speech", headers=headers1, json=json1)
    jstr = json.loads(response1.text)
    rate = jstr["rate"]
    base64_str = jstr["base64"]

    import base64
    audio = np.frombuffer(base64.b64decode(base64_str), dtype=np.float32)

    print("Saving wav...")
    tmp_wav_fn = os.path.join(tempfile.gettempdir(), str(hash(text)) + ".wav")
    save_wav(audio, tmp_wav_fn, rate)

    print("Playing wav...")
    play_wav(tmp_wav_fn)

except requests.exceptions.RequestException as e:
    print(f"请求失败, 错误信息：{e}")