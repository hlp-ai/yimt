import sys

import requests, json
import base64

END_POINT = "http://127.0.0.1:6666"  # for edit
token = "api_key"  # api_key
audio_file = sys.argv[1]

with open(audio_file, "rb") as f:    # 设置本地图片路径
    encoded_data = base64.b64encode(f.read())

# print(encoded_image) # for test
headers1 = {"Content-Type": "application/json"}
json1 = {"base64": encoded_data.decode('utf-8'),
         "token": token,
         }
try:
    response1 = requests.post(url=END_POINT+"/asr", headers=headers1, json=json1)
    print(response1.text)
    jstr = json.loads(response1.text)
except requests.exceptions.RequestException as e:
    print(f"请求失败, 错误信息：{e}")