import sys

import requests, json
import base64

END_POINT = "http://127.0.0.1:6666"  # for edit
token = "api_key"  # api_key
img_file = sys.argv[1]
lang = sys.argv[2]

with open(img_file, "rb") as image_file:    # 设置本地图片路径
    encoded_image = base64.b64encode(image_file.read())

# print(encoded_image) # for test
headers1 = {"Content-Type": "application/json"}
json1 = {"base64": encoded_image.decode('utf-8'),
         "token": token,
         "lang": lang,
         }
try:
    response1 = requests.post(url=END_POINT+"/ocr", headers=headers1, json=json1)
    print(response1.text)
    obj = json.loads(response1.text)
    print(obj)
except requests.exceptions.RequestException as e:
    print(f"请求失败, 错误信息：{e}")