import requests, json

END_POINT = "http://127.0.0.1:5000/translator"  # for edit


headers1 = {"Content-Type": "application/json"}
json1 = [{"id": 100,
         "src": "This is a test."}]
try:
    response1 = requests.post(url=END_POINT+"/translate", headers=headers1, json=json1)
    print(response1.text)
    jstr = json.loads(response1.text)
    tgt = jstr[0][0]["tgt"]
    print(jstr)
    print(tgt)
except requests.exceptions.RequestException as e:
    print(f"请求失败, 错误信息：{e}")