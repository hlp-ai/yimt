import yaml

with open("./config.yaml", 'r', encoding="utf-8") as stream:
    conf = yaml.safe_load(stream)

detection_models = conf["detection_models"]
recognition_models = conf["recognition_models"]