import os
import sys

import torch.backends.cudnn as cudnn
import yaml
from easyocr.trainer.crnn.train import train
from easyocr.trainer.crnn.utils import AttrDict
import pandas as pd


cudnn.benchmark = True
cudnn.deterministic = False


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)

    if opt.lang_char == 'None':  # 从文本标签中产生字符列表
        characters = ''
        for data in opt['select_data']:
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            all_char = ""
            with open(csv_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    parts = line.split("\t")
                    all_char += parts[1]

            # df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            # all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:  # 从配置产生字符列表
        opt.character = opt.number + opt.symbol + opt.lang_char

    for i, c in enumerate(opt.character):
        print(c, i)

    # 保存字典
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    with open(os.path.join(f'./saved_models/{opt.experiment_name}', "vocab.txt"), "w", encoding="utf-8") as vf:
        vf.write(opt.character)

    return opt


if __name__ == "__main__":
    conf_file = sys.argv[1]  # "config_files/en_filtered_config.yaml"
    opt = get_config(conf_file)
    train(opt, amp=False)

