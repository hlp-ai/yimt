import os
import re
import math
import torch
import pandas  as pd

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

from easyocr.recognition import NormalizePAD


def contrast_grey(img):
    """XXX: 不同于推理时的对比度计算"""
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/(high+low), high, low


def adjust_contrast_grey(img, target = 0.4):
    """调整对比度"""
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust = opt.contrast_adjust)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers), #prefetch_factor=2,persistent_workers=True,
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                # image, text = data_loader_iter.next()
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                # image, text = self.dataloader_iter_list[i].next()
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:  # 选择验证数据子集
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = OCRDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class OCRDataset(Dataset):

    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        print(root)
        self.examples = []
        with open(os.path.join(root,'labels.csv'), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                self.examples.append(parts)

        self.nSamples = len(self.examples)

        if self.opt.data_filtering_off:  # 不过滤数据
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
        else:  # 过滤数据
            self.filtered_index_list = []
            for index in range(self.nSamples):
                label = self.examples[index][1]
                try:
                    if len(label) > self.opt.batch_max_length:  # 长度过滤
                        continue
                except:
                    print(label)
                out_of_char = f'[^{self.opt.character}]'
                if re.search(out_of_char, label.lower()):  # 非字典字符过滤
                    continue
                self.filtered_index_list.append(index)
            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index = self.filtered_index_list[index]
        img_fname = self.examples[index][0]  # 图像文件
        img_fpath = os.path.join(self.root, img_fname)
        label = self.examples[index][1]  # 图像文本

        if self.opt.rgb:
            img = Image.open(img_fpath).convert('RGB')  # for color image
        else:
            img = Image.open(img_fpath).convert('L')

        if not self.opt.sensitive:  # 大小写不敏感
            label = label.lower()

        # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
        out_of_char = f'[^{self.opt.character}]'
        label = re.sub(out_of_char, '', label)  # 去掉非字典字符

        return (img, label)


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust = 0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)  # 过滤掉空数据
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size

                #### augmentation here - change contrast
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target = self.contrast_adjust)
                    image = Image.fromarray(image, 'L')

                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


if __name__ == "__main__":
    from easyocr.trainer.crnn.train_main import get_config

    opt = get_config("config_files/en_filtered_config.yaml")

    train_dataset = Batch_Balanced_Dataset(opt)
    image_tensors, labels = train_dataset.get_batch()

    print(image_tensors.shape)
    print(labels)
