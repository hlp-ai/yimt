from easyocr.recognition import get_recognizer, get_text
from easyocr.utils import group_text_box, get_image_list, get_paragraph, \
    diff, reformat_input, make_rotated_img_list, set_result_with_confidence, merge_to_free
from easyocr.config import *
from bidi.algorithm import get_display
import torch
import os
import sys
from logging import getLogger
import json
from pathlib import Path

LOGGER = getLogger(__name__)


class Reader(object):

    def __init__(self, lang_list, gpu=True,
                 model_storage_directory=None,
                 user_network_directory=None,
                 verbose=True,
                 quantize=True, cudnn_benchmark=False):
        """Create an EasyOCR Reader

        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.

            gpu (bool): Enable GPU support (default)

            model_storage_directory (string): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            user_network_directory (string): Path to directory for custom network architecture.
            If not specified, it is as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            download_enabled (bool): Enabled downloading of model data via HTTP (default).
        """
        self.verbose = verbose

        self.model_storage_directory = MODULE_PATH + '/model'
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        self.user_network_directory = MODULE_PATH + '/user_network'
        if user_network_directory:
            self.user_network_directory = user_network_directory
        Path(self.user_network_directory).mkdir(parents=True, exist_ok=True)
        sys.path.append(self.user_network_directory)

        if gpu is False:
            self.device = 'cpu'
            if verbose:
                LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
        elif gpu is True:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
                if verbose:
                    LOGGER.warning('Defaulting to CPU. Note: This module is much faster with a GPU.')
        else:
            self.device = gpu

        self.detection_models = detection_models
        self.recognition_models = recognition_models

        # check and download detection model
        self.quantize = quantize,
        self.cudnn_benchmark = cudnn_benchmark

        detector_path = self.getDetectorPath()

        recog_network = ""
        recog_model = None
        for model in recognition_models['gen1']:
            if set(lang_list).issubset(set(recognition_models['gen1'][model]["languages"])):
                recog_network = 'generation1'
                recog_model = recognition_models['gen1'][model]
                self.model_lang = recog_model['model_script']
                self.character = recog_model['characters']
                break

        for model in recognition_models['gen2']:
            if set(lang_list).issubset(set(recognition_models['gen2'][model]["languages"])):
                recog_network = 'generation2'
                recog_model = recognition_models['gen2'][model]
                self.model_lang = recog_model['model_script']
                self.character = recog_model['characters']
                break

        self.character = recog_model['characters']

        model_path = os.path.join(self.model_storage_directory, recog_model['filename'])
        print(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError("Missing %s" % model_path)

        self.detector = self.initDetector(detector_path)

        if recog_network == 'generation1':
            network_params = {
                'input_channel': 1,
                'output_channel': 512,
                'hidden_size': 512
            }
        elif recog_network == 'generation2':
            network_params = {
                'input_channel': 1,
                'output_channel': 256,
                'hidden_size': 256
            }
        # else:
        #     network_params = recog_config['network_params']
        self.recognizer, self.converter = get_recognizer(recog_network, network_params,
                                                         self.character,
                                                         model_path, device=self.device, quantize=quantize)

    def getDetectorPath(self):
        self.detect_network = 'craft'
        from .detection import get_detector, get_textbox
        self.get_textbox = get_textbox
        self.get_detector = get_detector
        detector_path = os.path.join(self.model_storage_directory,
                                     self.detection_models[self.detect_network]['filename'])
        if not os.path.isfile(detector_path):
            raise FileNotFoundError("Missing %s" % detector_path)

        return detector_path

    def initDetector(self, detector_path):
        return self.get_detector(detector_path,
                                 device=self.device,
                                 quantize=self.quantize,
                                 cudnn_benchmark=self.cudnn_benchmark
                                 )

    def detect(self, img, min_size=20, text_threshold=0.7, low_text=0.4,
               link_threshold=0.4, canvas_size=2560, mag_ratio=1.,
               slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
               width_ths=0.5, add_margin=0.1, reformat=True, optimal_num_chars=None,
               threshold=0.2, bbox_min_score=0.2, bbox_min_size=3,
               ):

        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box_list = self.get_textbox(self.detector,
                                         img,
                                         canvas_size=canvas_size,
                                         mag_ratio=mag_ratio,
                                         text_threshold=text_threshold,
                                         link_threshold=link_threshold,
                                         low_text=low_text,
                                         poly=False,
                                         device=self.device,
                                         optimal_num_chars=optimal_num_chars,
                                         threshold=threshold,
                                         bbox_min_score=bbox_min_score,
                                         bbox_min_size=bbox_min_size,
                                         )

        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,
                  decoder='greedy', beamWidth=5, batch_size=1,
                  workers=0, detail=1,
                  rotation_info=None, paragraph=False,
                  contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
                  y_ths=0.5, x_ths=1.0, reformat=True, output_format='standard'):

        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        ignore_char = ""

        if self.model_lang in ['chinese_tra', 'chinese_sim']: decoder = 'greedy'

        if (horizontal_list == None) and (free_list == None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        # without gpu/parallelization, it is faster to process image one by one
        if ((batch_size == 1) or (self.device == 'cpu')) and not rotation_info:
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height=imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,
                                   ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast,
                                   filter_ths,
                                   workers, self.device)
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height=imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,
                                   ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast,
                                   filter_ths,
                                   workers, self.device)
                result += result0
        # default mode will try to process multiple boxes at the same time
        else:
            image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height=imgH)
            image_len = len(image_list)
            if rotation_info and image_list:
                image_list = make_rotated_img_list(rotation_info, image_list)
                max_width = max(max_width, imgH)

            result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,
                              workers, self.device)

            if rotation_info and (horizontal_list + free_list):
                # Reshape result to be a list of lists, each row being for 
                # one of the rotations (first row being no rotation)
                result = set_result_with_confidence(
                    [result[image_len * i:image_len * (i + 1)] for i in range(len(rotation_info) + 1)])

        if self.model_lang == 'arabic':
            direction_mode = 'rtl'
            result = [list(item) for item in result]
            for item in result:
                item[1] = get_display(item[1])
        else:
            direction_mode = 'ltr'

        if paragraph:
            result = get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode=direction_mode)

        if detail == 0:
            return [item[1] for item in result]
        elif output_format == 'dict':
            if paragraph:
                return [{'boxes': item[0], 'text': item[1]} for item in result]
            return [{'boxes': item[0], 'text': item[1], 'confident': item[2]} for item in result]
        elif output_format == 'json':
            if paragraph:
                return [
                    json.dumps({'boxes': [list(map(int, lst)) for lst in item[0]], 'text': item[1]}, ensure_ascii=False)
                    for item in result]
            return [
                json.dumps({'boxes': [list(map(int, lst)) for lst in item[0]], 'text': item[1], 'confident': item[2]},
                           ensure_ascii=False) for item in result]
        elif output_format == 'free_merge':
            return merge_to_free(result, free_list)
        else:
            return result

    def readtext(self, image, decoder='greedy', beamWidth=5, batch_size=1,
                 workers=0, detail=1,
                 rotation_info=None, paragraph=False, min_size=20,
                 contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
                 text_threshold=0.7, low_text=0.4, link_threshold=0.4,
                 canvas_size=2560, mag_ratio=1.,
                 slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
                 width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1,
                 threshold=0.2, bbox_min_score=0.2, bbox_min_size=3,
                 output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img,
                                                 min_size=min_size, text_threshold=text_threshold,
                                                 low_text=low_text, link_threshold=link_threshold,
                                                 canvas_size=canvas_size, mag_ratio=mag_ratio,
                                                 slope_ths=slope_ths, ycenter_ths=ycenter_ths,
                                                 height_ths=height_ths, width_ths=width_ths,
                                                 add_margin=add_margin, reformat=False,
                                                 threshold=threshold, bbox_min_score=bbox_min_score,
                                                 bbox_min_size=bbox_min_size
                                                 )
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(img_cv_grey, horizontal_list, free_list,
                                decoder, beamWidth, batch_size,
                                workers, detail, rotation_info,
                                paragraph, contrast_ths, adjust_contrast,
                                filter_ths, y_ths, x_ths, False, output_format)

        return result
