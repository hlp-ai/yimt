# OCR

## Usage

``` python
import easyocr
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext('chinese.jpg')
```

The output will be in a list format, each item represents a bounding box, the text detected and confident level, respectively.

``` bash
[([[189, 75], [469, 75], [469, 165], [189, 165]], '愚园路', 0.3754989504814148),
 ([[86, 80], [134, 80], [134, 128], [86, 128]], '西', 0.40452659130096436),
 ([[517, 81], [565, 81], [565, 123], [517, 123]], '东', 0.9989598989486694),
 ([[78, 126], [136, 126], [136, 156], [78, 156]], '315', 0.8125889301300049),
 ([[514, 126], [574, 126], [574, 156], [514, 156]], '309', 0.4971577227115631),
 ([[226, 170], [414, 170], [414, 220], [226, 220]], 'Yuyuan Rd.', 0.8261902332305908),
 ([[79, 173], [125, 173], [125, 213], [79, 213]], 'W', 0.9848111271858215),
 ([[529, 173], [569, 173], [569, 213], [529, 213]], 'E', 0.8405593633651733)]
```
Note 1: `['ch_sim','en']` is the list of languages you want to read. You can pass
several languages at once but not all languages can be used together.
English is compatible with every language and languages that share common characters are usually compatible with each other.

Note 2: Instead of the filepath `chinese.jpg`, you can also pass an OpenCV image object (numpy array) or an image file as bytes. A URL to a raw image is also acceptable.

Note 3: The line `reader = easyocr.Reader(['ch_sim','en'])` is for loading a model into memory. It takes some time but it needs to be run only once.

You can also set `detail=0` for simpler output.

``` python
reader.readtext('chinese.jpg', detail = 0)
```
Result:
``` bash
['愚园路', '西', '东', '315', '309', 'Yuyuan Rd.', 'W', 'E']
```

Model weights for the chosen language will be automatically downloaded or you can
download them manually from the [model hub](https://www.jaided.ai/easyocr/modelhub) and put them in the '~/.EasyOCR/model' folder

In case you do not have a GPU, or your GPU has low memory, you can run the model in CPU-only mode by adding `gpu=False`.

``` python
reader = easyocr.Reader(['ch_sim','en'], gpu=False)
```

For more information, read the [tutorial](https://www.jaided.ai/easyocr/tutorial) and [API Documentation](https://www.jaided.ai/easyocr/documentation).

#### Run on command line

```shell
$ easyocr -l ch_sim en -f chinese.jpg --detail=1 --gpu=True
```

## Train/use your own model

For recognition model, [Read here](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md).

For detection model (CRAFT), [Read here](https://github.com/JaidedAI/EasyOCR/blob/master/trainer/craft/README.md).

## Implementation Roadmap

- Handwritten support
- Restructure code to support swappable detection and recognition algorithms
The api should be as easy as
``` python
reader = easyocr.Reader(['en'], detection='DB', recognition = 'Transformer')
```
The idea is to be able to plug in any state-of-the-art model into EasyOCR. There are a lot of geniuses trying to make better detection/recognition models, but we are not trying to be geniuses here. We just want to make their works quickly accessible to the public ... for free. (well, we believe most geniuses want their work to create a positive impact as fast/big as possible) The pipeline should be something like the below diagram. Grey slots are placeholders for changeable light blue modules.

![plan](examples/easyocr_framework.jpeg)


## Guideline for new language request

To request a new language, we need you to send a PR with the 2 following files:

1. In folder [easyocr/character](https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/character),
we need 'yourlanguagecode_char.txt' that contains list of all characters. Please see format examples from other files in that folder.
2. In folder [easyocr/dict](https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/dict),
we need 'yourlanguagecode.txt' that contains list of words in your language.
On average, we have ~30000 words per language with more than 50000 words for more popular ones.
More is better in this file.

If your language has unique elements (such as 1. Arabic: characters change form when attached to each other + write from right to left 2. Thai: Some characters need to be above the line and some below), please educate us to the best of your ability and/or give useful links. It is important to take care of the detail to achieve a system that really works.

Lastly, please understand that our priority will have to go to popular languages or sets of languages that share large portions of their characters with each other (also tell us if this is the case for your language). It takes us at least a week to develop a new model, so you may have to wait a while for the new model to be released.

See [List of languages in development](https://github.com/JaidedAI/EasyOCR/issues/91)
