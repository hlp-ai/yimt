import argparse
import easyocr


def parse_args():
    parser = argparse.ArgumentParser(description="Process EasyOCR.")
    parser.add_argument(
        "-l",
        "--lang",
        nargs='+',
        required=True,
        type=str,
        help="for languages",
    )
    parser.add_argument(
        "--gpu",
        type=bool,
        choices=[True, False],
        default=True,
        help="Using GPU (default: True)",
    )
    parser.add_argument(
        "--model_storage_directory",
        type=str,
        default=None,
        help="Directory for model (.pth) file",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        choices=[True, False],
        default=True,
        help="Print detail/warning",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        choices=[True, False],
        default=True,
        help="Use dynamic quantization",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        type=str,
        help="input file",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["greedy", 'beamsearch'],
        default='greedy',
        help="decoder algorithm",
    )
    parser.add_argument(
        "--beamWidth",
        type=int,
        default=5,
        help="size of beam search",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch_size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of processing cpu cores",
    )
    parser.add_argument(
        "--detail",
        type=int,
        choices=[0, 1],
        default=1,
        help="simple output (default: 1)",
    )
    parser.add_argument(
        "--rotation_info",
        type=list,
        default=None,
        help="Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations.",
    )
    parser.add_argument(
        "--paragraph",
        type=bool,
        choices=[True, False],
        default=False,
        help="Combine result into paragraph",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=20,
        help="最小文本框，像素",
    )
    parser.add_argument(
        "--contrast_ths",
        type=float,
        default=0.1,
        help="Text box with contrast lower than this value will be passed into model 2 times. First is with original image and second with contrast adjusted to 'adjust_contrast' value. The one with more confident level will be returned as a result.",
    )
    parser.add_argument(
        "--adjust_contrast",
        type=float,
        default=0.5,
        help="target contrast level for low contrast text box",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.7,
        help="文本置信度阈值",
    )
    parser.add_argument(
        "--low_text",
        type=float,
        default=0.4,
        help="文本下限分数",
    )
    parser.add_argument(
        "--link_threshold",
        type=float,
        default=0.4,
        help="连接置信度阈值",
    )
    parser.add_argument(
        "--canvas_size",
        type=int,
        default=2560,
        help="最大图片，超过会缩小",
    )
    parser.add_argument(
        "--mag_ratio",
        type=float,
        default=1.,
        help="Image magnification ratio",
    )
    parser.add_argument(
        "--slope_ths",
        type=float,
        default=0.1,
        help="Maximum slope (delta y/delta x) to considered merging. Low value means tiled boxes will not be merged.",
    )
    parser.add_argument(
        "--ycenter_ths",
        type=float,
        default=0.5,
        help="y方向最大偏移，y方向相差太大的box不应合并",
    )
    parser.add_argument(
        "--height_ths",
        type=float,
        default=0.5,
        help="box高度最大差异。高度相差太大不应合并",
    )
    parser.add_argument(
        "--width_ths",
        type=float,
        default=0.5,
        help="合并box的最大水平距离",
    )
    parser.add_argument(
        "--y_ths",
        type=float,
        default=0.5,
        help="段落模式下最大合并垂直距离",
    )
    parser.add_argument(
        "--x_ths",
        type=float,
        default=1.0,
        help="段落模式下最大合并水平距离",
    )
    parser.add_argument(
        "--add_margin",
        type=float,
        default=0.1,
        help="Extend bounding boxes in all direction by certain value. This is important for language with complex script (E.g. Thai).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["standard", 'dict', 'json'],
        default='standard',
        help="output format.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reader = easyocr.Reader(lang_list=args.lang,
                            gpu=args.gpu,
                            model_storage_directory=args.model_storage_directory,
                            verbose=args.verbose,
                            quantize=args.quantize)

    for line in reader.readtext(args.file,
                                decoder=args.decoder,
                                beamWidth=args.beamWidth,
                                batch_size=args.batch_size,
                                workers=args.workers,
                                detail=args.detail,
                                rotation_info=args.rotation_info,
                                paragraph=args.paragraph,
                                min_size=args.min_size,
                                contrast_ths=args.contrast_ths,
                                adjust_contrast=args.adjust_contrast,
                                text_threshold=args.text_threshold,
                                low_text=args.low_text,
                                link_threshold=args.link_threshold,
                                canvas_size=args.canvas_size,
                                mag_ratio=args.mag_ratio,
                                slope_ths=args.slope_ths,
                                ycenter_ths=args.ycenter_ths,
                                height_ths=args.height_ths,
                                width_ths=args.width_ths,
                                y_ths=args.y_ths,
                                x_ths=args.x_ths,
                                add_margin=args.add_margin,
                                output_format=args.output_format):
        print(line)


if __name__ == "__main__":
    main()
