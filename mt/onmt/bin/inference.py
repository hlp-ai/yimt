import json
import time
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed


def _get_parser():
    parser = ArgumentParser(description="inference.py")
    opts.translate_opts(parser)
    return parser


def evaluate(opt, inference_mode, input_file, out, method):
    print("# input file", input_file)
    run_results = {}
    # Build the translator (along with the model)
    if inference_mode == "py":
        print("Inference with py ...")
        from onmt.inference_engine import InferenceEnginePY

        engine = InferenceEnginePY(opt)
    elif inference_mode == "ct2":
        print("Inference with ct2 ...")
        from onmt.inference_engine import InferenceEngineCT2

        opt.src_subword_vocab = opt.models[0] + "/source_vocabulary.json"
        opt.tgt_subword_vocab = opt.models[0] + "/target_vocabulary.json"
        engine = InferenceEngineCT2(opt)

    start = time.time()

    if method == "file":
        engine.opt.src = input_file
        scores, preds = engine.infer_file()
    elif method == "list":
        src = open(input_file, "r", encoding="utf-8").readlines()
        scores, preds = engine.infer_list(src)

    engine.terminate()

    dur = time.time() - start
    print(f"Time to generate {len(preds)} answers: {dur}s")

    if inference_mode == "py":
        scores = [
            [_score for _score in _scores] for _scores in scores
        ]
    run_results = {"pred_answers": preds, "score": scores, "duration": dur}

    # output_filename = out + f"_{method}.json"
    # with open(output_filename, "w",  encoding="utf-8") as f:
    #     json.dump(run_results, f, ensure_ascii=False, indent=2)
    with open(out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p[0] + "\n")


def main():
    # Required arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-config", help="Inference config file", required=True, type=str
    )
    parser.add_argument(
        "-inference_mode",
        help="Inference mode",
        default="py",
        type=str,
        choices=["py", "ct2"],
    )
    parser.add_argument(
        "-input_file",
        help="File with formatted input examples.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-out",
        help="Output filename.",
        required=True,
        type=str,
    )

    args = parser.parse_args()
    inference_config_file = args.config
    base_args = ["-config", inference_config_file]
    parser = _get_parser()

    opt = parser.parse_args(base_args)
    print(opt)

    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)

    set_random_seed(opt.seed, use_gpu(opt))
    opt.models = opt.models

    evaluate(
        opt,
        inference_mode=args.inference_mode,
        input_file=args.input_file,
        out=args.out,
        method="file",
    )


if __name__ == "__main__":
    main()
