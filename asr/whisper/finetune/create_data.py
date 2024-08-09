import argparse
import json
import os

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a jsonl file to be used for fine-tuning a Whisper model"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help=(
            "Path to a text file containing audio filenames and transcriptions. This option is "
            "Each line must be in the format of "
            "`<audio_path>\t<transcription>`."
        ),
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the data",
    )
    parser.add_argument("--output", type=str, default="data.json", help="Path to output json file")
    parser.add_argument(
        "--dump-dir", type=str, default="dump", help="Directory to dump audio files"
    )

    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=223,
        help=(
            "Maximum length of prompt in Whisper tokens. Defaults to 223, which equals to "
            "`model.dims.n_text_ctx (=448) // 2 - 1` (-1 is for the special token `sot_prev` and "
            "the other half is for the transcribed tokens)."
        ),
    )
    parser.add_argument(
        "--max-tokens-length",
        type=int,
        default=219,
        help=(
            "Maximum length of text and timestamps tokens. Utterances longer than this will be "
            "skipped. Defaults to 219, which equals to `model.dims.n_text_ctx (=448) // 2 - 5` "
            "(5 is the maximum number of special tokens used other than the `sot_prev`."
        ),
    )
    parser.add_argument(
        "--subsampling-factor-for-silence",
        type=int,
        default=1,
        help=(
            "Subsampling factor for silence. This option is used to reduce the number of silence "
            "utterances. The original Whisper paper uses 1/10 of the number of silence utterances. "
            "Defaults to 1, which means no subsampling."
        ),
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="multilingual",
        choices=["multilingual", "english"],
        help=(
            "Type of Whisper tokenizer to use. Tokenizer is used to count the number of tokens "
            "in the transcriptions."
        ),
    )
    parser.add_argument("--normalize-unicode", action="store_true", help="Normalize unicode")
    return parser


DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio_path: str
    text: str  # text including timestamps
    language: str = "en"
    prompt: str = ""  # previous text including timestamps


class DataProcessor:
    def __init__(
        self,
        audio_dir: str = None,
        data_file: str = None,
        language: str = "en",
        output: str = "data.json",
        dump_dir: str = "dump",
        max_prompt_length: int = 223,
        max_tokens_length: int = 219,
        subsampling_factor_for_silence: int = 1,
        tokenizer_type: str = "multilingual",
        normalize_unicode: bool = False,
    ) -> None:
        self.audio_dir = audio_dir
        self.data_file = data_file
        self.language = language
        self.output = output
        self.dump_dir = dump_dir
        self.max_prompt_length = max_prompt_length
        self.max_tokens_length = max_tokens_length
        self.subsampling_factor_for_silence = subsampling_factor_for_silence
        self.tokenizer_type = tokenizer_type
        self.normalize_unicode = normalize_unicode

        self._verify_args()

        self.tokenizer = get_tokenizer(multilingual=(self.tokenizer_type == "multilingual"))
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

    def _verify_args(self) -> None:
        if self.language not in LANGUAGES:
            if self.language in TO_LANGUAGE_CODE:
                self.language = TO_LANGUAGE_CODE[self.language]
            else:
                raise ValueError(f"Unsupported language: {self.language}")

        if self.tokenizer_type not in ["multilingual", "english"]:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

        if Path(self.output).exists():
            raise ValueError(f"Output file {self.output} already exists")

    def run(self) -> None:
        self._process_without_timestamps()

        if self.subsampling_factor_for_silence > 1:
            self._subsample_silence()

    def _process_without_timestamps(self) -> None:
        records = []
        with open(self.data_file, encoding="utf-8") as f:
            for line in f:
                audio_path, text = line.strip().split("|", maxsplit=1)
                if self.normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)

                if self.audio_dir:
                    audio_path = os.path.join(self.audio_dir, audio_path)

                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_tokens_length:
                    print(
                        f"Skipping {audio_path} ({text}) because it is too long "
                        f"({len(tokens)} tokens)"
                    )
                    continue

                record = Record(audio_path=audio_path, text=text, language=self.language)
                records.append(record)

        self.write_records(records, self.output)

    @staticmethod
    def read_records(path: Union[str, Path]) -> List[Record]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                record = Record(
                    audio_path=data["audio_path"],
                    text=data["text"],
                    language=data["language"],
                    prompt=data["prompt"],
                )
                records.append(record)
        return records

    @staticmethod
    def write_records(records: List[Record], path: Union[str, Path]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                data = {
                    "audio_path": record.audio_path,
                    "text": record.text,
                    "language": record.language,
                    "prompt": record.prompt,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _subsample_silence(self) -> None:
        records = self.read_records(self.output)

        silence_records = filter(lambda record: record.text == "", records)
        non_silence_records = filter(lambda record: record.text != "", records)
        filtered_records = (
            list(non_silence_records)
            + list(silence_records)[:: self.subsampling_factor_for_silence]
        )

        Path(self.output).unlink()
        self.write_records(filtered_records, self.output)


def main():
    args = get_parser().parse_args()
    processor = DataProcessor(
        data_file=args.data_file,
        audio_dir=args.audio_dir,
        language=args.language,
        output=args.output,
        dump_dir=args.dump_dir,
        max_prompt_length=args.max_prompt_length,
        max_tokens_length=args.max_tokens_length,
        subsampling_factor_for_silence=args.subsampling_factor_for_silence,
        tokenizer_type=args.tokenizer_type,
        normalize_unicode=args.normalize_unicode,
    )
    processor.run()


if __name__ == "__main__":
    main()
