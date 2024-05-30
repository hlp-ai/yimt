import sentencepiece as spm


def load_spm(sp_model_path):
    """Load SentencePiece model from file"""
    return spm.SentencePieceProcessor(model_file=sp_model_path)