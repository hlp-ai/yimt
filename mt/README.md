# NMT


----
## For beginners:

There is a step-by-step and explained tuto (Thanks to Yasmin Moslem): [Tutorial](https://github.com/ymoslem/OpenNMT-Tutorial)

Please try to read and/or follow before raising newbies issues.

Otherwise you can just have a look at the [Quickstart](https://opennmt.net/OpenNMT-py/quickstart.html) steps

----
## New:

* You will need Pytorch v2 preferably v2.2 which fixes some `scaled_dot_product_attention` issues
* LLM support with converters for: Llama (+ Mistral), OpenLlama, Redpajama, MPT-7B, Falcon.
* Support for 8bit and 4bit quantization along with LoRA adapters, with or without checkpointing.
* You can finetune 7B and 13B models on a single RTX 24GB with 4-bit quantization.
* Inference can be forced in 4/8bit using the same layer quantization as in finetuning.
* Tensor parallelism when the model does not fit on one GPU's memory (both training and inference)
* Once your model is finetuned you can run inference either with OpenNMT-py or faster with CTranslate2.
* MMLU evaluation script, see results [here](https://github.com/OpenNMT/OpenNMT-py/blob/master/eval_llm/MMLU/readme.md)

For all usecases including NMT, you can now use Multiquery instead of Multihead attention (faster at training and inference) and remove biases from all Linear (QKV as well as FeedForward modules).


If you used previous versions of OpenNMT-py, you can check the [Changelog](https://github.com/OpenNMT/OpenNMT-py/blob/master/CHANGELOG.md) or the [Breaking Changes](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/changes.md)

----

## Tutorials:

* How to replicate Vicuna with a 7B or 13B llama (or Open llama, MPT-7B, Redpajama)  Language Model: [Tuto Vicuna](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/replicate_vicuna/ReplicateVicuna.md)
* How to finetune NLLB-200 with your dataset: [Tuto Finetune NLLB-200](https://forum.opennmt.net/t/finetuning-and-curating-nllb-200-with-opennmt-py/5238)
* How to create a simple OpenNMT-py REST Server: [Tuto REST](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392)
* How to create a simple Web Interface: [Tuto Streamlit](https://forum.opennmt.net/t/simple-web-interface/4527)
* Replicate the WMT17 en-de experiment: [WMT17 ENDE](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/wmt17/Translation.md)

----

## Setup

### Installing locally

OpenNMT-py requires:

- Python >= 3.8
- PyTorch >= 2.0 <2.2

Install `OpenNMT-py` from `pip`:
```bash
pip install OpenNMT-py
```

or from the source:
```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
pip install -e .
```

Note: if you encounter a `MemoryError` during installation, try to use `pip` with `--no-cache-dir`.

*(Optional)* Some advanced features (e.g. working pretrained models or specific transforms) require extra packages, you can install them with:

```bash
pip install -r requirements.opt.txt
```

### Manual installation of some dependencies

Apex is highly recommended to have fast performance (especially the legacy fusedadam optimizer and FusedRMSNorm)

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --no-build-isolation --config-settings --build-option="--cpp_ext --cuda_ext --deprecated_fused_adam --xentropy --fast_multihead_attn" ./
cd ..
```

Flash attention:

As of Oct. 2023 flash attention 1 has been upstreamed to pytorch v2 but it is recommended to use flash attention 2 with v2.3.1 for sliding window attention support.

When using regular `position_encoding=True` or Rotary with `max_relative_positions=-1` OpenNMT-py will try to use an optimized dot-product path.

if you want to use [flash attention](https://github.com/Dao-AILab/flash-attention#installation-and-features) then you need to manually install it first:

```bash
pip install flash-attn --no-build-isolation
```

if flash attention 2 is not installed, then we will use `F.scaled_dot_product_attention` from pytorch 2.x

When using `max_relative_positions > 0` or Alibi `max_relative_positions=-2` OpenNMT-py will use its legacy code for matrix multiplications.

flash attention and `F.scaled_dot_product_attention` are a bit faster and saves some GPU memory.


AWQ:

If you want to run inference or quantize an AWQ model you will need AutoAWQ.

For [AutoAWQ](https://github.com/casper-hansen/AutoAWQ):
    pip install autoawq



