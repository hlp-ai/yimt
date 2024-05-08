# # Retrain the models used for CI.
# # Should be done rarely, indicates a major breaking change. 
my_python=python


############### TEST TRANSFORMER
if false; then
$my_python build_vocab.py \
    -config data/data.yaml -save_data data/data \
    -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -overwrite true -share_vocab

$my_python train.py \
    -config data/data.yaml -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -save_model /tmp/tmp \
    -batch_type tokens -batch_size 8 -accum_count 4 \
    -layers 1 -hidden_size 16 -word_vec_size 16 \
    -encoder_type transformer -decoder_type transformer \
    -share_embedding -share_vocab \
    -train_steps 1000 -world_size 1 -gpu_ranks 0 \
    -dropout 0.1 \
    -normalization tokens \
    -max_grad_norm 0 -optim adam -decay_method noam \
    -learning_rate 2 -label_smoothing 0.1 \
    -position_encoding -param_init 0 \
    -warmup_steps 100 -param_init_glorot -adam_beta2 0.998

mv /tmp/tmp*1000.pt onmt/tests/test_model.pt
rm /tmp/tmp*.pt
fi


if false; then
$my_python translate.py -gpu 0 -model onmt/tests/test_model.pt \
  -src data/src-val.txt -output onmt/tests/output_hyp.txt -beam 5 -batch_size 16

fi

############### TEST LANGUAGE MODEL
if false; then
rm data/data_lm/*.python

$my_python build_vocab.py \
    -config data/lm_data.yaml -save_data data/data_lm -share_vocab \
    -src_vocab data/data_lm/data.vocab.src -tgt_vocab data/data_lm/data.vocab.tgt \
    -overwrite true

$my_python train.py -config data/lm_data.yaml -save_model /tmp/tmp \
 -accum_count 2 -dec_layers 2 -hidden_size 64 -word_vec_size 64 -batch_size 256 \
 -encoder_type transformer_lm -decoder_type transformer_lm -share_embedding \
 -train_steps 2000 -dropout 0.1 -normalization tokens \
 -share_vocab -transformer_ff 256 -max_grad_norm 0 -optim adam -decay_method noam \
 -learning_rate 2 -label_smoothing 0.1 -model_task lm -world_size 1 -gpu_ranks 0 \
 -attention_dropout 0.1 -heads 2 -position_encoding -param_init 0 -warmup_steps 100 \
 -param_init_glorot -adam_beta2 0.998 -src_vocab data/data_lm/data.vocab.src
#
mv /tmp/tmp*2000.pt onmt/tests/test_model_lm.pt
rm /tmp/tmp*.pt
fi
#
if false; then
$my_python translate.py -gpu 0 -model onmt/tests/test_model_lm.pt \
  -src data/src-val.txt -output onmt/tests/output_hyp.txt -beam 5 -batch_size 16

fi

