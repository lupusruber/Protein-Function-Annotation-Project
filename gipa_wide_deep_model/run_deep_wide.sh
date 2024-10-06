cd "$(dirname $0)"
python -u /content/gipa_wide_deep/train_gipa.py \
    --root "./" \
    --train-partition-num 1 \
    --eval-partition-num 1 \
    --eval-times 1 \
    --lr 0.0005 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 20 \
    --n-layers 10 \
    --dropout 0.4 \
    --n-hidden 64 \
    --input-drop 0.4 \
    --edge-drop 0.1 \
    --edge-agg-mode "single_softmax" \
    --edge-att-act "none" \
    --norm none \
    --edge-emb-size 8\
    --gpu 0 \
    --first-hidden 512 \
    --use-sparse-fea \
    --sparse-encoder "log" \
    --n-deep-layers 3 \
    --n-deep-hidden 64 \
    --deep-drop-out 0.4 \
    --deep-input-drop 0.1 \
    --model "gipa_deep_wide" \
    --log-file-name="run_default_wide_deep"