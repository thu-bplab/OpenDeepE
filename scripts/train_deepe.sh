RESULTS_DIR='../output/DeepE_humanvessel_sv1_3rd_train'

python -u ../train.py \
    --cate human_vessel \
    --results_dir $RESULTS_DIR \
    --epochs 500 \
    --decay_epochs 300 400 \
    --decay_factors 0.1 0.01 \
    --eval_epoch 1 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --model "deepe"
