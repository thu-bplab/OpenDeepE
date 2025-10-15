MODEL_PATH='../output/DeepE_humanvessel_sv1_3rd_train/weights/model_epoch_274.pth'
RESULTS_DIR='../output/DeepE_humanvessel_sv1_3rd_train/inference/epoch_274'

python ../test.py \
   --cate human_vessel \
   --model_path $MODEL_PATH \
   --batch_size 64 \
   --results_dir $RESULTS_DIR \
   --model "deepe"
