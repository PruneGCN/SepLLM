CUDA_VISIBLE_DEVICES=0  python ../../../main/evaluate_streaming_inputs_perplexity.py \
    --init_cache_size 4 \
    --local_size 320 \
    --cache_size 324 \
    --enable_kv_cache_manager True \
    --enable_SepLLM False \
    --enable_StreamingLLM True \
    --enable_pos_shift False \
    --num_samples 5000000 \
    --num_eval_tokens 20480 \
    --dataset_name wikitext \
    --task wikitext-2-raw-v1 \
    --output_dir ../../../outputs/demo/xxx   2>&1 | tee ../../../logs/demo/xxx.log