#python ./tools/ckpts/convert_neox_to_hf.py --input_dir /path/to/model/global_stepXXX --config_file your_config.yml --output_dir hf_model/save/location --precision {auto,fp16,bf16,fp32} --architecture {neox,mistral,llama}

python ./ckpts/convert_neox_to_hf.py --input_dir /lustre/fast/fast/txiao/shihan/saves/SepLLM-160m/checkpoints_n64ht_8cards_kernel_recompile_rotaryBiPE/global_step2 \
	--config_file /lustre/fast/fast/txiao/shihan/saves/SepLLM-160m/checkpoints_n64ht_8cards_kernel_recompile_rotaryBiPE/global_step2/configs/sepllm-160m-on-pythia-with-pile_deduped-n64HT-kernel_recompile_rotaryBiPE.yml \
	--output_dir /lustre/fast/fast/txiao/shihan/saves/SepLLM/hf_checkpoints/debugs/160m_n64h_BiPE_debug/global_step2 \
	--precision auto \
	--architecture sepllm_gpt_neox
	
# /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_seg/pythia160m-deduped/checkpoints_k0_w128
# /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_seg/pythia160m-deduped/scaled_masked_softmax_fusion/checkpoints_w64_ht2048