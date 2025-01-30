# 
# --num_fewshot 5 \
# --tasks  arc_challenge,arc_easy,lambada_openai,hendrycksTest*,winogrande\
# arc_challenge,arc_easy,blimp,lambada_openai,logiqa,mmlu,piqa,sciq,wikitext,winogrande,wsc \
# /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_ori/pythia160m-deduped/checkpoints_k0_w64/hf_checkpoints/global_step93000
# --tasks  arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc \
# --model_args pretrained=/lustre/home/txiao/shihan/workspace/gpt-neox-new_seg/mytest/official_json_step93000 \




# /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_seg/pythia160m-deduped/scaled_masked_softmax_fusion/checkpoints_w64_h2048/hf_checkpoints/global_step143000
# /lustre/fast/fast/txiao/shihan/saves/SepLLM/hf_checkpoints/debugs/160m_n64h_debug/global_step143000  ./new_convert_bash_debug_n64h.log
# --batch_size auto 2>&1 | tee ./old_n64h_repr.log
# --tasks    arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc,wikitext  \
CUDA_LAUNCH_BLOCKING=1
lm_eval --model hf \
	--model_args pretrained=/lustre/fast/fast/txiao/shihan/saves/SepLLM/hf_checkpoints/debugs/160m_n64h_debug/global_step143000 \
	--tasks    lambada_openai \
	--num_fewshot 5 \
	--device cuda:0\
	--batch_size 2 2>&1 | tee ./n64h_debug.log
