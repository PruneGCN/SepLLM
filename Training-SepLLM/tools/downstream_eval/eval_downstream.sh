lm_eval --model hf \
	--model_args pretrained=/path/to/outputs/hf_checkpoints/global_stepXXX \
	--tasks    arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc,wikitext  \
	--num_fewshot 5 \
	--device cuda:0\
	--batch_size auto 2>&1 | tee ./demo.log
