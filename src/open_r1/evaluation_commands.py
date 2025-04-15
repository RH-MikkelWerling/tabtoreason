NUM_GPUS=4
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=med_qa_4opt
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 

NUM_GPUS=4
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=med_qa_4opt
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

NUM_GPUS=4
MODEL=mikkel-werling/DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=8192,gpu_memory_utilization=0.2,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=med_qa_4opt
OUTPUT_DIR=data/evals/$MODEL

# pipeline_parallel_size
#--disable-async-output-proc

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR



NUM_GPUS=4
MODEL=mikkel-werling/DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=2,data_parallel_size=2,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=med_qa_4opt
OUTPUT_DIR=data/evals/$MODEL

# pipeline_parallel_size
#--disable-async-output-proc

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


make evaluate MODEL=mikkel-werling/DeepSeek-R1-Distill-Qwen-7B TASK=med_qa_4opt PARALLEL=tensor NUM_GPUS=4
