NUM_GPUS=4 # number of gpus on the VM
MODEL=mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B # model name
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_num_batched_tokens=32768,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# pretrained = model name
# dtype = datatype
# tensor_parallel_size= how many gpus to distribute the model across
# data_parallel_size = how many gpus to distribute the test data across (NOTE: experimental)
# max_num_batched_tokens=max number of batched tokens run simultaneously (HAS TO BE THE SAME AS max_model_length),
# max_model_length=max number of tokens the model can take (SPECIFIED USUALLY AS THIS IN TRAINING),
# gpu_memory_utilization= how much of the GPU is reserved for ... the model? Unsure
# generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}


TASK=pubmedqa # task name (NEEDS TO MATCH WHAT IS IN UTILS)
OUTPUT_DIR=data/evals/$MODEL # WHERE TO STORE THE OUTPUT

# pipeline_parallel_size
#--disable-async-output-proc

export VLLM_WORKER_MULTIPROC_METHOD=spawn # NEEDED FOR PARALLELIZATION
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \ # USE LIGHTEVAL WITH VLLM 
    --use-chat-template \ # NECESSARY ARGUMENT
    --output-dir $OUTPUT_DIR # USE OUTPUT DIRECTORY





NUM_GPUS=4
MODEL=mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_num_batched_tokens=32768,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=med_qa
OUTPUT_DIR=data/evals/$MODEL

# pipeline_parallel_size
#--disable-async-output-proc

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "helm|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

NUM_GPUS=4
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_num_batched_tokens=32768,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=med_qa_4opt_cardiovascular
OUTPUT_DIR=data/evals/$MODEL

# pipeline_parallel_size
#--disable-async-output-proc

# SAVE DETAILS TO GET ALL PREDICTIONS

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "helm|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

### THE ALOE MODEL

NUM_GPUS=4
MODEL=HPAI-BSC/Llama3-Aloe-8B-Alpha
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_num_batched_tokens=8192,max_model_length=8192,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}"
TASK=med_qa_4opt_cardiovascular
OUTPUT_DIR=data/evals/$MODEL

# pipeline_parallel_size
#--disable-async-output-proc

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "helm|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
