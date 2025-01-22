#!/bin/bash
source /home/test/test05/anaconda3/etc/profile.d/conda.sh
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)


MODEL_CKPT=$1
if [[ $MODEL_CKPT != /* ]]; then
  MODEL_CKPT="$PWD/$MODEL_CKPT"
fi
MODEL_CKPT=realpath $MODEL_CKPT
MODEL_NAME=$(echo $MODEL_CKPT | awk -F'/' '{print $(NF-2)"-"$(NF-1)"-"$NF}')
# MODEL_NAME=$(basename "$MODEL_CKPT")
OUTPUT_DIR="/home/test/test05/wzf/verl_prime/eval_results/$MODEL_NAME"
mkdir -p $OUTPUT_DIR

# 检查system prompt
#cat /home/test/test05/whb/project/eval-cgq-new/scripts/eval/system_prompt.md
cat /home/test/test05/wzf/eval/system_prompt.md

# 测试数据集，只测试my_array里的数据集
# my_array=(math500)
# my_array=(aime amc qwen math)
my_array=(aime amc math500 qwen leetcode) # 注意，只有这些测试被改过了读取prompt的地址！
#my_array=(leetcode livecodebench)

cd /home/test/test05/wzf/eval
OLD_BASE=/home/test/test05/whb/project/eval-cgq-new

if [[ " ${my_array[@]} " =~ " humaneval " ]]; then
    conda activate o1-new-vllm
    # humaneval # chat
    echo "running humaneval"
    mkdir -p $OUTPUT_DIR/human_eval_chat
    touch human_eval/human_eval
    python3 human_eval/evaluate_human_eval_chat_quicktest.py \
        --model $MODEL_CKPT \
        --save_dir $OUTPUT_DIR/human_eval_chat

    python $OLD_BASE/eval-harness/human-eval/human_eval/evaluate_functional_correctness.py --sample_file $OUTPUT_DIR/human_eval_chat/samples.jsonl

    nohup python $OLD_BASE/eval-harness/human-eval/human_eval/evaluate_functional_correctness.py \
            --sample_file $OUTPUT_DIR/human_eval_chat/samples.jsonl \
            >$OUTPUT_DIR/human_eval_chat/result.txt 2>&1 &
fi

if [[ " ${my_array[@]} " =~ " mbpp " ]]; then
    conda activate o1-new-vllm
    # mbpp # chat
    echo "running mbpp"
    mkdir -p $OUTPUT_DIR/mbpp_chat
    touch cache/mbpp
    python3 -u mbpp/run_mbpp_chat_quicktest.py \
        --model $MODEL_CKPT \
        --input_data /home/test/test05/whb/project/eval-cgq-new/scripts/eval/mbpp/new_mbpp.json \
        --save_dir $OUTPUT_DIR/mbpp_chat
fi

# 检查数组中是否包含math

if [[ " ${my_array[@]} " =~ " leetcode " ]]; then
    conda activate o1-new-vllm
    # leetcode # chat
    echo "running leetcode"
    mkdir -p $OUTPUT_DIR/leetcode_chat
    touch cache/leetcode
    python3 leetcode/evaluate_leetcode_chat_quicktest.py \
        --model $MODEL_CKPT \
        --save_dir $OUTPUT_DIR/leetcode_chat

    python ./leetcode/evaluate_leetcode.py --generation_path $OUTPUT_DIR/leetcode_chat/samples.jsonl --temp_dir ./cache

    nohup python ./leetcode/evaluate_leetcode.py \
            --generation_path $OUTPUT_DIR/leetcode_chat/samples.jsonl \
            --temp_dir ./cache \
            >$OUTPUT_DIR/leetcode_chat/result.txt 2>&1 &
fi

if [[ " ${my_array[@]} " =~ " gpqa " ]]; then
    conda activate o1-new-vllm
    # gpqa
    echo "running gpqa"
    mkdir -p $OUTPUT_DIR/gpqa
    python3 -u ./gpqa/run_gpqa_chat.py \
        --model $MODEL_CKPT \
        --data_dir /home/test/test05/whb/data/test_data_o1/gpqa \
        --save_dir $OUTPUT_DIR/gpqa
fi


if [[ " ${my_array[@]} " =~ " amc " ]]; then
    conda activate o1-new-vllm
    # AMC chat
    echo "running amc_chat(numina)"
    mkdir -p $OUTPUT_DIR/amc_chat
    python3 -u ./amc/evaluate_amc_chat_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir  /home/test/test05/whb/data/test_data_o1/AI-MO/aimo-validation-amc \
        --save_dir $OUTPUT_DIR/amc_chat
fi

if [[ " ${my_array[@]} " =~ " aime " ]]; then
    conda activate o1-new-vllm
    # AIME2024 chat
    echo "running aime_chat(numina)"
    mkdir -p $OUTPUT_DIR/aime_chat
    python3 -u ./aime/evaluate_aime_chat_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir  /home/test/test05/whb/data/test_data_o1/AI-MO/aimo-validation-aime \
        --save_dir $OUTPUT_DIR/aime_chat
fi

if [[ " ${my_array[@]} " =~ " math " ]]; then
    conda activate o1-new-vllm
    # math chat
    echo "running math_chat"
    mkdir -p $OUTPUT_DIR/math_chat
    python3 -u ./math/evaluate_math_chat_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir /home/test/test05/whb/data/test_data_o1/math \
        --save_dir $OUTPUT_DIR/math_chat
fi

if [[ " ${my_array[@]} " =~ " math500 " ]]; then
    conda activate o1-new-vllm
    # math chat
    echo "running math_chat 500"
    mkdir -p $OUTPUT_DIR/math_chat
    python3 -u ./math/evaluate_math_chat_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir /home/test/test05/whb/data/test_data_o1/math500 \
        --save_dir $OUTPUT_DIR/math_chat
fi

if [[ " ${my_array[@]} " =~ " qwen " ]]; then
    conda activate qwen-math
    echo "running qwen math eval datasets"
    cd ./Qwen25-Math/evaluation
    PROMPT_TYPE="qwen25-math-cot"
    MODEL_NAME_OR_PATH=$MODEL_CKPT
    mkdir -p $OUTPUT_DIR/qwen_math
    bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR/qwen_math
    cd ../../
fi

 if [[ " ${my_array[@]} " =~ " livecodebench " ]]; then
     # export OUTLINES_CACHE_DIR='~/.cache/outlines/$(uuidgen)'
     conda activate whb_lcb
     echo "running livecodebench"
     cd ./livecodebench/LiveCodeBench-main
     mkdir -p $OUTPUT_DIR/livecodebench
     # export OUTLINES_CACHE_DIR='~/.cache/outlines/$(uuidgen)'
     python -m lcb_runner.runner.main --model $MODEL_CKPT --scenario codegeneration --evaluate --release_version release_v2 --output_path $OUTPUT_DIR/livecodebench --n 1
     # cp -r /home/test/test05/whb/project/eval-cgq-new/scripts/eval/livecodebench/LiveCodeBench-main/output $OUTPUT_DIR/livecodebench
     cd ../../
 fi
#
#if [[ " ${my_array[@]} " =~ " livecodebench " ]]; then
#    # export OUTLINES_CACHE_DIR='~/.cache/outlines/$(uuidgen)'
#    conda activate whb_lcb
#    echo "running livecodebench"
#    cd ./scripts/eval/livecodebench/LiveCodeBench-main
#    mkdir -p $OUTPUT_DIR/livecodebench
#    # export OUTLINES_CACHE_DIR='~/.cache/outlines/$(uuidgen)'
#    python -m lcb_runner.runner.main --model $MODEL_CKPT --scenario codegeneration --evaluate --release_version release_v4 --output_path $OUTPUT_DIR/livecodebench
#    # cp -r /home/test/test05/whb/project/eval-cgq-new/scripts/eval/livecodebench/LiveCodeBench-main/output $OUTPUT_DIR/livecodebench
#    # v2
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-05-31 >$OUTPUT_DIR/livecodebench/lcb_v2.txt 2>&1 &
#    # v3
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-08-03 >$OUTPUT_DIR/livecodebench/lcb_v3.txt 2>&1 &
#    # v4
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_v4.txt 2>&1 &
#    # 08-
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2024-08-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_08_11.txt 2>&1 &
#    cd ../../../../
#fi
#




if [[ " ${my_array[@]} " =~ " mmlu " ]]; then
    # mmlu
    echo "running mmlu"
    mkdir -p $OUTPUT_DIR/mmlu
    python3 -u ./mmlu/evaluate_mmlu_quicktest.py \
        --model $MODEL_CKPT \
        --data_dir /home/test/test05/whb/data/test_data_o1/mmlu \
        --save_dir $OUTPUT_DIR/mmlu
fi







