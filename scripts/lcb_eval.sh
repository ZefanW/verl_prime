#!/bin/bash
source /home/test/test05/anaconda3/etc/profile.d/conda.sh   # set to your path
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
#cat /home/test/test05/wzf/eval/system_prompt.md

# 测试数据集，只测试my_array里的数据集
# my_array=(math500)
# my_array=(aime amc qwen math)
#my_array=(aime amc math500 qwen leetcode ) # 注意，只有这些测试被改过了读取prompt的地址！
my_array=( livecodebench)
#my_array=( leetcode)

cd /home/test/test05/wzf/PRIME/eval
#OLD_BASE=/home/test/test05/whb/project/eval-cgq-new
# Specify the test data set



if [[ " ${my_array[@]} " =~ " humaneval " ]]; then
    conda activate prime
    echo "running humaneval"
    mkdir -p $OUTPUT_DIR/human_eval_chat
    python3 Coding/human_eval/evaluate_human_eval.py \
        --model $MODEL_CKPT \
        --data_dir data/humaneval \
        --save_dir $OUTPUT_DIR/human_eval_chat
fi

if [[ " ${my_array[@]} " =~ " mbpp " ]]; then
    conda activate prime
    echo "running mbpp"
    mkdir -p $OUTPUT_DIR/mbpp_chat
    python3 -u Coding/mbpp/evaluate_mbpp.py \
        --model $MODEL_CKPT \
        --input_data data/mbpp/new_mbpp.json \
        --save_dir $OUTPUT_DIR/mbpp_chat
fi


if [[ " ${my_array[@]} " =~ " leetcode " ]]; then
    conda activate prime
    echo "running leetcode"
    mkdir -p $OUTPUT_DIR/leetcode_chat
    python3 Coding/leetcode/evaluate_leetcode.py \
        --model $MODEL_CKPT \
        --input_data data/leetcode/leetcode-test.json \
        --save_dir $OUTPUT_DIR/leetcode_chat
fi


if [[ " ${my_array[@]} " =~ " amc " ]]; then
    conda activate prime
    echo "running amc_chat(numina)"
    mkdir -p $OUTPUT_DIR/amc_chat
    python3 -u Math/amc/evaluate_amc.py \
        --model $MODEL_CKPT \
        --data_dir  data/AI-MO/aimo-validation-amc \
        --save_dir $OUTPUT_DIR/amc_chat
fi

if [[ " ${my_array[@]} " =~ " aime " ]]; then
    conda activate prime
    # AIME2024 chat
    echo "running aime_chat(numina)"
    mkdir -p $OUTPUT_DIR/aime_chat
    python3 -u Math/aime/evaluate_aime.py \
        --model $MODEL_CKPT \
        --data_dir  data/AI-MO/aimo-validation-aime \
        --save_dir $OUTPUT_DIR/aime_chat
fi


if [[ " ${my_array[@]} " =~ " math500 " ]]; then
    conda activate prime
    # math chat
    echo "running math_chat 500"
    mkdir -p $OUTPUT_DIR/math_chat
    python3 -u Math/math/evaluate_math.py \
        --model $MODEL_CKPT \
        --data_dir data/math500 \
        --save_dir $OUTPUT_DIR/math_chat
fi

if [[ " ${my_array[@]} " =~ " qwen " ]]; then
    conda activate qwen_math
    echo "running qwen math eval datasets"
    cd ./Math/Qwen25-Math/evaluation
    PROMPT_TYPE="qwen25-math-cot"
    MODEL_NAME_OR_PATH=$MODEL_CKPT
    mkdir -p $OUTPUT_DIR/qwen_math
    bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR/qwen_math
    cd ../../../
fi


if [[ " ${my_array[@]} " =~ " livecodebench " ]]; then
    conda activate lcb
    echo "running livecodebench"
    cd ./Coding/livecodebench/LiveCodeBench-main
    mkdir -p $OUTPUT_DIR/livecodebench
    python -m lcb_runner.runner.main --model $MODEL_CKPT --scenario codegeneration --evaluate --release_version release_v2 --output_path $OUTPUT_DIR/livecodebench
    # v2
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-05-31 >$OUTPUT_DIR/livecodebench/lcb_v2.txt 2>&1 &
#    # v3
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-08-03 >$OUTPUT_DIR/livecodebench/lcb_v3.txt 2>&1 &
#    # v4
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_v4.txt 2>&1 &
#    # 08-
#    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2024-08-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_08_11.txt 2>&1 &
    cd ../../../
fi








