#!/bin/bash
#SBATCH --partition=code
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
# 用srun启动，带--pty参数

#exec > >(tee -a "slurm-${SLURM_JOB_ID}.out") 2>&1

# 批量提交lambda test任务，每台机器运行一个。本程序后面带的参数全部视作全局变量

HEAD_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
echo "Head node is: ${HEAD_NODE}"

HEAD_IP=$(getent hosts $HEAD_NODE | awk "{print $1}")

HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist ${HEAD_NODE} hostname -i | head -n 1)
echo "Head node internal IP: ${HEAD_IP}"

export HEAD_NODE
export HEAD_IP


source /home/wangzefan/anaconda3/etc/profile.d/conda.sh
conda activate verl083

cd /home/wangzefan/data/verl_prime
source examples/0302/wzf_qy_env.sh

# 启动 Ray 集群，在首结点启动 head，其它节点作为 worker 连接

echo "Running on $(hostname), SLURM_NODEID=$SLURM_NODEID"
ray stop --force

eval "$@"

ret_code=$?
if [ $ret_code -ne 0 ]; then
  echo "Main script exited with code ${ret_code}. "
fi

echo "Job finished on $(date)"