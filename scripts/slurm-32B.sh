#!/bin/bash
#SBATCH --partition=code
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
# 用srun启动，带--pty参数

#exec > >(tee -a "slurm-${SLURM_JOB_ID}.out") 2>&1

# 获取首结点（head node），这里利用 SLURM_NODELIST，第一个节点即为首结点
HEAD_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
echo "Head node is: ${HEAD_NODE}"

HEAD_IP=$(getent hosts $HEAD_NODE | awk "{print $1}")

HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist ${HEAD_NODE} hostname -i | head -n 1)
echo "Head node internal IP: ${HEAD_IP}"

export HEAD_NODE
export HEAD_IP

srun -n8 -N8 bash -c '
source /home/wangzefan/anaconda3/etc/profile.d/conda.sh
conda activate verl083

cd /home/wangzefan/data/verl_prime
source examples/0302/wzf_qy_env.sh

# 先提取第二张网卡名称
export NIC=$(ip addr | awk -F'\'': '\'' '\''/^[0-9]+:/{print $2}'\'' | sed -n '\''2p'\'')
# 检查该网卡是否处于 DOWN 状态
if ip addr show "$NIC" | grep -q "state DOWN"; then
    # 如果是 DOWN，则获取第三张网卡名称
    export NIC=$(ip addr | awk -F'\'': '\'' '\''/^[0-9]+:/{print $2}'\'' | sed -n '\''3p'\'')
fi

# 提取IB号
export NCCL_IB_HCA=$(ibstatus | grep "Infiniband device" | awk -F"'\''" '\''{print $2}'\'' | paste -sd, -)

export GLOO_SOCKET_IFNAME=$NIC
# export UCX_NET_DEVICES=$NIC
# export NCCL_SOCKET_IFNAME=$NIC

# 设置通信并发，似乎小的数值对all reduce更友好

# export CUDA_DEVICE_MAX_CONNECTIONS=32

# 启动 Ray 集群，在首结点启动 head，其它节点作为 worker 连接

echo "Running on $(hostname), SLURM_NODEID=$SLURM_NODEID"
ray stop --force
if [ "$SLURM_NODEID" -eq 0 ]; then
  echo "Starting Ray head on $(hostname)"
  ray start --head --port=6379 --node-ip-address=$HEAD_IP --resources='\''{"worker": 1}'\'' --system-config='\''{
  "worker_register_timeout_seconds": 60,
  "object_timeout_milliseconds": 600000,
  "gcs_redis_heartbeat_interval_milliseconds": 10000,
  "core_worker_internal_heartbeat_ms": 10000,
  "num_workers_soft_limit": 8
}'\''
#--num-cpus=32

else
  echo "Starting Ray worker on $(hostname), connecting to $HEAD_NODE"
  echo $HEAD_IP:6379
  ray start --address=$HEAD_IP:6379 --resources='\''{"worker": 1}'\''
  # --num-cpus=32
fi


# 等待 10 秒钟，确保 Ray 集群启动完成
sleep 15

# 注意mask task node不要和head node重合

while true; do
    if [ "$SLURM_NODEID" -eq 1 ]; then
      echo "Starting main Ray script on head node..."
      export WANDB_DIR=/home/wangzefan/data/verl_prime/
      bash examples/32BPRIME/prime-after.sh
      ret_code=$?
      if [ $ret_code -ne 0 ]; then
        echo "Main script exited with code ${ret_code}. Restarting after 10 seconds..."
        sleep 10
      else
        echo "Main script completed successfully. Exiting loop."
        break
      fi
    else
      sleep 1000
    fi
done
'


echo "Job finished on $(date)"