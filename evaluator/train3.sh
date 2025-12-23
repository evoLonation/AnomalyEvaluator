#!/bin/bash
# 将所有参数传入，同时额外加入--eval参数

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM 信号，终止所有子进程
trap 'kill $(jobs -p) 2>/dev/null; exit' SIGINT SIGTERM

# 在后台并行运行两个命令
CUDA_VISIBLE_DEVICES=1 python -m evaluator.train3 "$@" --eval &
CUDA_VISIBLE_DEVICES=0 python -m evaluator.train3 "$@" &

# 等待所有后台进程完成
wait