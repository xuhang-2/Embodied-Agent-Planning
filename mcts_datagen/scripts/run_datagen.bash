#!/bin/bash


xvfb_pid=$(pgrep -f "Xvfb :1")

if [ -n "$xvfb_pid" ]; then
    echo "找到 Xvfb 进程，PID: $xvfb_pid"
    echo "正在终止 Xvfb 进程..."
    kill -9 $xvfb_pid
    echo "Xvfb 进程已终止"
else
    echo "未找到正在运行的 Xvfb 进程"
fi


# 启动 Xvfb
Xvfb :1 -screen 0 1024x768x16 &

# 设置 DISPLAY 环境变量
export DISPLAY=:1

# 运行 Python 脚本
python ../Embodied-Agent-Planning/mcts_datagen/src/main.py
