# TWIST2 中文使用说明

TWIST2 是一个面向类人机器人（尤其是 Unitree G1）的“遥操作 + 运动数据采集 + 策略控制”系统。

它的核心思路可以概括为两层：

- 高层：通过 PICO VR / 运动文件生成机器人的目标动作。
- 低层：使用训练好的控制策略，让机器人稳定地执行这些动作。

如果你只是想先跑起来看效果，最简单的方式是直接使用仓库自带的 ONNX 控制器模型，不一定要先自己从头训练。

---

## 1. 先看一下它是怎么工作的

TWIST2 主要包含下面几个部分：

1. 训练/部署策略
   - 训练一个低层控制策略。
   - 最后导出成 ONNX 模型。

2. 高层遥操作
   - 通过 PICO VR 或离线运动文件生成目标动作。
   - 这些动作会被发送到 Redis。

3. 低层执行
   - 低层控制器从 Redis 读取动作。
   - 调用策略模型进行推理。
   - 再通过 PD 控制把输出变成关节力矩，驱动机器人或仿真环境。

---

## 2. 环境要求

建议配置：

- Ubuntu 20.04 / 22.04
- NVIDIA GPU（如果你要跑训练或 ONNX 推理）
- Conda
- Redis
- PICO VR（如果你要做在线遥操作）
- Unitree G1（如果你要做真实机器人部署）

---

## 3. 安装环境

TWIST2 需要两个 Conda 环境：

- `twist2`：用于训练、部署、数据采集
- `gmr`：用于在线重定向和遥操作

### 3.1 创建 `twist2` 环境

```bash
conda env remove -n twist2
conda create -n twist2 python=3.8
conda activate twist2
```

### 3.2 安装 Isaac Gym

如果你没有 Isaac Gym，需要先下载官方安装包，然后安装：

```bash
cd isaacgym/python
pip install -e .
```

### 3.3 安装项目依赖

```bash
cd /path/to/TWIST2
cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..
cd pose && pip install -e . && cd ..
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor zmq
pip install redis[hiredis]
pip install pyttsx3
pip install onnx onnxruntime-gpu
pip install customtkinter
```

### 3.4 安装并启动 Redis

如果你第一次用 Redis，建议先安装并启动它：

```bash
sudo apt update
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

然后修改配置文件：

```bash
sudo nano /etc/redis/redis.conf
```

把下面两项改成：

```bash
bind 0.0.0.0
protected-mode no
```

最后重启 Redis：

```bash
sudo systemctl restart redis-server
```

### 3.5 如果你要做 sim2real，安装 Unitree SDK

如果你打算用笔记本直接连真实 G1，需要安装 Unitree SDK 2 的 Python binding：

```bash
git clone https://github.com/YanjieZe/unitree_sdk2.git
cd unitree_sdk2
sudo apt-get update
sudo apt-get install build-essential cmake python3-dev python3-pip pybind11-dev
pip install pybind11 pybind11-stubgen numpy
cd python_binding
export UNITREE_SDK2_PATH=$(pwd)/..
bash build.sh --sdk-path $UNITREE_SDK2_PATH
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Installing to: $SITE_PACKAGES"
sudo cp build/lib/unitree_interface.cpython-*-linux-gnu.so $SITE_PACKAGES/unitree_interface.so
```

### 3.6 创建 `gmr` 环境（用于 PICO 遥操作）

```bash
conda create -n gmr python=3.10 -y
conda activate gmr

git clone https://github.com/YanjieZe/GMR.git
cd GMR
pip install -e .
cd ..

conda install -c conda-forge libstdcxx-ng -y
```

### 3.7 安装 PICO SDK / XRoboToolkit

如果你打算用 PICO VR 做在线遥操作，还需要安装对应 SDK。

建议按仓库自带说明中的步骤进行安装，这里不再展开所有细节。核心是：

- 在 PICO 侧安装 XRoboToolkit 的客户端。
- 在电脑上安装对应的 PC Service 和 Python SDK。
- 确保它们能和你的 PC 通信。

---

## 4. 先不用训练，直接跑一个示例

仓库已经提供了一个预训练好的策略模型：

- [assets/ckpts/twist2_1017_20k.onnx](assets/ckpts/twist2_1017_20k.onnx)

这意味着你可以先不自己训练，直接做仿真验证。

### 4.1 先启动高层 motion server

```bash
bash run_motion_server.sh
```

这一步会把一个离线运动文件通过 Redis 发给低层控制器。

### 4.2 再启动低层仿真控制器

```bash
bash sim2sim.sh
```

这一步会在仿真环境里运行策略模型。你会看到机器人在仿真中执行动作，或者默认站立等待指令。

如果你的机器性能足够，终端中会看到类似的 FPS 输出：

```bash
=== Policy Execution FPS Results (steps 1-1000) ===
Average Policy FPS: 38.88
Max Policy FPS: 41.55
Min Policy FPS: 24.72
Std Policy FPS: 1.92
Expected FPS (from decimation): 50.00
```

---

## 5. 用 PICO VR 做在线遥操作

如果你有 PICO VR，TWIST2 的重点能力就在这里。

### 5.1 启动遥操作脚本

```bash
bash teleop.sh
```

这条链路会把你的人体动作通过 GMR 重定向成机器人的目标动作，再由低层控制器执行。

### 5.2 你可以用的控制方式

仓库里的说明中提到了一些遥操作按键逻辑，例如：

- 右手 A：开始/暂停 teleop
- 左手 X：退出 teleop，进入默认站姿
- 左手 axis click：急停
- 右手/左手 grip/index grip：控制手部开合

如果你的 PICO 设备和 SDK 配置正常，应该就能开始进行遥操作。

---

## 6. 自己训练一个策略

如果你想训练自己的控制器，可以按照下面流程来：

### 6.1 训练策略

```bash
bash train.sh 1021_twist2 cuda:0
```

其中：

- 第一个参数是实验 ID
- 第二个参数是 GPU 设备名，例如 `cuda:0`

### 6.2 导出 ONNX

训练完之后，可以把策略导出为 ONNX：

```bash
bash to_onnx.sh $YOUR_POLICY_PATH
```

这里的 `$YOUR_POLICY_PATH` 通常是训练产出的 `.pt` 文件路径。

---

## 7. 真实机器人部署（Sim2Real）

如果你想把策略部署到真实的 Unitree G1 上，可以参考下面流程。

### 7.1 连接机器人

1. 启动机器人并用网线连接到电脑。
2. 把电脑网卡 IP 配置为：
   - IP: `192.168.123.222`
   - 子网掩码: `255.255.255.0`
3. 确认能 ping 通机器人：

```bash
ping 192.168.123.164
```

4. 用遥控器按下 `L2 + R2`，进入开发模式。

### 7.2 启动真实控制器

```bash
bash sim2real.sh
```

### 7.3 录制数据

如果你想采集数据，可以运行：

```bash
bash data_record.sh
```

---

## 8. 常见问题

### 8.1 Redis 没启动

如果脚本报错，先确认 Redis 是否正常：

```bash
sudo systemctl status redis-server
```

### 8.2 PICO 没有连上

请检查：

- PICO 侧是否已安装对应 SDK
- PC 侧服务是否启动
- 电脑网络是否通畅

### 8.3 找不到 ONNX 或策略文件

请确认：

- 你是否已经训练出策略模型
- 或者是否使用了仓库自带的模型：
  - [assets/ckpts/twist2_1017_20k.onnx](assets/ckpts/twist2_1017_20k.onnx)

---

## 9. 最推荐的新手路径

如果你是第一次接触 TWIST2，建议按这个顺序开始：

1. 安装环境和依赖
2. 启动 Redis
3. 先运行离线示例：
   - `bash run_motion_server.sh`
   - `bash sim2sim.sh`
4. 再尝试 PICO 遥操作：
   - `bash teleop.sh`
5. 最后再考虑真实机器人部署：
   - `bash sim2real.sh`

---

## 10. 一句话总结

TWIST2 的本质是：

- 用 PICO/VR 或离线运动数据生成高层动作；
- 用训练好的 RL 策略做低层控制；
- 把它们在仿真和真实机器人中串起来，完成遥操作与数据采集。

如果你愿意，我也可以继续帮你把这个文档再整理成“适合 Ubuntu 22.04 + conda 的一步一步命令版”，这样你可以直接照着复制执行。
