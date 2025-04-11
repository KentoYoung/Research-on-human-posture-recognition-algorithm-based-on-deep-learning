# PyTorch OpenPose 实现

基于PyTorch的OpenPose实现，支持人体姿态估计和手部关键点检测。此项目将CMU原始的OpenPose模型从Caffe转换到PyTorch。

## 技术栈

- Python 3.7+
- PyTorch
- OpenCV (cv2)
- NumPy
- SciPy
- Matplotlib
- CUDA (GPU加速，可选)

## 功能

- 人体骨架关键点检测
- 手部关键点检测
- 支持摄像头实时检测
- 支持图片和视频文件处理

## 安装

1. 创建Python环境：

```bash
conda create -n pytorch-openpose python=3.7
conda activate pytorch-openpose
```

2. 安装PyTorch (请访问 https://pytorch.org/ 获取适合您系统的安装命令)

3. 安装其他依赖：

```bash
pip install -r requirements.txt
```

4. 下载预训练模型并放置在 `model` 目录下

## 使用方法

### 摄像头实时检测

```bash
python demo_camera.py
```

### 图片检测

```bash
python lowdemo.py
```

### 视频处理

```bash
python demo_video2picture.py <视频文件路径>
```

## 项目结构

- `src/`: 模型和工具函数源代码
  - `model.py`: 网络架构定义
  - `body.py`: 人体姿态估计实现
  - `hand.py`: 手部关键点检测实现
  - `util.py`: 工具函数
- `model/`: 预训练模型目录
- `images/`: 图片示例
- `videos/`: 视频示例
- `demo_camera.py`: 摄像头实时检测脚本
- `demo_video2picture.py`: 视频处理脚本
- `lowdemo.py`: 图片检测脚本

## 参考

此项目基于CMU的OpenPose项目实现，使用PyTorch框架重新实现。 