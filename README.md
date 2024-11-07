# YOLOv8 & DeepSORT Tracking Project

This project provides a Python implementation of real-time object tracking using **YOLOv8** for object detection and **DeepSORT** for multi-object tracking. The script processes an input video, detects objects using YOLOv8, and tracks them frame by frame using DeepSORT.

## Prerequisites

- Python 3.7+
- [PyTorch](https://pytorch.org/get-started/locally/) (for YOLO and DeepSORT)
- OpenCV (`cv2`) for image processing
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort) Python implementation

## Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/wishhappy2hmq/YOLOv8_Deepsort_Pytorch.git
   cd YOLOv8_Deepsort_Pytorch
   ```

2. **Create a virtual environment**:
   You can choose your own name for the environment; here we use `mypytorch` as the environment name:

   ```bash
   conda create -n mypytorch python=3.9
   ```

   After successfully creating the environment, activate the `mypytorch` environment:

   ```bash
   conda activate mypytorch
   ```

3. **Install PyTorch**:
   Install PyTorch in the newly created `mypytorch` environment by executing the following command:

   ```bash
   conda install pytorch torchvision torchaudio pytorch cuda=11.8 -c pytorch -c nvidia
   ```
   
   **Note**: Replace `11.8` with the CUDA version installed on your machine.

   **Offline Installation**:
   - Download URL: 
     For Windows (win-64):
     [https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorc](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorc)
   - Install PyTorch 2.0 version: `pytorch-2.0.0-py3.9_cuda11.8_cudnn8_0.tar.bz`
   - Install using the command:
     ```bash
     conda install --offline pytorch-2.0.0-py3.9_cuda11.8_cudnn8_0.tar.bz
     ```

4. **Install dependencies**:
   Use the provided `requirements.txt` to install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt** content:

   ```
   easydict==1.13
   ipdb==0.13.13
   matplotlib==3.9.2
   motmetrics==1.4.0
   numpy==2.1.3
   opencv_python==4.10.0.84
   pandas==2.2.3
   PyYAML==6.0.2
   scipy==1.14.1
   torch==2.5.1
   torchvision==0.20.1
   ultralytics==8.3.28
   ```

5. **Download the YOLOv8 model weights**:

   - You need to download the YOLOv8 model weights (`best.pt` or any other version). You can train your own model or download pre-trained weights from [Ultralytics' YOLOv8 releases](https://github.com/ultralytics/yolov8/releases).

6. **DeepSORT configuration and weights**:

   - Make sure the DeepSORT configuration file (`deep_sort.yaml`) is correctly set up. You can modify it based on your hardware and requirements.
   - **Note**: You also need to download the DeepSORT re-identification weights file (`ckpt.t7`) and place it in the path `deep_sort_pytorch/deep_sort/deep/checkpoint/`. This file is not included in the repository and must be downloaded separately.

## Usage

To run the object tracking, use the following command:

```bash
python tracker.py --input_path <input_video_path> --output_path <output_video_path> --model_path <model_weights_path> --device <device>
```

### Arguments

- `--input_path`: Path to the input video file.
- `--output_path`: Path to save the output video file with tracked objects.
- `--model_path`: Path to the YOLOv8 model weights (default is `best.pt`).
- `--device`: Choose either `cuda` for GPU or `cpu` for CPU (default is `cuda`).

### Example

```bash
python tracker.py --input_path demo.mp4 --output_path output.mp4 --model_path best.pt --device cuda
```

## Important Notes

- Make sure to **download the `best.pt` model weights** before running the script. The weights are not included in the repository.
- Make sure to **download the DeepSORT re-identification weights (`ckpt.t7`)** and place it in the appropriate folder as mentioned above.
- Ensure that you have a compatible **CUDA** environment if you want to use GPU acceleration.
- Adjust **DeepSORT** configuration based on your use case and hardware.

## Output

The script processes the video, performs object detection and tracking, and outputs a video file (`output.mp4`) with bounding boxes and tracking IDs displayed over the tracked objects.

## Contributing

If you have any suggestions or improvements, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License.

---

# YOLOv8 & DeepSORT 跟踪项目

本项目提供了一个基于 **YOLOv8** 进行目标检测和 **DeepSORT** 进行多目标跟踪的实时目标跟踪 Python 实现。脚本处理输入视频，使用 YOLOv8 进行目标检测，并使用 DeepSORT 对检测到的对象逐帧进行跟踪。

## 先决条件

- Python 3.7+
- [PyTorch](https://pytorch.org/get-started/locally/) （用于 YOLO 和 DeepSORT）
- OpenCV (`cv2`) 用于图像处理
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort) Python 实现

## 安装

1. **克隆此存储库**：

   ```bash
   git clone https://github.com/wishhappy2hmq/YOLOv8_Deepsort_Pytorch.git
   cd YOLOv8_Deepsort_Pytorch
   ```

2. **创建虚拟环境**：
   环境名字可自己确定，这里本人使用 `mypytorch` 作为环境名：

   ```bash
   conda create -n mypytorch python=3.9
   ```

   安装成功后激活 `mypytorch` 环境：

   ```bash
   conda activate mypytorch
   ```

3. **安装 PyTorch**：
   在所创建的 `mypytorch` 环境下安装 PyTorch，执行命令：

   ```bash
   conda install pytorch torchvision torchaudio pytorch cuda=11.8 -c pytorch -c nvidia
   ```
   
   **注意**：`11.8` 处应为自己电脑上的 CUDA 版本号。

   **离线安装**：
   - 下载网址: 
     Windows (win-64):
     [https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorc](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorc)
   - 安装 PyTorch 2.0 版本：`pytorch-2.0.0-py3.9_cuda11.8_cudnn8_0.tar.bz`
   - 使用以下命令安装：
     ```bash
     conda install --offline pytorch-2.0.0-py3.9_cuda11.8_cudnn8_0.tar.bz
     ```

4. **安装依赖项**：
   使用提供的 `requirements.txt` 安装依赖项。

   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt** 内容：

   ```
   easydict==1.13
   ipdb==0.13.13
   matplotlib==3.9.2
   motmetrics==1.4.0
   numpy==2.1.3
   opencv_python==4.10.0.84
   pandas==2.2.3
   PyYAML==6.0.2
   scipy==1.14.1
   torch==2.5.1
   torchvision==0.20.1
   ultralytics==8.3.28
   ```

5. **下载 YOLOv8 模型权重**：

   - 您需要下载 YOLOv8 模型权重（`best.pt` 或其他版本）。您可以训练自己的模型或从 [Ultralytics' YOLOv8 releases](https://github.com/ultralytics/yolov8/releases) 下载预训练的权重。

6. **DeepSORT 配置和权重**：

   - 确保正确设置了 DeepSORT 配置文件（`deep_sort.yaml`）。您可以根据硬件和需求进行修改。
   - **注意**：您还需要下载 DeepSORT 重识别权重文件（`ckpt.t7`），并将其放置在 `deep_sort_pytorch/deep_sort/deep/checkpoint/` 路径下。此文件不包含在存储库中，必须单独下载。

## 用法

使用以下命令运行目标跟踪：

```bash
python tracker.py --input_path <input_video_path> --output_path <output_video_path> --model_path <model_weights_path> --device <device>
```

### 参数

- `--input_path`：输入视频文件的路径。
- `--output_path`：保存带有跟踪目标的视频文件的路径。
- `--model_path`：YOLOv8 模型权重的路径（默认是 `best.pt`）。
- `--device`：选择 `cuda`（用于 GPU）或 `cpu`（默认是 `cuda`）。

### 示例

```bash
python tracker.py --input_path demo.mp4 --output_path output.mp4 --model_path best.pt --device cuda
```

## 重要说明

- 请确保在运行脚本之前 **下载 `best.pt` 模型权重**。权重文件不包含在存储库中。
- 请确保 **下载 DeepSORT 重识别权重文件 (`ckpt.t7`)** 并将其放置在适当的文件夹中，如上所述。
- 如果希望使用 GPU 加速，请确保您有兼容的 **CUDA** 环境。
- 根据您的使用情况和硬件调整 **DeepSORT** 配置。

## 输出

脚本处理视频，执行目标检测和跟踪，并输出一个带有边界框和跟踪 ID 的视频文件 (`output.mp4`)。

## 贡献

如果您有任何建议或改进，欢迎创建 issue 或提交 pull request。

## 许可证

本项目基于 MIT 许可证。

