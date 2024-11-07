import torch
import cv2
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
import numpy as np

# deepsort配置文件路径
config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"

class Tracker:
    def __init__(self, input_path, output_path, model_path='best.pt', device='cuda', img_size=640):
        """
        初始化追踪器

        :param input_path: 视频输入路径
        :param output_path: 视频输出路径
        :param model_path: YOLOv8 模型路径
        :param device: 设备（'cuda' 或 'cpu'）
        :param img_size: 输入图片大小
        """
        self.device = device
        self.img_size = img_size
        self.model = YOLO(model_path)  # 加载 YOLOv8 模型
        self.cap = cv2.VideoCapture(input_path)  # 打开视频文件
        assert self.cap.isOpened(), f"无法打开视频文件 {input_path}"

        # 设置输出视频文件的编码格式和帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, int(self.cap.get(cv2.CAP_PROP_FPS)),
                                   (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # 初始化 DeepSORT
        self.deepsort = self.init_deepsort()

    def init_deepsort(self):
        """
        初始化 DeepSORT 配置并返回实例

        :return: DeepSORT 实例
        """
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)  # 加载配置文件
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE,
                            n_init=cfg.DEEPSORT.N_INIT,
                            nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=self.device != 'cpu')  # 根据设备选择 CUDA 或 CPU
        return deepsort

    def prepare_for_deepsort(self, detections):
        """
        将 YOLOv8 的检测结果转化为 DeepSORT 的输入格式

        :param detections: YOLOv8 的检测结果
        :return: bbox_xyxy, confs, cls_ids
        """
        if detections is None or len(detections) == 0:
            return np.zeros((0, 4)), np.zeros((0, 1)), np.zeros((0, 1))

        # 还原后的边界框 (x1, y1, x2, y2)
        bbox_xyxy = detections.xyxy.cpu().numpy()
        confs = detections.conf.cpu().numpy().reshape(-1, 1)  # 置信度
        cls_ids = detections.cls.cpu().numpy().reshape(-1, 1)  # 类别 ID

        # 过滤掉无效的边界框
        valid_indices = (bbox_xyxy[:, 2] > bbox_xyxy[:, 0]) & (bbox_xyxy[:, 3] > bbox_xyxy[:, 1])
        bbox_xyxy = bbox_xyxy[valid_indices]
        confs = confs[valid_indices]
        cls_ids = cls_ids[valid_indices]

        return bbox_xyxy, confs, cls_ids

    def track(self):
        """
        开始目标追踪
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取视频帧，或视频已播放完毕")
                break

            # 使用 YOLOv8 进行目标检测
            results = self.model(frame)  # 获取检测结果
            detections = results[0].boxes  # 获取 YOLOv8 的检测框

            # 如果有检测结果
            if len(detections) > 0:
                bbox_xyxy = detections.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confs = detections.conf.cpu().numpy()  # 置信度
                cls_ids = detections.cls.cpu().numpy()  # 类别 ID

                # 将边界框从 [x1, y1, x2, y2] 转换为 [cx, cy, w, h]
                bbox_xywh = []
                for box in bbox_xyxy:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    bbox_xywh.append([cx, cy, w, h])

                # 转换为 tensor 格式
                bbox_xywh = torch.Tensor(bbox_xywh)
                confs = torch.Tensor(confs)

                # 使用 DeepSORT 进行目标追踪
                outputs = self.deepsort.update(bbox_xywh, confs, cls_ids, frame)

                # 绘制边界框和追踪 ID
                if isinstance(outputs, np.ndarray):
                    for output in outputs:
                        x1, y1, x2, y2, track_id, class_id = output
                        # 确保框的位置在图像范围内
                        x1, y1, x2, y2 = map(lambda x: max(0, min(x, frame.shape[1] - 1)), [x1, y1, x2, y2])

                        # 绘制边界框
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f'ID: {track_id}, Class: {class_id}'
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 显示视频帧
                resized_frame = cv2.resize(frame, (1280, 720))  # 调整分辨率适配显示窗口
                cv2.imshow("Tracking", resized_frame)
                self.out.write(frame)  # 保存视频帧

            # 按键 'Q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# 主程序
if __name__ == "__main__":
    # 创建 Tracker 实例并启动追踪
    tracker = Tracker(input_path="demo.mp4", output_path="output.mp4", device='cuda')
    tracker.track()
