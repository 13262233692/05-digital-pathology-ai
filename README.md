# 05-digital-pathology-ai

面向医疗影像硬核落地场景的全玻片图像（WSI）病理超分辨率重建与筛查管线。

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI 调度层                           │
│  REST API / 任务管理 / 文件上传下载 / 状态监控                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Celery 异步任务层                          │
│  分块任务分发 / 批处理推理 / 结果汇聚 / 容错重试                 │
│  Broker: Redis                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌──────────────────────┐      ┌────────────────────────────────┐
│   WSI 分块模块        │      │    Triton 推理服务              │
│  - OpenSlide 读取     │      │  - gRPC 协议                   │
│  - 金字塔滑窗         │      │  - 动态批处理                   │
│  - 512x512 分块       │      │  - SRGAN 4x 超分               │
│  - 32px 重叠区        │      │                                │
│  - 组织区域检测       │      └────────────────────────────────┘
└──────────────────────┘                           │
           │                                       │
           └───────────────┬───────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    高斯距离加权混合缝合模块                      │
│  - 边缘权重衰减                                                 │
│  - 重叠区平滑过渡                                               │
│  - OME-TIFF 标准格式输出                                        │
│  - 金字塔层级写入                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. WSI 动态分块模块 ([src/wsi_processor/](file:///d:/SOLO-9/05-digital-pathology-ai/src/wsi_processor/))

- **WSIReader**: 基于 OpenSlide 解析 .svs 格式全玻片图像，支持金字塔多层级读取
- **TileExtractor**: 金字塔层级滑窗算法，512x512 分块 + 32px 重叠区，自动组织区域检测过滤

### 2. SRGAN 模型 ([src/models/](file:///d:/SOLO-9/05-digital-pathology-ai/src/models/))

- 16 残差块生成器，4 倍超分辨率
- VGG19 特征提取感知损失
- 训练器支持 PSNR 验证和自动模型导出

### 3. Triton 推理客户端 ([src/triton_client/](file:///d:/SOLO-9/05-digital-pathology-ai/src/triton_client/))

- gRPC 协议同步/异步推理
- BatchProcessor 自动批处理队列
- 支持动态批量大小和并发控制

### 4. 图像缝合模块 ([src/image_stitcher/](file:///d:/SOLO-9/05-digital-pathology-ai/src/image_stitcher/))

- **GaussianStitcher**: 高斯距离加权混合算法，边缘权重衰减
- **OME_TIFFWriter**: 标准 OME-TIFF 格式写入，支持金字塔层级，LZW 压缩

### 5. Celery 任务流 ([src/celery_tasks/](file:///d:/SOLO-9/05-digital-pathology-ai/src/celery_tasks/))

- `process_wsi`: 主任务，WSI 分块 + Chord 工作流编排
- `process_tile_batch`: 批处理推理任务
- `stitch_and_save`: 缝合保存回调任务

### 6. FastAPI 控制台 ([src/api/](file:///d:/SOLO-9/05-digital-pathology-ai/src/api/))

- `POST /api/v1/wsi/upload`: WSI 文件上传并启动处理
- `GET /api/v1/tasks/{task_id}`: 查询任务状态
- `GET /api/v1/tasks/{task_id}/result/download`: 下载超分结果
- `DELETE /api/v1/tasks/{task_id}`: 取消任务

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 模型训练

```bash
python scripts/train_model.py \
  --train_dir /path/to/hr/train \
  --val_dir /path/to/hr/val \
  --batch_size 16 \
  --num_epochs 200
```

### 导出 Triton 模型

```bash
python scripts/export_triton_model.py \
  --checkpoint ./checkpoints/srgan_best.pth \
  --output ./triton_model_repository/srgan/1/model.pt
```

### 单张 WSI 本地处理

```bash
python scripts/process_single_wsi.py \
  --wsi_path /path/to/image.svs \
  --output_dir ./output \
  --triton_url localhost:8001
```

### Docker 部署

```bash
docker-compose up -d
```

启动的服务：
- Redis: localhost:6379
- Triton Server: localhost:8000/8001/8002
- Celery Worker
- FastAPI: http://localhost:8000

## 配置文件

[configs/config.yaml](file:///d:/SOLO-9/05-digital-pathology-ai/configs/config.yaml) 包含所有可配置参数：

- WSI 分块参数（tile_size, overlap）
- SRGAN 网络结构
- Triton 服务配置
- Celery 队列配置
- 缝合算法参数

## 核心技术特性

1. **WSI 高效分块**: 金字塔层级滑窗 + 组织区域检测，跳过空白区域
2. **重叠区混合**: 32px 重叠区 + 高斯距离加权，消除分块边缘接缝
3. **显存克制调度**: 流式分块处理，避免一次性加载整张千兆级图像
4. **异步任务编排**: Celery Chord 工作流，支持分布式扩展
5. **医疗标准输出**: OME-TIFF 标准格式，兼容主流病理图像查看器

## 运行测试

```bash
# 测试 SRGAN 模型
python tests/test_srgan_model.py

# 测试高斯缝合模块
python tests/test_gaussian_stitcher.py
```

## 目录结构

```
.
├── src/
│   ├── wsi_processor/      # WSI 读取与分块
│   ├── models/             # SRGAN 模型定义与训练
│   ├── triton_client/      # Triton 推理客户端
│   ├── image_stitcher/     # 高斯缝合与 OME-TIFF 写入
│   ├── celery_tasks/       # Celery 异步任务
│   └── api/                # FastAPI 接口
├── configs/
│   └── config.yaml         # 全局配置
├── scripts/                # 工具脚本
├── tests/                  # 单元测试
├── triton_model_repository/ # Triton 模型仓库
├── docker-compose.yml      # Docker 编排
└── requirements.txt        # Python 依赖
```
