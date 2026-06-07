import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import List, Dict, Optional, Tuple
from loguru import logger
import numpy as np


class NucleusDetector:
    def __init__(
        self,
        num_classes: int = 2,
        min_score: float = 0.5,
        mask_threshold: float = 0.5,
        device: str = "auto",
        backbone: str = "resnet50",
        trainable_backbone_layers: int = 3,
    ):
        self.num_classes = num_classes
        self.min_score = min_score
        self.mask_threshold = mask_threshold
        self.backbone = backbone
        self.trainable_backbone_layers = trainable_backbone_layers
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._model: Optional[nn.Module] = None
        self._build_model()

    def _build_model(self) -> None:
        logger.info(f"Building Mask R-CNN with {self.backbone} backbone, {self.num_classes} classes")
        
        self._model = maskrcnn_resnet50_fpn_v2(
            pretrained=False,
            num_classes=self.num_classes,
            trainable_backbone_layers=self.trainable_backbone_layers,
        )
        
        in_features_box = self._model.roi_heads.box_predictor.cls_score.in_features
        self._model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features_box,
            num_classes=self.num_classes
        )
        
        in_features_mask = self._model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self._model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=hidden_layer,
            num_classes=self.num_classes
        )
        
        self._model.to(self.device)
        self._model.eval()
        
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    def load_weights(self, weight_path: str) -> None:
        state_dict = torch.load(weight_path, map_location=self.device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self._model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {weight_path}")

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_masks: bool = True,
        return_boxes: bool = True,
    ) -> Dict:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.copy()
            if image_float.max() > 1.0:
                image_float = image_float / 255.0
        
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1).float()
        image_tensor = image_tensor.to(self.device)
        
        self._model.eval()
        outputs = self._model([image_tensor])
        
        output = outputs[0]
        
        scores = output["scores"].cpu().numpy()
        keep = scores >= self.min_score
        
        result = {
            "scores": scores[keep],
            "labels": output["labels"].cpu().numpy()[keep],
        }
        
        if return_boxes:
            result["boxes"] = output["boxes"].cpu().numpy()[keep]
        
        if return_masks:
            masks = output["masks"].cpu().numpy()[keep]
            masks = (masks > self.mask_threshold).astype(np.uint8)
            result["masks"] = masks[:, 0, :, :]
        
        result["num_instances"] = int(keep.sum())
        
        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Dict]:
        tensors = []
        for image in images:
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.dtype == np.uint8:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image.copy()
                if image_float.max() > 1.0:
                    image_float = image_float / 255.0
            t = torch.from_numpy(image_float).permute(2, 0, 1).float()
            tensors.append(t.to(self.device))
        
        self._model.eval()
        outputs = self._model(tensors)
        
        results = []
        for output in outputs:
            scores = output["scores"].cpu().numpy()
            keep = scores >= self.min_score
            
            result = {
                "scores": scores[keep],
                "labels": output["labels"].cpu().numpy()[keep],
                "boxes": output["boxes"].cpu().numpy()[keep],
                "masks": (output["masks"].cpu().numpy()[keep] > self.mask_threshold).astype(np.uint8)[:, 0, :, :],
                "num_instances": int(keep.sum()),
            }
            results.append(result)
        
        return results

    def extract_contours(
        self,
        masks: np.ndarray,
    ) -> List[np.ndarray]:
        import cv2
        
        contours_list = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contours_list.append(largest)
            else:
                contours_list.append(np.empty((0, 1, 2), dtype=np.int32))
        
        return contours_list

    def train_mode(self) -> None:
        self._model.train()

    def eval_mode(self) -> None:
        self._model.eval()
    
    @property
    def model(self) -> nn.Module:
        return self._model
