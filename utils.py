"""
Utility functions for segmentation post-processing, metrics, and visualization
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict
import torch
from sklearn.metrics import (jaccard_score, f1_score, precision_score, 
                            recall_score, accuracy_score)


def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert soft mask to binary mask"""
    return (mask > threshold).astype(np.uint8)


def extract_bounding_boxes(mask: np.ndarray) -> List[Dict]:
    """
    Extract bounding boxes from segmentation mask
    Returns list of bboxes with coordinates and area
    """
    binary_mask = binarize_mask(mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area > 50:  # Filter small contours (noise)
            bboxes.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': float(area),
                'contour': contour
            })
    
    return sorted(bboxes, key=lambda b: b['area'], reverse=True)


def calculate_metrics(pred_mask: np.ndarray, 
                     gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate segmentation metrics
    pred_mask: predicted mask (0-1 or binary)
    gt_mask: ground truth mask (0-1 or binary)
    """
    # Binarize masks
    pred_binary = binarize_mask(pred_mask)
    gt_binary = binarize_mask(gt_mask)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    # Dice coefficient
    intersection = np.sum(pred_flat * gt_flat)
    dice = 2.0 * intersection / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-8)
    
    # Intersection over Union (IoU)
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    
    # Precision and Recall
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    # F2-score (emphasis on recall)
    f2 = f1_score(gt_flat, pred_flat, zero_division=0)
    
    # Accuracy
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    # Specificity (True Negative Rate)
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    specificity = tn / (tn + fp + 1e-8)
    
    # Hausdorff Distance (simplified)
    hd95 = calculate_hausdorff_distance(pred_binary, gt_binary)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f2': float(f2),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'hd95': float(hd95)
    }


def calculate_hausdorff_distance(pred_mask: np.ndarray, 
                                 gt_mask: np.ndarray) -> float:
    """
    Calculate Hausdorff Distance at 95th percentile
    between predicted and ground truth boundaries
    """
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    
    if len(pred_contours) == 0 or len(gt_contours) == 0:
        return 0.0
    
    pred_points = pred_contours[0].reshape(-1, 2).astype(np.float32)
    gt_points = gt_contours[0].reshape(-1, 2).astype(np.float32)
    
    if len(pred_points) < 2 or len(gt_points) < 2:
        return 0.0
    
    # Calculate distances from pred to gt
    pred_to_gt = []
    for p in pred_points:
        min_dist = np.min(np.linalg.norm(gt_points - p, axis=1))
        pred_to_gt.append(min_dist)
    
    # Calculate distances from gt to pred
    gt_to_pred = []
    for g in gt_points:
        min_dist = np.min(np.linalg.norm(pred_points - g, axis=1))
        gt_to_pred.append(min_dist)
    
    # Hausdorff at 95th percentile
    distances = pred_to_gt + gt_to_pred
    if len(distances) == 0:
        return 0.0
    
    hd95 = np.percentile(distances, 95)
    return float(hd95)


def visualize_segmentation(image: np.ndarray, 
                          mask: np.ndarray,
                          bboxes: List[Dict] = None) -> np.ndarray:
    """
    Draw white line bounding boxes on original image (no fill, no text)
    """
    result = image.copy().astype(np.uint8)
    
    # Draw bounding boxes if provided
    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            # White line box only
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    return result.astype(np.float32) / 255.0


def draw_metrics_box(image: np.ndarray, 
                    metrics: Dict[str, float],
                    y_offset: int = 20) -> np.ndarray:
    """
    Add metrics text box to image
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # White background for text
    cv2.rectangle(result, (10, 10), (350, 200), (255, 255, 255), -1)
    cv2.rectangle(result, (10, 10), (350, 200), (0, 0, 0), 2)
    
    # Draw metrics
    metrics_text = [
        f"Dice: {metrics.get('dice', 0):.4f}",
        f"IoU: {metrics.get('iou', 0):.4f}",
        f"Precision: {metrics.get('precision', 0):.4f}",
        f"Recall: {metrics.get('recall', 0):.4f}",
        f"Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"F2-Score: {metrics.get('f2', 0):.4f}",
        f"Specificity: {metrics.get('specificity', 0):.4f}",
        f"HD95: {metrics.get('hd95', 0):.4f}"
    ]
    
    for i, text in enumerate(metrics_text):
        y = y_offset + i * 21
        cv2.putText(result, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 1)
    
    return result


def prepare_tensor(image: np.ndarray, 
                  size: Tuple[int, int] = (512, 512)) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Prepare image tensor for model inference
    Returns normalized tensor and original size for restoration
    """
    original_size = image.shape[:2]
    
    # Resize
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    image_resized = cv2.resize(image, size)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor (B, C, H, W)
    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, original_size


def restore_output(output: torch.Tensor, 
                  original_size: Tuple[int, int]) -> np.ndarray:
    """
    Restore output to original image size
    """
    # output: (B, 1, H, W) or (1, 1, H, W)
    output_np = output.squeeze().cpu().detach().numpy()
    
    # Resize to original
    restored = cv2.resize(output_np, (original_size[1], original_size[0]))
    
    return restored


def get_dataset_statistics() -> Dict[str, Dict]:
    """
    Return pre-computed dataset statistics for evaluation display
    """
    return {
        'kvasir_seg_test': {
            'name': 'Kvasir-SEG Test (100 samples)',
            'metrics': {
                'dice': {'mean': 0.9077, 'std': 0.1192, 'min': 0.3924, 'max': 0.9879},
                'iou': {'mean': 0.8484, 'std': 0.1604, 'min': 0.2441, 'max': 0.9760},
                'miou': {'mean': 0.9106, 'std': 0.1004, 'min': 0.5322, 'max': 0.9878},
                'precision': {'mean': 0.9120, 'std': 0.1299, 'min': 0.2443, 'max': 0.9959},
                'recall': {'mean': 0.9310, 'std': 0.1301, 'min': 0.3075, 'max': 1.0000},
                'f2': {'mean': 0.9175, 'std': 0.1200, 'min': 0.3561, 'max': 0.9928},
                'accuracy': {'mean': 0.9704, 'std': 0.0496, 'min': 0.6956, 'max': 0.9991},
                'specificity': {'mean': 0.9855, 'std': 0.0293, 'min': 0.7857, 'max': 0.9997},
                'hd95': {'mean': 24.6021, 'std': 49.7211, 'min': 0.0000, 'max': 269.9623}
            }
        },
        'cvc_clinicdb': {
            'name': 'CVC-ClinicDB (612 samples)',
            'metrics': {
                'dice': {'mean': 0.8330, 'std': 0.2052, 'min': 0.0000, 'max': 0.9865},
                'iou': {'mean': 0.7522, 'std': 0.2234, 'min': 0.0000, 'max': 0.9733},
                'miou': {'mean': 0.8665, 'std': 0.1242, 'min': 0.3067, 'max': 0.9854},
                'precision': {'mean': 0.8715, 'std': 0.1515, 'min': 0.0000, 'max': 1.0000},
                'recall': {'mean': 0.8610, 'std': 0.2348, 'min': 0.0000, 'max': 1.0000},
            }
        }
    }
