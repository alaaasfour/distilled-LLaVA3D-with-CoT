#!/usr/bin/env python3
"""
Evaluation metrics for 3D VLM tasks.
Includes text generation metrics (BLEU, ROUGE, METEOR) and 3D task metrics.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. BLEU metrics will use simple implementation.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. ROUGE metrics will be unavailable.")

try:
    from nltk.translate.meteor_score import meteor_score
    import nltk
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    print("Warning: METEOR not available.")


def normalize_text(text: str) -> str:
    """Normalize text for evaluation."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove punctuation (optional - can keep for some metrics)
    # text = re.sub(r'[^\w\s]', '', text)
    return text


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization if NLTK is not available."""
    text = normalize_text(text)
    return text.split()


def compute_bleu(reference: str, candidate: str, n_gram: int = 4) -> Dict[str, float]:
    """
    Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4).
    
    Args:
        reference: Reference text
        candidate: Candidate text
        n_gram: Maximum n-gram order (default: 4)
        
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    if NLTK_AVAILABLE:
        try:
            ref_tokens = word_tokenize(normalize_text(reference))
            cand_tokens = word_tokenize(normalize_text(candidate))
            
            if len(ref_tokens) == 0 or len(cand_tokens) == 0:
                return {f'bleu-{i}': 0.0 for i in range(1, n_gram + 1)}
            
            smoothing = SmoothingFunction().method1
            scores = {}
            for i in range(1, n_gram + 1):
                try:
                    score = sentence_bleu([ref_tokens], cand_tokens, weights=[1.0/i] * i, smoothing_function=smoothing)
                    scores[f'bleu-{i}'] = float(score)
                except:
                    scores[f'bleu-{i}'] = 0.0
            return scores
        except Exception as e:
            print(f"Warning: BLEU calculation error: {e}")
            return {f'bleu-{i}': 0.0 for i in range(1, n_gram + 1)}
    else:
        # Simple BLEU implementation
        ref_tokens = simple_tokenize(reference)
        cand_tokens = simple_tokenize(candidate)
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return {f'bleu-{i}': 0.0 for i in range(1, n_gram + 1)}
        
        scores = {}
        for n in range(1, n_gram + 1):
            # Count n-grams
            ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)])
            
            # Compute precision
            matches = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())
            
            if total == 0:
                scores[f'bleu-{n}'] = 0.0
            else:
                precision = matches / total
                scores[f'bleu-{n}'] = float(precision)
        
        return scores


def compute_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    if not ROUGE_AVAILABLE:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            'rouge-1': scores['rouge1'].fmeasure,
            'rouge-2': scores['rouge2'].fmeasure,
            'rouge-l': scores['rougeL'].fmeasure,
            'rouge-1-p': scores['rouge1'].precision,
            'rouge-1-r': scores['rouge1'].recall,
            'rouge-2-p': scores['rouge2'].precision,
            'rouge-2-r': scores['rouge2'].recall,
            'rouge-l-p': scores['rougeL'].precision,
            'rouge-l-r': scores['rougeL'].recall,
        }
    except Exception as e:
        print(f"Warning: ROUGE calculation error: {e}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}


def compute_meteor(reference: str, candidate: str) -> float:
    """
    Compute METEOR score.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        METEOR score (0-1)
    """
    if not METEOR_AVAILABLE:
        return 0.0
    
    try:
        if NLTK_AVAILABLE:
            ref_tokens = word_tokenize(normalize_text(reference))
            cand_tokens = word_tokenize(normalize_text(candidate))
        else:
            ref_tokens = simple_tokenize(reference)
            cand_tokens = simple_tokenize(candidate)
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        score = meteor_score([ref_tokens], cand_tokens)
        return float(score)
    except Exception as e:
        print(f"Warning: METEOR calculation error: {e}")
        return 0.0


def exact_match(reference: str, candidate: str) -> float:
    """
    Compute exact match accuracy (binary).
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    ref_norm = normalize_text(reference)
    cand_norm = normalize_text(candidate)
    return 1.0 if ref_norm == cand_norm else 0.0


def compute_text_metrics(reference: str, candidate: str) -> Dict[str, float]:
    """
    Compute all text generation metrics.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        Dictionary with all text metrics
    """
    metrics = {}
    
    # BLEU scores
    bleu_scores = compute_bleu(reference, candidate)
    metrics.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = compute_rouge(reference, candidate)
    metrics.update(rouge_scores)
    
    # METEOR
    metrics['meteor'] = compute_meteor(reference, candidate)
    
    # Exact match
    metrics['exact_match'] = exact_match(reference, candidate)
    
    return metrics


def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, 
                          valid_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute depth estimation metrics.
    
    Args:
        pred_depth: Predicted depth map (H, W)
        gt_depth: Ground truth depth map (H, W)
        valid_mask: Optional mask for valid pixels (H, W)
        
    Returns:
        Dictionary with depth metrics (RMSE, MAE, δ1, δ2, δ3, REL)
    """
    if valid_mask is not None:
        pred_depth = pred_depth[valid_mask]
        gt_depth = gt_depth[valid_mask]
    else:
        # Only consider valid (non-zero) depth values
        valid_mask = (gt_depth > 0) & (pred_depth > 0)
        pred_depth = pred_depth[valid_mask]
        gt_depth = gt_depth[valid_mask]
    
    if len(pred_depth) == 0:
        return {
            'rmse': float('inf'),
            'mae': float('inf'),
            'delta1': 0.0,
            'delta2': 0.0,
            'delta3': 0.0,
            'rel': float('inf')
        }
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_depth - gt_depth))
    
    # Relative Error (REL)
    rel = np.mean(np.abs(pred_depth - gt_depth) / gt_depth)
    
    # Accuracy thresholds (δ1, δ2, δ3)
    # δ1: max(pred/gt, gt/pred) < 1.25
    # δ2: max(pred/gt, gt/pred) < 1.25^2
    # δ3: max(pred/gt, gt/pred) < 1.25^3
    ratios = np.maximum(pred_depth / gt_depth, gt_depth / pred_depth)
    delta1 = np.mean(ratios < 1.25)
    delta2 = np.mean(ratios < 1.25 ** 2)
    delta3 = np.mean(ratios < 1.25 ** 3)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
        'rel': float(rel)
    }


def compute_detection_metrics(pred_boxes: List[Dict], gt_boxes: List[Dict], 
                               iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute object detection metrics (simplified mAP).
    
    Args:
        pred_boxes: List of predicted boxes, each with 'bbox', 'score', 'class'
        gt_boxes: List of ground truth boxes, each with 'bbox', 'class'
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with detection metrics
    """
    # Simplified mAP calculation
    # For full mAP, need per-class AP and average across classes
    
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'map@0.5': 1.0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'map@0.5': 0.0}
    
    if len(pred_boxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'map@0.5': 0.0}
    
    # Sort predictions by score
    pred_boxes = sorted(pred_boxes, key=lambda x: x.get('score', 0.0), reverse=True)
    
    # Match predictions to ground truth
    matched_gt = set()
    true_positives = 0
    false_positives = 0
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            
            # Compute IoU (simplified - assumes bbox format [x1, y1, x2, y2])
            iou = compute_iou(pred_box['bbox'], gt_box['bbox'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and pred_box.get('class') == gt_boxes[best_gt_idx].get('class'):
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1
    
    false_negatives = len(gt_boxes) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Simplified mAP (using precision at recall threshold)
    map_score = precision  # Full mAP requires AP per class
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'map@0.5': float(map_score),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives)
    }


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with mean metrics
    """
    if len(metrics_list) == 0:
        return {}
    
    # Get all metric keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    aggregated = {}
    for key in all_keys:
        values = [m.get(key, 0.0) for m in metrics_list if key in m]
        if len(values) > 0:
            aggregated[key] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
    
    return aggregated


if __name__ == "__main__":
    # Test metrics
    reference = "The cat sat on the mat."
    candidate = "A cat was sitting on the mat."
    
    print("Testing text metrics...")
    text_metrics = compute_text_metrics(reference, candidate)
    for key, value in text_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTesting depth metrics...")
    pred_depth = np.random.rand(100, 100) * 10
    gt_depth = pred_depth + np.random.randn(100, 100) * 0.5
    depth_metrics = compute_depth_metrics(pred_depth, gt_depth)
    for key, value in depth_metrics.items():
        print(f"  {key}: {value:.4f}")

