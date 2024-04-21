from coco_eval.coco_eval import COCOeval
import json
from pycocotools.coco import COCO
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_keypoints(pred_keypoints, gt_keypoints, images):
    """Function for drawing detected and ground truth keypoint during evaluation"""
    gt_keypoints = gt_keypoints.view(-1, 3).cpu().numpy()
    pred_keypoints = pred_keypoints.view(-1, 3).cpu().numpy()

    image_empty = Image.new('RGB', (images.shape[2], images.shape[3]))
    draw = ImageDraw.Draw(image_empty)

    # Visualize keypoints
    for x, y, v in gt_keypoints:
        if v > 0:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 0, 0), width=1)
    for x, y, v in pred_keypoints:
        if v > 0:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 0), width=1)

            # Visualize skeleton - add/delete pairs of keypoint indices to add or remove a line
    for j, (start, end) in enumerate([(1, 3), (3, 5), (2, 4), (4, 6), (7, 9), (9, 11), (8, 10), (10, 12), (7, 8), (0, 7), (0, 8)]):
        x1, y1, v1 = gt_keypoints[start, :]
        x2, y2, v2 = gt_keypoints[end, :]
        if v1 > 0 and v2 > 0:
            draw.line((x1, y1, x2, y2), fill=(0, 0, 255), width=1)
    for j, (start, end) in enumerate([(1, 3), (3, 5), (2, 4), (4, 6), (7, 9), (9, 11), (8, 10), (10, 12), (7, 8), (0, 7), (0, 8)]):
        x1, y1, v1 = pred_keypoints[start, :]
        x2, y2, v2 = pred_keypoints[end, :]
        if v1 > 0 and v2 > 0:
            draw.line((x1, y1, x2, y2), fill=(0, 125, 125), width=1)

    return transforms.ToTensor()(image_empty)

def evaluate_keypoint_detection(model, device, ann_file, data_loader, num_keypoints):
    """Method for performing coco-style evaluation of keypoints"""
    # Assign a default score to each detection (you can modify this)
    default_score = 0.9
    # Initialize COCO evaluation object
    coco_gt = COCO(ann_file)
    coco_dt = []

    # Run model on validation data and generate predictions
    with torch.no_grad():
        for images, _, image_id in data_loader:
            images = images.to(device)

            predictions = model(images)

            for i, keypoints in enumerate(predictions):
                keypoints = keypoints.cpu().numpy()

                keypoints = keypoints.tolist()

                coco_dt.append({
                    'image_id': image_id[i].item(),
                    'category_id': 20,  # Person category ID
                    'keypoints': keypoints,
                    'score': default_score,
                })

    # Save predictions in COCO format
    dt_path = 'keypoint_predictions.json'
    with open(dt_path, 'w') as f:
        json.dump(coco_dt, f)

    # Load predictions for evaluation
    coco_dt = coco_gt.loadRes(dt_path)

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints', num_keypoints=num_keypoints)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()