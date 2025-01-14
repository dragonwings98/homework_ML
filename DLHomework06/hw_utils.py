import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    if box_format == "midpoint":
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    else:  # box_format == "corners"
        box1_x1, box1_y1, box1_x2, box1_y2 = boxes_preds[..., 0], boxes_preds[..., 1], boxes_preds[..., 2], boxes_preds[..., 3]
        box2_x1, box2_y1, box2_x2, box2_y2 = boxes_labels[..., 0], boxes_labels[..., 1], boxes_labels[..., 2], boxes_labels[..., 3]

    # Intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Clamp to avoid negative values
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Areas of boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Union area
    union = box1_area + box2_area - intersection

    return intersection / union



def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):

    # Ensure the input is a list
    assert type(bboxes) == list

    # Filter boxes by confidence score threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort boxes by confidence score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    nms_bboxes = []  

    while bboxes:
        # Select the box with the highest confidence score
        chosen_box = bboxes.pop(0)

        # Remove boxes of the same class with IoU above the threshold
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] 
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), 
                torch.tensor(box[2:]),      
                box_format=box_format         
            ) < iou_threshold                
        ]

        # Add the chosen box to the NMS output
        nms_bboxes.append(chosen_box)

    # Return the filtered list of boxes
    return nms_bboxes


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):

    average_precisions = []
    epsilon = 1e-6  # To avoid division by zero

    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        amount_bboxes = len(ground_truths)
        if amount_bboxes == 0:
            continue

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            gt_bboxes = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            for idx, gt in enumerate(gt_bboxes):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                TP[detection_idx] = 1
                ground_truths.pop(best_gt_idx)
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

