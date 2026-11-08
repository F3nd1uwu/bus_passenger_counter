"""Conservative removal of partial/full detections of the same body."""
import numpy as np


def distinct_person_indices(boxes, features):
    """Keep separate people; remove a contained crop of the same body.

    Standard IoU NMS can retain a short torso crop alongside its full-body box.
    Require coincident head/side boundaries AND similar ReID features before
    suppressing it. Overlap or appearance alone is insufficient.
    """
    if len(boxes) < 2 or features is None:
        return np.arange(len(boxes), dtype=int)
    boxes = np.asarray(boxes)
    features = np.asarray(features)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    kept = []
    for i in np.argsort(-areas, kind='stable'):
        duplicate = False
        for j in kept:
            if areas[i] <= 0 or areas[j] <= 0:
                continue
            intersection = np.maximum(0, np.minimum(boxes[i, 2:4], boxes[j, 2:4])
                                      - np.maximum(boxes[i, :2], boxes[j, :2]))
            contained = float(np.prod(intersection)) / areas[i] >= .95
            same_width = min(widths[i], widths[j]) / max(widths[i], widths[j]) >= .85
            same_top = abs(boxes[i, 1] - boxes[j, 1]) <= .04 * heights[j]
            substantial = areas[i] / areas[j] >= .60
            if (contained and same_width and same_top and substantial
                    and float(features[i] @ features[j]) >= .80):
                duplicate = True
                break
        if not duplicate:
            kept.append(int(i))
    return np.asarray(sorted(kept), dtype=int)
