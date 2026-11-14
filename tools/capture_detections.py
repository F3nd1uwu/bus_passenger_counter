"""Cache raw detector outputs and ReID vectors for reproducible tracker comparisons.

No database writes. The cache retains low-confidence detections; replay uses
the chosen association thresholds without repeating neural inference.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from config.settings import Settings
from core.detector import PersonDetector
from core.video_source import VideoSource, parse_start, source_kind


def capture(cfg, start, end, output, confidence=.10):
    cfg.validate()
    if source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND) != 'file' or not np.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError('Capture requires a video file and confidence in (0, 1]')
    output=Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    if start < 0 or end < start or output.suffix != '.npz':
        raise ValueError('Expected 0 <= start <= end and an unused .npz output')
    output.parent.mkdir(parents=True,exist_ok=True)
    detector=PersonDetector(cfg)
    reader=VideoSource(cfg,parse_start(cfg.VIDEO_START) if cfg.VIDEO_START else
                       datetime(2026,11,9,tzinfo=timezone.utc),resume_frame=start)
    boxes=[];features=[];valid=[];offsets=[0];indices=[];timestamps=[];shape=None;wall=time.perf_counter()
    try:
        while reader.index<=end:
            packet=reader.read()
            if packet is None:break
            shape=packet.image.shape[:2]
            raw=detector.model.predict(packet.image,classes=[cfg.PERSON_CLASS_ID],conf=confidence,
                                       imgsz=cfg.IMAGE_SIZE,device=cfg.DEVICE,verbose=False)[0].boxes.data.cpu().numpy()
            vectors=detector.reid.encode_many(packet.image,raw[:,:4])
            boxes.extend(raw)
            for vector in vectors:
                valid.append(vector is not None)
                features.append(vector if vector is not None else np.zeros(detector.reid.dimension,np.float32))
            offsets.append(len(boxes));indices.append(packet.index);timestamps.append(packet.timestamp.timestamp())
            if len(indices)%500==0:print(packet.index,'fps',round(len(indices)/(time.perf_counter()-wall),2),flush=True)
    finally:reader.close()
    if shape is None:
        raise ValueError('No frames read in requested interval')
    np.savez_compressed(output,boxes=np.asarray(boxes,np.float32).reshape(-1,6),
                        features=np.asarray(features,np.float32).reshape(-1,detector.reid.dimension),
                        valid=np.asarray(valid,bool),offsets=np.asarray(offsets),frames=np.asarray(indices),
                        timestamps=np.asarray(timestamps),shape=np.asarray(shape))
    meta={'source':cfg.VIDEO_SOURCE,'start':start,'end':end,'frames':len(indices),'model':cfg.MODEL_PATH,
          'model_sha256':hashlib.sha256(cfg.path(cfg.MODEL_PATH).read_bytes()).hexdigest(),
          'reid_sha256':hashlib.sha256(cfg.path(cfg.REID_MODEL_PATH).read_bytes()).hexdigest(),
          'confidence':confidence,'elapsed':time.perf_counter()-wall,'video_start':reader.start.isoformat(),
          'timestamps':'file presentation time relative to video_start; source edits not recovered'}
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='models/candidates/person_manual_v1.onnx')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    print(capture(replace(Settings(),VIDEO_SOURCE=a.source,MODEL_PATH=a.model,CPU_THREADS=a.threads),a.start,a.end,a.output))
