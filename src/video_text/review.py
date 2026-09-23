from __future__ import annotations
from pathlib import Path
import cv2
from .io_probe import bottom_roi

def export_event_contact_sheet(video_path,event,output_dir,roi_fraction,context=2):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode MP4: {video_path}")
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y1,y2=bottom_roi(height,roi_fraction)
    start=max(0,event.start_frame-context)
    end=max(start,event.end_frame+context)
    crops=[]
    for fi in range(start,end+1):
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
        ok,frame=cap.read()
        if not ok:
            continue
        crop=frame[y1:y2].copy()
        cv2.putText(crop,f"{event.event_type} f={fi} t={fi/fps:.3f}s",(8,26),
                    cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,255),2,cv2.LINE_AA)
        crops.append(crop)
    cap.release()
    if not crops:
        raise RuntimeError("no frames available for review event")
    sheet=cv2.hconcat(crops)
    out=output_dir/f"track_{event.track_id:04d}_{event.event_type}_frame_{event.start_frame:06d}.jpg"
    if not cv2.imwrite(str(out),sheet,[int(cv2.IMWRITE_JPEG_QUALITY),95]):
        raise RuntimeError(f"cannot write review sheet: {out}")
    return out
