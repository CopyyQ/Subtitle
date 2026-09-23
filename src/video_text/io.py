from __future__ import annotations
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import cv2

class CodecUnavailableError(RuntimeError):
    pass

def _ffmpeg():
    p=Path("/snap/bin/ffmpeg")
    if p.exists():
        return str(p)
    found=shutil.which("ffmpeg")
    if not found:
        raise CodecUnavailableError("FFmpeg is unavailable")
    return found

def ffmpeg_video_codec(codec):
    m={"h264":"libx264","h265":"libx265"}
    try:
        return m[codec.lower()]
    except KeyError:
        raise ValueError("codec must be h264 or h265")

def _run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode:
        raise CodecUnavailableError(p.stderr[-2000:] or "ffmpeg failed")
    return p

def build_rawvideo_mux_command(
    source_path,
    output_path,
    width,
    height,
    fps,
    codec="h264",
    preset="veryfast",
):
    enc=ffmpeg_video_codec(codec)
    cmd=[
        _ffmpeg(),
        "-y",
        "-loglevel","error",
        "-f","rawvideo",
        "-pix_fmt","bgr24",
        "-s",f"{int(width)}x{int(height)}",
        "-r",f"{float(fps):.8f}",
        "-i","pipe:0",
        "-i",str(source_path),
        "-map","0:v:0",
        "-map","1:a?",
        "-c:v",enc,
    ]
    if codec.lower()=="h264":
        cmd += ["-profile:v","high"]
    cmd += [
        "-pix_fmt","yuv420p",
        "-preset",str(preset),
        "-crf","18",
        "-movflags","+faststart",
        "-c:a","copy",
        "-shortest",
        str(output_path),
    ]
    return cmd


def encode_raw_frames_with_audio(
    frames,
    source_path,
    output_path,
    fps,
    codec="h264",
    preset="veryfast",
):
    output=Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    it=iter(frames)
    try:
        first=next(it)
    except StopIteration:
        raise ValueError("frames must not be empty")
    h,w=first.shape[:2]
    cmd=build_rawvideo_mux_command(
        source_path=source_path,
        output_path=output,
        width=w,
        height=h,
        fps=fps,
        codec=codec,
        preset=preset,
    )
    proc=subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        proc.stdin.write(first.tobytes())
        for frame in it:
            if frame.shape[:2]!=(h,w):
                raise ValueError("all frames must have identical dimensions")
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr=proc.stderr.read()
        stdout=proc.stdout.read()
        rc=proc.wait()
    except Exception:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        raise
    if rc:
        message=stderr.decode("utf-8","replace")[-2000:] if stderr else "ffmpeg failed"
        raise CodecUnavailableError(message)
    return output

def encode_video(frames,output_path,fps,codec="h264"):
    output=Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    it=iter(frames)
    try:
        first=next(it)
    except StopIteration:
        raise ValueError("frames must not be empty")
    h,w=first.shape[:2]
    tmp=output.with_name(output.stem+".raw.mp4")
    writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*"mp4v"),float(fps),(w,h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create temporary video: {tmp}")
    writer.write(first)
    for frame in it:
        if frame.shape[:2]!=(h,w):
            writer.release()
            raise ValueError("all frames must have identical dimensions")
        writer.write(frame)
    writer.release()
    enc=ffmpeg_video_codec(codec)
    try:
        _run([_ffmpeg(),"-y","-loglevel","error","-i",str(tmp),
              "-c:v",enc,"-pix_fmt","yuv420p","-an",str(output)])
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass
    return output

def transcode_video(video_only_path,output_path,codec="h264"):
    enc=ffmpeg_video_codec(codec)
    _run([_ffmpeg(),"-y","-loglevel","error","-i",str(video_only_path),
          "-c:v",enc,"-pix_fmt","yuv420p","-an",str(output_path)])
    return Path(output_path)

def mux_audio(video_only_path,source_path,output_path):
    cmd=[_ffmpeg(),"-y","-loglevel","error","-i",str(video_only_path),"-i",str(source_path),
         "-map","0:v:0","-map","1:a?","-c:v","copy","-c:a","copy",str(output_path)]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode:
        _run([_ffmpeg(),"-y","-loglevel","error","-i",str(video_only_path),"-i",str(source_path),
              "-map","0:v:0","-map","1:a?","-c:v","copy","-c:a","aac",str(output_path)])
    return Path(output_path)

def write_coordinate_json(
    path,metadata,records,events,display_shapes=None,*,frame_count=None,fps=None
):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    event_rows=[e.__dict__ if hasattr(e,"__dict__") else {
        "event_type":e.event_type,"track_id":e.track_id,
        "start_frame":e.start_frame,"end_frame":e.end_frame
    } for e in events]
    payload={"metadata":metadata,"records":records,"events":event_rows}
    if frame_count is not None:
        count=int(frame_count)
        frame_fps=float(fps if fps is not None else metadata.get("fps",0.0))
        by_frame={i:[] for i in range(count)}
        for row in records:
            fi=int(row["frame"])
            if fi not in by_frame:
                continue
            box={
                "track_id":int(row.get("track_id",0)),
                "subtitle_id":int(row.get("subtitle_id",row.get("track_id",0))),
                "line_id":int(row.get("line_id",0)),
                "bbox":[int(round(float(x))) for x in row["bbox"]],
                "reconstructed":bool(row.get("reconstructed",False)),
            }
            if "source" in row:
                box["source"]=row["source"]
            if "confidence" in row:
                box["confidence"]=row["confidence"]
            by_frame[fi].append(box)
        payload["frames"]=[
            {
                "frame":i,
                "timestamp":(i/frame_fps if frame_fps>0 else None),
                "boxes":by_frame[i],
            }
            for i in range(count)
        ]
    if display_shapes is not None:
        payload["display_shapes"]=display_shapes
    p.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return p

def _srt_time(seconds):
    ms=int(round(float(seconds)*1000))
    h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(path,segments):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    blocks=[]
    for i,seg in enumerate(segments,1):
        text=str(seg.get("text","")).strip()
        if not text:
            raise ValueError("SRT segment requires recognized text")
        blocks.append(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{text}\n")
    p.write_text("\n".join(blocks),encoding="utf-8")
    return p
