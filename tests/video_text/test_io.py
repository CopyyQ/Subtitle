from pathlib import Path

from src.video_text.io import build_rawvideo_mux_command


def test_rawvideo_mux_command_is_h264_high_yuv420p_and_preserves_audio(monkeypatch):
    monkeypatch.setattr("src.video_text.io._ffmpeg", lambda: "ffmpeg")
    cmd = build_rawvideo_mux_command(
        source_path=Path("source.mp4"),
        output_path=Path("out.mp4"),
        width=720,
        height=1280,
        fps=30.0,
        codec="h264",
    )

    assert cmd[:4] == ["ffmpeg", "-y", "-loglevel", "error"]
    assert ["-f", "rawvideo"] == cmd[4:6]
    assert "bgr24" in cmd
    assert "720x1280" in cmd
    assert "libx264" in cmd
    assert ["-profile:v", "high"] == cmd[cmd.index("-profile:v"):cmd.index("-profile:v")+2]
    pix_positions=[i for i,x in enumerate(cmd) if x=="-pix_fmt"]
    assert ["-pix_fmt", "yuv420p"] == cmd[pix_positions[-1]:pix_positions[-1]+2]
    assert ["-preset", "veryfast"] == cmd[cmd.index("-preset"):cmd.index("-preset")+2]
    assert ["-crf", "18"] == cmd[cmd.index("-crf"):cmd.index("-crf")+2]
    assert ["-movflags", "+faststart"] == cmd[cmd.index("-movflags"):cmd.index("-movflags")+2]
    assert ["-map", "0:v:0"] == cmd[cmd.index("-map"):cmd.index("-map")+2]
    assert "1:a?" in cmd
    assert ["-c:a", "copy"] == cmd[cmd.index("-c:a"):cmd.index("-c:a")+2]
    assert "-shortest" in cmd


def test_h265_rawvideo_command_uses_x265_without_h264_profile(monkeypatch):
    monkeypatch.setattr("src.video_text.io._ffmpeg", lambda: "ffmpeg")
    cmd = build_rawvideo_mux_command(
        source_path="source.mp4",
        output_path="out.mp4",
        width=1920,
        height=1080,
        fps=29.97,
        codec="h265",
    )
    assert "libx265" in cmd
    assert "-profile:v" not in cmd
    assert "yuv420p" in cmd


def test_rawvideo_mux_command_accepts_ultrafast_preset(monkeypatch):
    monkeypatch.setattr("src.video_text.io._ffmpeg", lambda: "ffmpeg")
    cmd=build_rawvideo_mux_command(
        source_path="source.mp4", output_path="out.mp4",
        width=720, height=1280, fps=30.0, codec="h264", preset="ultrafast",
    )
    assert ["-preset","ultrafast"] == cmd[cmd.index("-preset"):cmd.index("-preset")+2]


def test_rawvideo_mux_command_supports_nvenc(monkeypatch):
    monkeypatch.setattr("src.video_text.io._ffmpeg", lambda: "ffmpeg")
    cmd=build_rawvideo_mux_command(
        source_path="source.mp4", output_path="out.mp4",
        width=720, height=1280, fps=30.0,
        codec="h264", preset="ultrafast", engine="nvenc",
    )
    assert "h264_nvenc" in cmd
    assert ["-preset","p1"] == cmd[cmd.index("-preset"):cmd.index("-preset")+2]
    assert "-crf" not in cmd
    assert ["-cq:v","18"] == cmd[cmd.index("-cq:v"):cmd.index("-cq:v")+2]
