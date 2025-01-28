import subprocess
import os
from src.pose_analysis.state_machine import PoseState

def clip_video_with_ffmpeg(video_path, segment_info, output_dir):
    """
    Clips the video based on segment info using FFmpeg.

    Args:
        video_path (str): Path to the input video file.
        segment_info (dict): Dictionary containing start frame as key and (state, duration) as value.
        output_dir (str): Directory to save the clipped videos.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the video frame rate
    frame_rate = get_video_frame_rate(video_path)

    for start_frame, (state, duration) in segment_info.items():
        if state != PoseState.MOVEMENT:
            continue

        start_time = start_frame / frame_rate
        end_time = (start_frame + duration + int(frame_rate)) / frame_rate
        print("Start Time:", start_time)
        print("End Time:", end_time)
        print("Frame Rate:", frame_rate)

        output_file = os.path.join(output_dir, f"{state.value}_{start_frame}_{start_frame + duration + int(frame_rate)}.mp4")

        # FFmpeg command to extract the segment
        cmd = [
            "ffmpeg", "-y",  # Overwrite output file if it exists
            "-i", video_path,
            "-ss", f"{start_time:.2f}",
            "-to", f"{end_time:.2f}",
            "-c:v", "libx264",  # Video codec
            "-preset", "fast",  # Encoding speed
            "-crf", "23",  # Constant Rate Factor (quality)
            output_file
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error clipping video for segment {start_frame}: {result.stderr}")
        else:
            print(f"Segment saved: {output_file}")

def get_video_frame_rate(video_path):
    """
    Get the frame rate of a video using FFprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Frame rate of the video.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f"Error fetching frame rate: {result.stderr}")

    # Calculate frame rate (e.g., "30/1" -> 30.0)
    frame_rate_str = result.stdout.strip()
    num, denom = map(int, frame_rate_str.split("/"))
    return num / denom