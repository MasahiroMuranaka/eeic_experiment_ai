import cv2
import numpy as np
import os

def create_dummy_video(path, frames=30, width=640, height=360):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')# type: ignore
    writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
    for i in range(frames):
        # Create a frame with a moving circle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (int(width * i / frames), height // 2), 50, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    print(f"Created dummy video: {path}")

def create_dummy_npz(path, frames=30, num_area=8):
    # Create dummy probabilities: one area is 1.0, others are 0.0, moving across frames
    probs = np.zeros((frames, num_area), dtype=np.float32)
    for i in range(frames):
        probs[i, i % num_area] = 1.0
    np.savez(path, probs=probs)
    print(f"Created dummy npz: {path}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add parent directory to sys.path to import viz2
    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir.parent))

    video_path = str(this_dir / "test_video.mp4")
    model_npz = str(this_dir / "test_model.npz")
    answer_npz = str(this_dir / "test_answer.npz")
    
    create_dummy_video(video_path)
    create_dummy_npz(model_npz)
    create_dummy_npz(answer_npz)
    
    # Now run viz2.py logic
    from viz2 import create_viz_videos
    
    try:
        create_viz_videos(video_path, model_npz, answer_npz)
        print("Test execution successful!")
    except Exception as e:
        print(f"Test execution failed: {e}")
    finally:
        # Cleanup
        # for p in [video_path, model_npz, answer_npz]:
        #     if os.path.exists(p):
        #         os.remove(p)
        pass
