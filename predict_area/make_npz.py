import subprocess
import os

if __name__ == '__main__':
    image_num = 2
    frame_dir = "/mnt/c/Users/angw1/koukizikkenn/AI/images/image_2"
    out_dir = "/mnt/c/Users/angw1/koukizikkenn/AI/npz"
    det_json_dir = "/mnt/c/Users/angw1/koukizikkenn/AI/detections"
    y_json_dir = "/mnt/c/Users/angw1/koukizikkenn/AI/answers/image_2"

    scene_names = ["meyer-green-2019-03-16_0", "nvidia-aud-2019-04-18_0", "packard-poster-session-2019-03-20_0",
                   "packard-poster-session-2019-03-20_1", "packard-poster-session-2019-03-20_2", "stlc-111-2019-04-19_0",
                   "svl-meeting-gates-2-2019-04-08_0", "svl-meeting-gates-2-2019-04-08_1", "tressider-2019-03-16_0",
                   "tressider-2019-03-16_1", "tressider-2019-04-26_2"]
    
    """
    for f in os.listdir(frame_dir):
        if os.path.isdir(os.path.join(frame_dir, f)):
            scene_names.append(f)
    """
    
    for scene in scene_names:
        frame_path = os.path.join(frame_dir, scene)
        out_path = out_dir
        det_json_path = os.path.join(det_json_dir, f"{scene}_image{image_num}.json")
        y_json_path = os.path.join(y_json_dir, f"{scene}.json")

        subprocess.run(
            ["python", "-m", "src.preprocess.cli", "--frames-dir", frame_path, 
             "--out-dir", out_path, "--det-json", det_json_path, "--y-json", y_json_path],
            check=True
        )