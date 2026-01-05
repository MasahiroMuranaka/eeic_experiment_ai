import os
import glob
import json
from PIL import Image
import argparse
import json
import numpy as np
import cv2
from typing import Optional

def calc_answer(box_list, score_list, num_area, im_w):
    used_width = [0]*num_area
    
    boaders = []
    area_width = im_w / num_area
    for i in range(num_area):
        boaders.append(area_width * (i + 1))

    for i in range(len(box_list)):
        box = box_list[i]
        score = score_list[i]
        x = box[0]
        width = box[2]
        
        while width > 0:
            area = int(x // area_width)
            if area >= num_area:
                break

            valid_width = min(boaders[area] - x, width)
            if valid_width < 0:
                break

            used_width[area] += valid_width * score

            width -= valid_width
            x += valid_width
        
    min_index = [0]
    for i in range(1, num_area):
        a = used_width[i] - used_width[min_index[0]]
        if a < 0:
            min_index = [i]
        elif a < 1e-5:
            min_index.append(i)
    
    p = []
    for i in range(num_area):
        if i in min_index:
            p.append(1/len(min_index))
        else:
            p.append(0)

    return p

def get_answer(scene_name, camera_num, num_area):
    detection_path = "./detections"
    image_path = "./images"

    with open(os.path.join(detection_path, f"{scene_name}_image{camera_num}.json"), 'r') as f:
        dataset = json.load(f)["detections"]

    file_names = sorted(
        [os.path.basename(p) for p in glob.glob(os.path.join(image_path, f"image_{camera_num}", scene_name, "*.jpg"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    ans_dict = {}
    im_w = None

    for file_name in file_names:
        file_dataset = dataset.get(file_name)

        if file_dataset is None:
            ans_dict[file_name] = None
            continue

        if len(file_dataset) == 0:
            ans_dict[file_name] = [1/num_area]*num_area
            continue

        box_list   = [d["box"] for d in file_dataset]
        score_list = [d["score"] for d in file_dataset]

        if im_w is None:
            with Image.open(os.path.join(image_path, f"image_{camera_num}", scene_name, file_name)) as img:
                im_w, _ = img.size

        ans_dict[file_name] = calc_answer(box_list, score_list, num_area, im_w)

    return ans_dict


def get_answer_from_video(video_path: str, detections_json: Optional[str], num_area: int):
    """
    動画 (任意の長さ) と detections JSON (キーが "<frame_idx>.jpg" 形式) を受け、
    フレームごとの確率分布を返す（dict: frame_index -> list of probs）

    detections_json が None の場合は全フレームで一様分布を返す。
    """
    # load detections if provided
    dataset = None
    if detections_json is not None:
        with open(detections_json, 'r', encoding='utf-8') as f:
            dataset = json.load(f).get('detections', {})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    frame_idx = 0
    ans_dict = {}
    im_w = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if im_w is None:
            im_w = frame.shape[1]

        file_name = f"{frame_idx}.jpg"
        file_dataset = None
        if dataset is not None:
            file_dataset = dataset.get(file_name)

        if file_dataset is None:
            ans_dict[file_name] = [1/num_area]*num_area
        else:
            if len(file_dataset) == 0:
                ans_dict[file_name] = [1/num_area]*num_area
            else:
                box_list   = [d["box"] for d in file_dataset]
                score_list = [d["score"] for d in file_dataset]
                if im_w is None:
                    im_w = frame.shape[1]
                ans_dict[file_name] = calc_answer(box_list, score_list, num_area, im_w)

        frame_idx += 1

    cap.release()
    return ans_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="input video path (optional)")
    parser.add_argument("--detections", help="detections json path (optional)")
    parser.add_argument("--scene", help="scene name (for image-based mode)")
    parser.add_argument("--camera", type=int, help="camera num (for image-based mode)")
    parser.add_argument("--num_area", type=int, default=8)
    parser.add_argument("--out_json", default="answers.json")
    parser.add_argument("--out_npz", default=None)
    args = parser.parse_args()

    num_area = args.num_area

    if args.video:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        ans_dict = get_answer_from_video(args.video, args.detections, num_area)
        # save json
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(ans_dict, f, indent=2)
        # save npz if requested: convert ordered by frame index
        if args.out_npz:
            # sort keys by integer frame index
            keys = sorted(ans_dict.keys(), key=lambda x: int(os.path.splitext(x)[0]))
            arr = np.stack([np.array(ans_dict[k], dtype=np.float32) for k in keys], axis=0)
            np.savez_compressed(args.out_npz, probs=arr)
            print(f"saved npz to {args.out_npz}")
        print(f"saved json to {args.out_json}")
    else:
        # original image-based batch processing
        for i in range(5):
            camera_num = 2 * i
            scene_names = os.listdir(os.path.join("./images", f"image_{camera_num}"))
            num_area = args.num_area
            print(f"split images to {num_area} areas")

            os.makedirs("answers", exist_ok=True)
            os.makedirs(os.path.join("answers", f"image_{camera_num}"), exist_ok=True)

            for scene_name in scene_names:
                ans_dict = get_answer(scene_name, camera_num, num_area)

                out_path = os.path.join("answers", f"image_{camera_num}", f"{scene_name}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(ans_dict, f, indent=3)