import os
import glob
import json
from PIL import Image

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

if __name__ == "__main__":
    camera_num = 0
    scene_names = os.listdir(os.path.join("./images", f"image_{camera_num}"))
    num_area = 8
    print(f"split images to {num_area} areas")

    os.makedirs("answers", exist_ok=True)
    os.makedirs(os.path.join("answers", f"image_{camera_num}"), exist_ok=True)

    for scene_name in scene_names:
        ans_dict = get_answer(scene_name, camera_num, num_area)

        out_path = os.path.join("answers", f"image_{camera_num}", f"{scene_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ans_dict, f, indent=3)