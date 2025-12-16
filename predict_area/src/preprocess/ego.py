import cv2
import numpy as np


class EgoMotionTracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.prev_gray = None
        self.prev_pts = None

    def update(self, frame, exclude_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ego_vx, ego_vy = 0.0, 0.0

        if self.prev_gray is not None:
            if self.prev_pts is None or len(self.prev_pts) < 50:
                self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)

            if self.prev_pts is not None and len(self.prev_pts) > 0:
                good_prev_pts = []
                for pt in self.prev_pts:
                    x, y = pt.ravel()
                    is_in_person = False
                    for box in exclude_boxes:
                        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                            is_in_person = True
                            break
                    if not is_in_person:
                        good_prev_pts.append(pt)

                if len(good_prev_pts) == 0:
                    self.prev_pts = None
                else:
                    good_prev_pts = np.array(good_prev_pts, dtype=np.float32)

                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, good_prev_pts, None, **self.lk_params
                    )

                    fb_thresh = 1.5
                    good_prev = []
                    good_next = []
                    for i in range(len(good_prev_pts)):
                        if status is None or status[i][0] == 0:
                            continue
                        p_next = next_pts[i:i + 1]
                        p_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
                            gray, self.prev_gray, p_next, None, **self.lk_params
                        )
                        if st_back is None or st_back[0][0] == 0:
                            continue
                        err = np.linalg.norm(good_prev_pts[i] - p_back[0])
                        if err <= fb_thresh:
                            good_prev.append(good_prev_pts[i])
                            good_next.append(next_pts[i])

                    if len(good_prev) == 0:
                        self.prev_pts = None
                    else:
                        prev_arr = np.array(good_prev, dtype=np.float32).reshape(-1, 1, 2)
                        next_arr = np.array(good_next, dtype=np.float32).reshape(-1, 1, 2)

                        pts_prev = prev_arr.reshape(-1, 2)
                        pts_next = next_arr.reshape(-1, 2)

                        if len(pts_prev) >= 3:
                            M, inliers = cv2.estimateAffinePartial2D(
                                pts_prev, pts_next, method=cv2.RANSAC, ransacReprojThreshold=3.0
                            )
                            if M is not None:
                                ego_vx, ego_vy = float(M[0, 2]), float(M[1, 2])
                                if inliers is not None:
                                    in_mask = inliers.ravel() == 1
                                    self.prev_pts = (pts_next[in_mask] if np.sum(in_mask) > 0 else pts_next).reshape(-1, 1, 2)
                                else:
                                    self.prev_pts = pts_next.reshape(-1, 1, 2)
                            else:
                                motion = pts_next - pts_prev
                                med = np.median(motion, axis=0)
                                ego_vx, ego_vy = float(med[0]), float(med[1])
                                self.prev_pts = pts_next.reshape(-1, 1, 2)
                        else:
                            motion = pts_next - pts_prev
                            med = np.median(motion, axis=0)
                            ego_vx, ego_vy = float(med[0]), float(med[1])
                            self.prev_pts = pts_next.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()
        return float(ego_vx), float(ego_vy)
