import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

keypoint_names = [
    "nose", "left eye (inner)", "left eye", "left eye (outer)", "right eye (inner)",
    "right eye", "right eye (outer)", "left ear", "right ear", "mouth (left)",
    "mouth (right)", "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left pinky", "right pinky", "left index",
    "right index", "left thumb", "right thumb", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle", "left heel",
    "right heel", "left foot index", "right foot index"
]

selected_keypoints = [0,7,8,11,12,13,14,15,16,23,24,25,26,27,28]

connections = [
    (0,1), (0,2), # Nose to Ears
    (3,5), (4,6), # Shoulders to Elbows
    (5,7), (6,8), # Elbows to Wrists
    (9,11), (10,12), # Hips to Knees
    (11,13), (12,14), # Knees to Ankles
    (3, 4), (4, 10), (10, 9), (9, 3) # body
]

# for cosine similarity
vector_list = [
    (1, 2),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (9, 11),
    (10, 12),
    (11, 13),
    (12, 14)
]


# Get keypoints data & bounded box size from 1 frame
def get_keypoints_and_boxsize(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    keypoints = []
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in selected_keypoints:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

    xmin, xmax, ymin, ymax, zmin, zmax = 0, 0, 0, 0, 0, 0
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            if xmin == 0:
                xmin, ymin, zmin = landmark.x, landmark.y, landmark.z
            
            else:
                xmin, xmax, ymin, ymax, zmin, zmax = min(xmin, landmark.x), max(xmax, landmark.x), min(ymin, landmark.y), max(ymax, landmark.y), min(zmin, landmark.z), max(zmax, landmark.z)
    
    boxsize = (xmin, xmax, ymin, ymax, zmin, zmax)
    boxsize = [boxsize[2 * i + 1] - boxsize[2 * i] for i in range(3)]

    return keypoints, boxsize


# Calculate OKS value from 2 keypoints data from each data
def oks(gt, preds, idx, boxsize):
    sigmas = np.array([.026, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
    dx = gt[0] - preds[0]
    dy = gt[1] - preds[1]
    bbox_gt = boxsize[0] ** 2 + boxsize[1] ** 2
    kp_c = sigmas[idx]
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * (bbox_gt) * (kp_c**2)))


# Make cosine similarity to percent form
def cosine_similarity_to_percentage(similarity_list):
    similarity = np.mean(similarity_list)
    return (similarity + 1) * 50


# Calculate cosine similarity from each keypoint data
def cos_sim_w_keypoint(keypoints1, keypoints2):
    global vector_list
    cos_sim_list = []

    for vector in vector_list:
        z_num = 2
        idx1, idx2 = vector
        vec1 = (keypoints1[idx2][:z_num] - keypoints1[idx1][:z_num]).reshape(1, -1)
        vec2 = (keypoints2[idx2][:z_num] - keypoints2[idx1][:z_num]).reshape(1, -1)
        sim_value = cosine_similarity(vec1, vec2)
        cos_sim_list.append(sim_value)
    
    return cos_sim_list


# Calculate OKS & Cosine similarity from each keypoint data
def weighted_similarity(keypoints1, keypoints2, boxsize):
    if keypoints1.shape != keypoints2.shape:
        print(keypoints1.shape, keypoints2.shape)
        raise ValueError("Keypoint shapes do not match!")
    
    oks_list = []
    for i in range(len(keypoints1)):
        oks_list.append(oks(keypoints1[i][:3], keypoints2[i][:3], i, boxsize))

    cos_sim_list = cos_sim_w_keypoint(keypoints1, keypoints2)

    return cosine_similarity_to_percentage(np.mean(cos_sim_list)), (np.mean(oks_list)) * 100


# Make mean coordinate data from keypoints list
def mean_value_of_keypoints(keypoints):
    mean_of_keypoints = np.zeros_like(keypoints[0])
    for key in keypoints:
        mean_of_keypoints += key

    mean_of_keypoints /= len(keypoints)
    return mean_of_keypoints


if __name__ == "__main__":
    video_path1 = "normalized_video1.mp4"
    video_path2 = "normalized_video2_30fps.mp4"

    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    frame_count = -1

    # List of OKS & Cosine similarity from each frame
    okslist = []
    cos_list = []

    # List of OKS & Cosine similarity from every 15 frame
    okslist_mean = []
    cos_list_mean = []

    # Make keypoint list
    list_keypoints1 = []
    list_keypoints2 = []

    while cap1.isOpened() and cap2.isOpened():
        frame_count += 1
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            keypoints1, boxsize = get_keypoints_and_boxsize(frame1)
            keypoints2, _ = get_keypoints_and_boxsize(frame2)

            list_keypoints1.append(keypoints1)
            list_keypoints2.append(keypoints2)

            frame_with_overlay = draw_keypoints_and_connections(frame2, keypoints1, (0, 0, 255))              # Draw Origin keypoint on User's video
            frame_with_overlay = draw_keypoints_and_connections(frame_with_overlay, keypoints2, (0, 255, 0))  # Draw User's keypoint on User's video
            frame1_with_overlay = draw_keypoints_and_connections(frame1, keypoints1, (0, 0, 255))             # Draw Origin keypoint on Original video

            frame_with_overlay = cv2.resize(frame_with_overlay, (480, 640), interpolation=cv2.INTER_CUBIC)    # Resize Each Video to watch
            frame1_with_overlay = cv2.resize(frame1_with_overlay, (480, 640), interpolation=cv2.INTER_CUBIC)  # Resize Each Video to watch

            similarity, oks_percent = weighted_similarity(keypoints1, keypoints2, boxsize) # Calculate Scores from each frame
            okslist.append(oks_percent)
            cos_list.append(similarity)
            print(f"Frame {frame_count+1}: Weighted similarity between keypoints1 and video: {similarity}")
            print(f"Frame {frame_count+1}: Weighted similarity between keypoints1 and video: {oks_percent}")

            if len(list_keypoints1) == 15:
                mean_keypoints1 = mean_value_of_keypoints(list_keypoints1)
                mean_keypoints2 = mean_value_of_keypoints(list_keypoints2)

                similarity_mean, oks_percent_mean = weighted_similarity(mean_keypoints1, mean_keypoints2, boxsize) # Calculate Scores from each mean frame
                okslist_mean.append(oks_percent_mean)
                cos_list_mean.append(similarity_mean)
                print(f"Frame {frame_count+1}: Weighted similarity between mean keypoints1 and video: {similarity_mean}")
                print(f"Frame {frame_count+1}: Weighted similarity between mean keypoints1 and video: {oks_percent_mean}")

                list_keypoints1 = []
                list_keypoints2 = []

            cv2.imshow('Overlay Keypoints', frame_with_overlay)
            cv2.imshow('original', frame1_with_overlay)
            
            # Press 'q' to exit the loop and close the video window
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        else:
            break
    
    
    print(f'oks = {np.mean(okslist)}, cos = {np.mean(cos_list)}')           # Print the score from each frame
    print(f'oks = {np.mean(okslist_mean)}, cos = {np.mean(cos_list_mean)}') # Print the score from every 15 frame
    cap2.release()
    cv2.destroyAllWindows()