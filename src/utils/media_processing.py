import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def compute_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def resize_with_padding(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = resized
    return padded


# --- Constants for Image and Video Processing ---

# General Constants
IMAGE_PROCESS_SIZE = 640

# Image Analysis Constants
IMAGE_IOU_INTERACTION_THRESHOLD = 0.1
IMAGE_NORMALIZED_DISTANCE_THRESHOLD = 0.8
IMAGE_INSECT_CONF_THRESHOLD = 0.4
IMAGE_FLOWER_CONF_THRESHOLD = 0.6

# Video Analysis Constants
VIDEO_IOU_FRAME_INTERACTION_THRESHOLD = 0.05
VIDEO_NORMALIZED_DISTANCE_THRESHOLD = 0.8
VIDEO_INSECT_CONF_THRESHOLD = 0.2
VIDEO_FLOWER_CONF_THRESHOLD = 0.3

# DeepSort Parameters for Video
DEEPSORT_MAX_AGE = 150
DEEPSORT_N_INIT = 7
DEEPSORT_NN_BUDGET = 50

# Track Filtering and Merging Parameters for Video
MIN_TRACK_FRAMES = 5
MIN_CONF_RATIO = 0.2
MERGE_IOU_THRESHOLD = 0.3
SUFFICIENT_VISITS_THRESHOLD = 5


def calculate_distance(bbox1, bbox2):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    # Normalize by average bounding box diagonal
    diag1 = np.sqrt((bbox1[2] - bbox1[0]) ** 2 + (bbox1[3] - bbox1[1]) ** 2)
    diag2 = np.sqrt((bbox2[2] - bbox2[0]) ** 2 + (bbox2[3] - bbox2[1]) ** 2)
    avg_diag = (diag1 + diag2) / 2
    normalized_distance = distance / avg_diag
    return distance, normalized_distance


def detect_interactions(
    insect_tracks, flower_tracks, iou_threshold, normalized_distance_threshold
):
    interactions = []
    interaction_pairs = set()  # To avoid duplicate interactions for the same pair
    flower_visits = {}  # {flower_id: set of insect_ids}

    for insect_id, (
        insect_frames,
        insect_total_conf,
        insect_bbox,
    ) in insect_tracks.items():
        for flower_id, (
            flower_frames,
            flower_total_conf,
            flower_bbox,
        ) in flower_tracks.items():
            distance, normalized_distance = calculate_distance(insect_bbox, flower_bbox)
            iou = compute_iou(insect_bbox, flower_bbox)

            is_interaction = False
            interaction_type = ""

            if iou > iou_threshold:
                is_interaction = True
                interaction_type = "direct_contact"
            elif normalized_distance < normalized_distance_threshold:
                is_interaction = True
                interaction_type = "close_proximity"

            if is_interaction:
                pair_key = (min(insect_id, flower_id), max(insect_id, flower_id))
                if pair_key not in interaction_pairs:
                    interaction_pairs.add(pair_key)
                    # Calculate average confidence for the interaction
                    avg_insect_conf = (
                        insect_total_conf / insect_frames if insect_frames > 0 else 0
                    )
                    avg_flower_conf = (
                        flower_total_conf / flower_frames if flower_frames > 0 else 0
                    )
                    combined_confidence = (avg_insect_conf + avg_flower_conf) / 2

                    interactions.append(
                        {
                            "insect_id": insect_id,
                            "flower_id": flower_id,
                            "interaction_type": interaction_type,
                            "distance": distance,
                            "normalized_distance": normalized_distance,
                            "iou": iou,
                            "insect_frames": insect_frames,
                            "flower_frames": flower_frames,
                            "confidence_score": combined_confidence,
                        }
                    )
                    # Record flower visits
                    if flower_id not in flower_visits:
                        flower_visits[flower_id] = set()
                    flower_visits[flower_id].add(insect_id)

    # Number of visits per flower and sufficient pollination
    flower_visit_counts = {
        flower_id: len(insects) for flower_id, insects in flower_visits.items()
    }
    sufficiently_pollinated = sum(
        1
        for count in flower_visit_counts.values()
        if count > SUFFICIENT_VISITS_THRESHOLD
    )
    return interactions, len(interactions), flower_visit_counts, sufficiently_pollinated


def analyze_interaction_patterns(
    interactions, flower_visit_counts, sufficiently_pollinated, total_flowers
):
    if not interactions:
        return {
            "total_interactions": 0,
            "avg_distance": 0,
            "direct_contacts": 0,
            "close_proximities": 0,
            "avg_confidence": 0,
            "flower_visit_counts": {},
            "sufficiently_pollinated_flowers": 0,
            "sufficient_pollination_percentage": 0.0,
        }

    total_interactions = len(interactions)
    direct_contacts = sum(
        1 for i in interactions if i["interaction_type"] == "direct_contact"
    )
    close_proximities = sum(
        1 for i in interactions if i["interaction_type"] == "close_proximity"
    )
    avg_normalized_distance = np.mean([i["normalized_distance"] for i in interactions])
    avg_confidence = np.mean([i["confidence_score"] for i in interactions])
    sufficient_pollination_percentage = (
        ((sufficiently_pollinated) / total_flowers * 100) if total_flowers > 0 else 0.0
    )

    return {
        "total_interactions": total_interactions,
        "avg_distance": avg_normalized_distance,
        "direct_contacts": direct_contacts,
        "close_proximities": close_proximities,
        "avg_confidence": avg_confidence,
        "flower_visit_counts": flower_visit_counts,
        "sufficiently_pollinated_flowers": sufficiently_pollinated,
        "sufficient_pollination_percentage": sufficient_pollination_percentage,
    }


def detect_frame_interactions(
    current_insects, current_flowers, iou_threshold, normalized_distance_threshold
):
    frame_interactions = []

    for insect_id, insect_bbox in current_insects.items():
        for flower_id, flower_bbox in current_flowers.items():
            distance, normalized_distance = calculate_distance(insect_bbox, flower_bbox)
            iou = compute_iou(insect_bbox, flower_bbox)

            if iou > iou_threshold:
                frame_interactions.append(
                    {
                        "insect_id": insect_id,
                        "flower_id": flower_id,
                        "interaction_type": "direct_contact",
                        "distance": distance,
                        "iou": iou,
                    }
                )
            elif normalized_distance < normalized_distance_threshold:
                frame_interactions.append(
                    {
                        "insect_id": insect_id,
                        "flower_id": flower_id,
                        "interaction_type": "close_proximity",
                        "distance": distance,
                        "normalized_distance": normalized_distance,
                        "iou": iou,
                    }
                )
    return frame_interactions


def process_media(file, model_path):
    interactions = []
    interaction_count = 0
    flower_visit_counts = {}
    sufficiently_pollinated = 0

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    total_insects, total_flowers = 0, 0
    sample_frame = None

    if file.name.endswith((".mp4", ".avi")):  # Video Processing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            temp.write(file.read())
            temp_file = temp.name

        cap = cv2.VideoCapture(temp_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        # Initialize DeepSort with video-specific parameters
        tracker = DeepSort(
            max_age=DEEPSORT_MAX_AGE,
            n_init=DEEPSORT_N_INIT,
            nn_budget=DEEPSORT_NN_BUDGET,
            embedder="mobilenet",  # Using mobilenet as it's common and should be available
            half=False,  # Set to False for robustness, can be True if performance is critical and hardware supports
            bgr=True,
        )

        # Dictionaries to store track lifetimes, confidences, and last bounding box
        # track_id: [frame_count, total_confidence, last_bbox]
        insect_tracks = {}
        flower_tracks = {}

        # Process each frame
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame: Resize for better detection
            processed_frame = resize_with_padding(
                frame, (IMAGE_PROCESS_SIZE, IMAGE_PROCESS_SIZE)
            )

            # YOLO detection with video-specific confidence thresholds
            results = model(
                processed_frame,
                imgsz=IMAGE_PROCESS_SIZE,
                conf=min(VIDEO_INSECT_CONF_THRESHOLD, VIDEO_FLOWER_CONF_THRESHOLD),
                iou=0.5,
                device=device,
            )

            # Prepare detections for DeepSort
            deepsort_detections = []
            current_insects = (
                {}
            )  # Detections in current frame for frame-level interactions
            current_flowers = (
                {}
            )  # Detections in current frame for frame-level interactions

            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Filter detections by class-specific confidence
                if (cls == 1 and conf >= VIDEO_INSECT_CONF_THRESHOLD) or (
                    cls == 0 and conf >= VIDEO_FLOWER_CONF_THRESHOLD
                ):
                    # Convert to DeepSort format [x, y, w, h]
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    deepsort_detections.append(([x, y, w, h], conf, cls))

            # Update the tracker
            tracks = tracker.update_tracks(deepsort_detections, frame=processed_frame)

            # Process confirmed tracks and update aggregated track data
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    cls = track.det_class
                    conf = (
                        track.det_conf if track.det_conf else 0.0
                    )  # Use 0.0 if confidence is None
                    bbox = track.to_tlbr()  # (x1, y1, x2, y2)

                    if cls == 1:  # Insect
                        current_insects[track_id] = bbox  # Store for frame interactions
                        if track_id not in insect_tracks:
                            insect_tracks[track_id] = [
                                0,
                                0.0,
                                bbox,
                            ]  # [frames_seen, total_confidence, last_bbox]
                        insect_tracks[track_id][0] += 1
                        insect_tracks[track_id][1] += conf
                        insect_tracks[track_id][2] = bbox  # Update last bbox

                    elif cls == 0:  # Flower
                        current_flowers[track_id] = bbox  # Store for frame interactions
                        if track_id not in flower_tracks:
                            flower_tracks[track_id] = [
                                0,
                                0.0,
                                bbox,
                            ]  # [frames_seen, total_confidence, last_bbox]
                        flower_tracks[track_id][0] += 1
                        flower_tracks[track_id][1] += conf
                        flower_tracks[track_id][2] = bbox  # Update last bbox

            # Capture sample frame with track IDs and interactions (from middle of video)
            if i == frame_count // 2:
                sample_frame = results[0].plot()  # YOLO's plot function
                sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)

                # Draw track IDs on sample frame
                for track in tracks:
                    if track.is_confirmed():
                        bbox = track.to_tlbr()
                        track_id = track.track_id
                        cls = track.det_class
                        cv2.putText(
                            sample_frame,
                            f"ID: {track_id} ({'insect' if cls == 1 else 'flower'})",
                            (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Draw interaction lines on sample frame using per-frame detections
                frame_interactions = detect_frame_interactions(
                    current_insects,
                    current_flowers,
                    VIDEO_IOU_FRAME_INTERACTION_THRESHOLD,
                    VIDEO_NORMALIZED_DISTANCE_THRESHOLD,
                )
                for interaction in frame_interactions:
                    if (
                        interaction["insect_id"] in current_insects
                        and interaction["flower_id"] in current_flowers
                    ):
                        insect_bbox = current_insects[interaction["insect_id"]]
                        flower_bbox = current_flowers[interaction["flower_id"]]

                        insect_center = (
                            int((insect_bbox[0] + insect_bbox[2]) / 2),
                            int((insect_bbox[1] + insect_bbox[3]) / 2),
                        )
                        flower_center = (
                            int((flower_bbox[0] + flower_bbox[2]) / 2),
                            int((flower_bbox[1] + flower_bbox[3]) / 2),
                        )

                        if interaction["interaction_type"] == "direct_contact":
                            color = (0, 0, 255)  # Red
                        elif interaction["interaction_type"] == "close_proximity":
                            color = (255, 0, 0)  # Blue
                        else:
                            continue

                        cv2.line(sample_frame, insect_center, flower_center, color, 2)
                        cv2.putText(
                            sample_frame,
                            interaction["interaction_type"].replace("_", " "),
                            (insect_center[0], insect_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

            progress.progress((i + 1) / frame_count)

        cap.release()
        os.remove(temp_file)

        # --- Post-processing for Video: Filtering and Merging Tracks ---
        valid_insect_tracks = {}
        valid_flower_tracks = {}

        # Filter tracks by lifetime and average confidence
        for track_id, (frames, total_conf, bbox) in insect_tracks.items():
            avg_conf = total_conf / frames if frames > 0 else 0
            if frames >= MIN_TRACK_FRAMES and avg_conf >= MIN_CONF_RATIO:
                valid_insect_tracks[track_id] = (frames, total_conf, bbox)

        for track_id, (frames, total_conf, bbox) in flower_tracks.items():
            avg_conf = total_conf / frames if frames > 0 else 0
            if frames >= MIN_TRACK_FRAMES and avg_conf >= MIN_CONF_RATIO:
                valid_flower_tracks[track_id] = (frames, total_conf, bbox)

        # Merge highly overlapping tracks (to correct for potential ID switches)
        merged_insect_tracks = {}
        for track_id, (frames, total_conf, bbox) in valid_insect_tracks.items():
            merged = False
            for existing_id, (
                existing_frames,
                existing_total_conf,
                existing_bbox,
            ) in merged_insect_tracks.items():
                if compute_iou(bbox, existing_bbox) > MERGE_IOU_THRESHOLD:
                    # Merge by keeping the track with higher total confidence
                    if total_conf > existing_total_conf:
                        merged_insect_tracks[existing_id] = (frames, total_conf, bbox)
                    merged = True
                    break
            if not merged:
                merged_insect_tracks[track_id] = (frames, total_conf, bbox)

        merged_flower_tracks = {}
        for track_id, (frames, total_conf, bbox) in valid_flower_tracks.items():
            merged = False
            for existing_id, (
                existing_frames,
                existing_total_conf,
                existing_bbox,
            ) in merged_flower_tracks.items():
                if compute_iou(bbox, existing_bbox) > MERGE_IOU_THRESHOLD:
                    # Merge by keeping the track with higher total confidence
                    if total_conf > existing_total_conf:
                        merged_flower_tracks[existing_id] = (frames, total_conf, bbox)
                    merged = True
                    break
            if not merged:
                merged_flower_tracks[track_id] = (frames, total_conf, bbox)

        # Calculate overall interactions and analysis based on merged tracks
        (
            interactions,
            interaction_count,
            flower_visit_counts,
            sufficiently_pollinated,
        ) = detect_interactions(
            merged_insect_tracks,
            merged_flower_tracks,
            VIDEO_IOU_FRAME_INTERACTION_THRESHOLD,
            VIDEO_NORMALIZED_DISTANCE_THRESHOLD,
        )

        interaction_analysis = analyze_interaction_patterns(
            interactions,
            flower_visit_counts,
            sufficiently_pollinated,
            len(merged_flower_tracks),  # Use count of merged flowers for percentage
        )

        total_insects = len(merged_insect_tracks)
        total_flowers = len(merged_flower_tracks)

    else:  # Image Processing
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = resize_with_padding(
            image, (IMAGE_PROCESS_SIZE, IMAGE_PROCESS_SIZE)
        )  # Preprocess image

        # YOLO detection with image-specific confidence thresholds
        results = model(
            image,
            imgsz=IMAGE_PROCESS_SIZE,
            conf=min(IMAGE_INSECT_CONF_THRESHOLD, IMAGE_FLOWER_CONF_THRESHOLD),
            device=device,
        )

        total_insects = sum(
            1
            for box in results[0].boxes
            if int(box.cls[0]) == 1 and box.conf[0] >= IMAGE_INSECT_CONF_THRESHOLD
        )
        total_flowers = sum(
            1
            for box in results[0].boxes
            if int(box.cls[0]) == 0 and box.conf[0] >= IMAGE_FLOWER_CONF_THRESHOLD
        )
        sample_frame = results[0].plot()
        sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)

        # Store detections as lists for image interaction detection
        insect_detections = []
        flower_detections = []

        for box in results[0].boxes:
            bbox = (
                box.xyxy[0][0].item(),
                box.xyxy[0][1].item(),
                box.xyxy[0][2].item(),
                box.xyxy[0][3].item(),
            )
            cls = int(box.cls[0])
            conf = box.conf[0].item()

            if cls == 1 and conf >= IMAGE_INSECT_CONF_THRESHOLD:
                insect_detections.append((1, conf, bbox))  # (dummy_frames, conf, bbox)

            elif cls == 0 and conf >= IMAGE_FLOWER_CONF_THRESHOLD:
                flower_detections.append((1, conf, bbox))  # (dummy_frames, conf, bbox)

        # Convert to dictionary format for detect_interactions (using dummy IDs for image)
        # For images, each detection is treated as a "track" of 1 frame for interaction calculation
        (
            interactions,
            interaction_count,
            flower_visit_counts,
            sufficiently_pollinated,
        ) = detect_interactions(
            {i: det for i, det in enumerate(insect_detections)},
            {i: det for i, det in enumerate(flower_detections)},
            IMAGE_IOU_INTERACTION_THRESHOLD,
            IMAGE_NORMALIZED_DISTANCE_THRESHOLD,
        )

        interaction_analysis = analyze_interaction_patterns(
            interactions, flower_visit_counts, sufficiently_pollinated, total_flowers
        )

        # Draw interaction lines on sample frame for image
        # Need to map the interaction's insect_id/flower_id back to the original detections
        # This assumes the enumerate(insect_detections) mapping holds
        insect_detection_map = {i: det[2] for i, det in enumerate(insect_detections)}
        flower_detection_map = {i: det[2] for i, det in enumerate(flower_detections)}

        for interaction in interactions:
            insect_bbox = insect_detection_map.get(interaction["insect_id"])
            flower_bbox = flower_detection_map.get(interaction["flower_id"])

            if insect_bbox and flower_bbox:
                insect_center = (
                    int((insect_bbox[0] + insect_bbox[2]) / 2),
                    int((insect_bbox[1] + insect_bbox[3]) / 2),
                )
                flower_center = (
                    int((flower_bbox[0] + flower_bbox[2]) / 2),
                    int((flower_bbox[1] + flower_bbox[3]) / 2),
                )

                color = (
                    (0, 0, 255)  # Red for direct contact
                    if interaction["interaction_type"] == "direct_contact"
                    else (255, 0, 0)  # Blue for close proximity
                )
                cv2.line(sample_frame, insect_center, flower_center, color, 2)
                cv2.putText(
                    sample_frame,
                    interaction["interaction_type"].replace("_", " "),
                    (insect_center[0], insect_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

    return {
        "total_insects": total_insects,
        "total_flowers": total_flowers,
        "sample_frame": sample_frame,
        "interactions": interactions,
        "interaction_count": interaction_count,
        "interaction_analysis": interaction_analysis,
        "flower_visit_counts": flower_visit_counts,
        "sufficiently_pollinated_flowers": sufficiently_pollinated,
    }
