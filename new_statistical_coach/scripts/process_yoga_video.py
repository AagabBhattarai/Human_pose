import torch
import matplotlib.pyplot as plt
import os

from src.pose_analysis.state_machine import YogaPoseStateMachine, PoseState
from src.pose_analysis.biomechanical_feature_extractor import BiomechanicalFeatureExtractor
from src.pose_analysis.video_processor import keypoint_extractor
from src.utils.video_utils import clip_video_with_ffmpeg
import numpy as np

def process_yoga_video(video_path, 
                       movement_threshold=0.3, 
                       hold_threshold=0.1, 
                       hold_duration=30,
                       give_segmented_clip= False):
    """
    Process a yoga video and extract pose segments.
    
    Args:
        video_path (str): Path to the input video
        movement_threshold (float): Threshold for movement detection
        hold_threshold (float): Threshold for hold state
        hold_duration (int): Duration to consider a hold
    """
    # Extract keypoints
    # Check if features exist in features_for_dev directory
    features_dev_path = os.path.join('..', 'features_for_dev', 'first.pt')
    
    if os.path.exists(features_dev_path):
        # Load pre-computed features if they exist
        features = torch.load(features_dev_path, weights_only=False)
    else:
        # Extract keypoints if no pre-computed features found
        features = keypoint_extractor(video_path)
        features = torch.from_numpy(features)
    
    # Initialize extractors
    extractor = BiomechanicalFeatureExtractor()
    
    # Compute velocity
    velocity = extractor.extract_features(features)["Joint Acceleration"]
    
    # Compute velocity magnitude
    velocity_magnitude = torch.sqrt(velocity[..., 0]**2 + velocity[...,1]**2 + velocity[...,2]**2)
    print("SHAPE OF VELOCITY MAGNITUDE:",velocity_magnitude.shape)
    smoothed_velocity = velocity_magnitude.sum(dim=-1)
    print("SHAPE OF SMOOTHED VELOCITY:",smoothed_velocity.shape)
    smoothed_velocity = smoothed_velocity.clamp_(min=0, max=1.5).pow_(2).clamp_(max=1.5)
    
    # Initialize state machine
    state_machine = YogaPoseStateMachine(
        movement_threshold=movement_threshold,
        hold_threshold=hold_threshold,
        hold_duration=hold_duration 
    )
    
    # Process each frame
    hold_frame_number = None
    for frame_number, frame_velocity in enumerate(smoothed_velocity):
        frame_state = state_machine.process_frame(frame_velocity, extractor, features)
        if frame_state['state'] == PoseState.HOLD:
            hold_frame_number = frame_number
            break
    
    if hold_frame_number is not None:
        # Extract angles for the specific hold frame
        hold_frame_features = features[hold_frame_number-5: hold_frame_number]
        print("HOLD FRAME FEATURES SHAPE:", hold_frame_features.shape)
        angles_dict = extractor.extract_features(
            hold_frame_features, 
            return_angles_dict=True
        )['Angles Dictionary']
    #    extractor.compute_joint_angles() 
        print(f"Joint Angles at Frame {hold_frame_number}:")
        for joint, angle in angles_dict.items():
            print(f"{joint}: {angle}")
    
    # Visualize velocity
    plt.figure(figsize=(10, 5))
    plt.plot(state_machine.velocity_whole_buffer, label='Smoothed Out Velocity')
    plt.plot(smoothed_velocity[:hold_frame_number], alpha=0.5, label='Raw Velocity Data', color='orange')
    plt.title('Velocity Whole Buffer')
    plt.xlabel('Frame')
    plt.ylabel('Velocity')
    plt.grid()
    plt.legend()
    plt.xticks(range(0, len(smoothed_velocity[:hold_frame_number]), 8))  # Set x-axis ticks at intervals of 5
    plt.yticks(np.arange(0, 1.6, 0.1))
    plt.show()
    
    # Clip video segments
    segment_info = state_machine.get_state_history()
    print(segment_info)
    if give_segmented_clip:
        output_dir = f"clipped_videos_{os.path.basename(video_path)}"
        clip_video_with_ffmpeg(video_path, segment_info, output_dir)
    
    return state_machine, angles_dict

def extract_angles_from_video_segment(video_path, start_frame, end_frame, features_dev_path=None):
    features_dev_path = os.path.join('..', 'features_for_dev', 'first.pt')
    
    if os.path.exists(features_dev_path):
        # Load pre-computed features if they exist
        features = torch.load(features_dev_path, weights_only=False)
    else:
        # Extract keypoints if no pre-computed features found
        features = keypoint_extractor(video_path)
        features = torch.from_numpy(features)
    
    # Initialize extractors
    extractor = BiomechanicalFeatureExtractor()

    hold_frame_features = features[start_frame: end_frame]
    print("HOLD FRAME FEATURES SHAPE:", hold_frame_features.shape)
    angles_dict = extractor.extract_features(
        hold_frame_features, 
        return_angles_dict=True
    )['Angles Dictionary']

    print(f"Joint Angles at Frame {start_frame}:")
    # for joint, angle in angles_dict.items():
    #     print(f"{joint}: {angle}")
    
    return angles_dict


def main():
    video_path = 'tiktok_data/mountain/first.mp4'
    process_yoga_video(video_path)

if __name__ == "__main__":
    main()