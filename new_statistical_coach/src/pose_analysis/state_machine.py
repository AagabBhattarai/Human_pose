from enum import Enum
from typing import Dict, Tuple
import numpy as np

class PoseState(Enum):
    WAITING = "waiting"
    MOVEMENT = "movement"
    HOLD = "hold"

class YogaPoseStateMachine:
    def __init__(self, 
                 movement_threshold=0.5,
                 hold_threshold=0.2,
                 hold_duration=15,
                 noise_floor=0.1):
        """
        Initialize the Yoga Pose State Machine.
        
        Args:
            movement_threshold (float): Threshold to detect movement
            hold_threshold (float): Threshold to detect hold state
            hold_duration (int): Number of frames to consider a hold
            noise_floor (float): Minimum velocity to ignore
        """
        self.reset_state_tracking(
            movement_threshold, 
            hold_threshold, 
            hold_duration, 
            noise_floor
        )
    
    def reset_state_tracking(self, 
                              movement_threshold,
                              hold_threshold,
                              hold_duration,
                              noise_floor):
        """Reset all state tracking variables."""
        self.state = PoseState.WAITING
        self.movement_threshold = movement_threshold
        self.hold_threshold = hold_threshold
        self.hold_duration = hold_duration
        self.noise_floor = noise_floor
        
        # State tracking variables
        self.frames_in_hold = 0
        self.movement_start_frame = None
        self.hold_start_frame = None
        self.current_frame = 0
        
        # Velocity smoothing
        self.velocity_buffer = []
        self.buffer_size = 5
        self.velocity_whole_buffer = []
        
        # State history tracking
        self.state_history: Dict[int, Tuple[PoseState, int]] = {}
        self.last_state_change_frame = 0
        self.current_state_start_frame = 0
    
    def get_smoothed_velocity(self, velocity):
        """Smooth velocity using a simple moving average."""
        self.velocity_buffer.append(velocity)
        if len(self.velocity_buffer) > self.buffer_size:
            self.velocity_buffer.pop(0)
        
        smoothed_velocity = np.mean(self.velocity_buffer)
        self.velocity_whole_buffer.append(smoothed_velocity)
        return smoothed_velocity
    
    def update_state_history(self, new_state: PoseState) -> None:
        """Update state history when state changes."""
        if new_state != self.state:
            # Calculate duration of the previous state
            duration = self.current_frame - self.current_state_start_frame
            
            # Store the state change with its duration
            self.state_history[self.current_state_start_frame] = (self.state, duration)
            
            # Update tracking variables
            self.last_state_change_frame = self.current_frame
            self.current_state_start_frame = self.current_frame
            self.state = new_state
    
    def process_frame(self, squared_velocity, feature_extractor=None, features=None):
        """
        Process a single frame of squared velocity data.
        
        Args:
            squared_velocity (float): Squared velocity of the frame
            feature_extractor (optional): Extractor to get joint angles
            features (optional): Features to extract joint angles
        
        Returns:
            dict: State information for the current frame
        """
        self.current_frame += 1
        smoothed_velocity = self.get_smoothed_velocity(squared_velocity)
        
        if self.state == PoseState.WAITING:
            if smoothed_velocity > self.movement_threshold:
                self.update_state_history(PoseState.MOVEMENT)
                self.movement_start_frame = self.current_frame
                print(f"Movement detected at frame {self.current_frame}")
                
        elif self.state == PoseState.MOVEMENT:
            if smoothed_velocity < self.hold_threshold:
                self.frames_in_hold += 1
                if self.frames_in_hold >= self.hold_duration:
                    self.update_state_history(PoseState.HOLD)
                    self.hold_start_frame = self.current_frame - self.hold_duration
                    print(f"Hold phase detected at frame {self.hold_start_frame}")
            else:
                self.frames_in_hold = 0
                
        elif self.state == PoseState.HOLD:
            if smoothed_velocity > self.movement_threshold:
                self.update_state_history(PoseState.MOVEMENT)
                self.frames_in_hold = 0
                print(f"Movement detected during hold at frame {self.current_frame}")
            
        # Optional: Print joint angles during hold state
        # if self.state == PoseState.HOLD and feature_extractor and features is not None:
        #     angles_dict = feature_extractor.extract_features(features, return_angles_dict=True)['Angles Dictionary']
        #     print(f"Joint Angles at Frame {self.current_frame}:")
        #     for joint, angle in angles_dict.items():
        #         print(f"{joint}: {angle}")
        
        return {
            'state': self.state,
            'movement_start': self.movement_start_frame,
            'hold_start': self.hold_start_frame,
            'current_frame': self.current_frame,
            'smoothed_velocity': smoothed_velocity,
            'state_history': self.state_history
        }
    
    def get_state_history(self) -> Dict[int, Tuple[PoseState, int]]:
        """
        Return the complete state history.
        
        Returns:
            dict: State history with start frames, states, and durations
        """
        # Update the duration for the current state before returning
        current_duration = self.current_frame - self.current_state_start_frame
        history = self.state_history.copy()
        history[self.current_state_start_frame] = (self.state, current_duration)
        return history
    
    def reset(self):
        """Reset the state machine to initial configuration."""
        self.reset_state_tracking(
            movement_threshold=self.movement_threshold,
            hold_threshold=self.hold_threshold,
            hold_duration=self.hold_duration,
            noise_floor=self.noise_floor
        )
