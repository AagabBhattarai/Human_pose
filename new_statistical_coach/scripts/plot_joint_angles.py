import matplotlib.pyplot as plt

def plot_joint_angles(joint_angle_values):
    frames = list(joint_angle_values.keys())

    right_shoulder_angles = [joint_angle_values[frame]['right_shoulder_3d'].item() for frame in frames]
    left_shoulder_angles = [joint_angle_values[frame]['left_shoulder_3d'].item() for frame in frames]
    # right_ankle_angles = [joint_angle_values[frame]['right_ankle_3d'].item() for frame in frames]
    # # # left_ankle_angles = [joint_angle_values[frame]['left_ankle_3d'].item() for frame in frames]
    # # right_elbow_angles = [joint_angle_values[frame]['right_elbow_3d'].item() for frame in frames]
    # left_elbow_angles = [joint_angle_values[frame]['left_elbow_3d'].item() for frame in frames]

    plt.figure(figsize=(12, 6))
    plt.plot(frames, right_shoulder_angles, label='Right Shoulder', color='r')
    plt.plot(frames, left_shoulder_angles, label='Left Shoulder', color='g')
    # plt.plot(frames, right_ankle_angles, label='Right Ankle', color='b')
    # # # plt.plot(frames, left_ankle_angles, label='Left Ankle', color='y')
    # # plt.plot(frames, right_elbow_angles, label='Right Elbow', color='m')
    # plt.plot(frames, left_elbow_angles, label='Left Elbow', color='c')

    plt.xticks(range(0, len(frames), 25))
    plt.xlabel('Frame Number')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles Over Frames')
    plt.legend()
    plt.grid(True)
    plt.show()