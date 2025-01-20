# So, the joint angle data is the format in dictionary as returned by current function
{
  `right_shoulder`: tensor(ANGLES_IN_DEGREE, dtype=Float),
  `left_shoulder` : tensor(ANGLES_IN_DEGREE, dtype=Float),
  ...
}
# Like the following is one instance of such data
{
  'right_shoulder': tensor(174.2156, dtype=torch.float64),
  'left_shoulder': tensor(156.9292, dtype=torch.float64),
  'right_elbow': tensor(152.8174, dtype=torch.float64),
 }


# Additionally for each key in the above data we have additional joint names which can provide info regarding which limbs are involved
joint_configs = {
      # Upper body
      'right_shoulder': {
          'joints': (13, 11, 23),  # right_elbow, right_shoulder, right_hip
          'joint_names': ('right_elbow', 'right_shoulder', 'right_hip'),
          'planes': ['sagittal', 'transverse', 'frontal']
      },
      'left_shoulder': {
          'joints': (14, 12, 24),  # left_elbow, left_shoulder, left_hip
          'joint_names': ('left_elbow', 'left_shoulder', 'left_hip'),
          'planes': ['sagittal', 'transverse', 'frontal']
      },
  ...
}

---
Given this form of data, the goal is to generate pose correction feedback.
So, the task is to form a nice instruction/prompt to prompt a language model of sufficient capacity to be able to understand angles and provide
layperson interpretable feedback.

My naive and rough sketch of what prompt body would include
```txt
  User_Joint_Info:
  Target_Joint_Info:
  Target_Pose_type: <ACTION_TYPE>
  Provide feedback understandable to layperson in 20-30 words.
```

Target_Pose_type value will be extracted from action classification model.

---
What capacity language model will do the job in one shot?
how to test out language model capacity for this job?

 
