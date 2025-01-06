Highest Level Overview
--------------------------------
Input Feature -> Biomechanical Feature Extractor -> Biomechnical Features
-------------------------------
  Operations with Biomechanical Features:
    1. Offline storage 
    2. Biomechanical Feature displayable/Visualization of the features (Visualization helps
    in identifying any incorrect calculation of joint features)
    3. Standard Format
-------------------------------
## Format of storage or Standard Format

  ### Types of Biomechanical Features
      -------------------------------------------------
      | 1. Static  Features (angles, positions)       |
      | 2. Dynamic Features (velocities, acceleration)|
      | 3. Balance Metric ? Undefined?
      | 4. Relative Alignment ? Undefined?
      -------------------------------------------------

      `Related Activities`
      - Phase Detection 
        > Entry Phase (Motion/Energy heavy phase)
        > Holding Phase (Static Pose/Unchanging activity)
        > Exit Phase (No clue or no idea what to do here)
      

HOW TO SAVE the biomechanical features
## Naive Idea:
    ```txt

        Body Joint_i = Position {},
        Body Joint_ijk_angle = angle {},
        Velocity Joint_i_wrt_prev_frame = velocity{}
    ```      
    So, in naive idea we compute these and store the features as a JSON file
    this information along with `sequence id` can be stored.
    So like this
    ```JSON
      {
      Sequence ID: 1000(id number),
      Total_Feature_Vector: [Feature_Vector_FRAME1, ..., Feature_Vector_FRAMEK]
      (N dimensional where N is the total number of biomechanical features)
      (K is total number of frames of captured data, K is different for different sample)
      },
      {
        Sequence ID: 1001,
        Total_Feature_Vector: [Feature_Vector_FRAME1, ..., Feature_Vector_FRAMEK]
        (Here again K is Different form sequence id 1000)
      },
      ...
    ```


## Maybe a better idea:
  -------------------------------
  Input Features -> Biomehcanical Feature Extractor -> Output Feature = Tensor
  -------------------------------

  Save the output feature as a `.pt\.pth` file with sequence id as the name of the file.
    > SEQUENCE_ID.pt
  So, the feature vector is processed and stored as a Tensor
   So, how will the Tensor look:

        So first look at how biomechanical features are represented mathematically:

          To represent a single joint position you need 3 floats (x,y,z)
          So for a single joint position this is the max(aka 3) numbers are required.

          Joint Position 
            (FRAME_NO, NUMBER_OF_JOINTS, 3)
          Joint Angles
            (FRAME_NO, NUMBER_OF_TOTAL_ANGLES, 1)
          Joint Velocity
            ~~(FRAME_NO, NUMBER_OF_JOINTS=33, 1)~~
            (FRAME_NO, NUMBER_OF_JOINTS=33, 3)
              > (v_x, v_y, v_z) is required
          Joint Acceleration
            ~~(FRAME_NO, NUMBER_OF_JOINTS=33, 1)~~
            (FRAME_NO, NUMBER_OF_JOINTS=33, 3)
              > (a_x, a_y, a_z) is required

          Joint Position, Velocity and Acceleration can be stored in a single Tensor but for 
          simplicity and modularity let's not do that
                    
      That means for each class of the features, (Here class includes
       (Joint Position, Joint Angles, Joint Velocity, Joint Acceleration)),
      we create a dictionary
      {
        Joint Position    : Tensor(FRAME_NO, NUMBER_OF_JOINTS, 3)
        Joint Angles      : Tensor(FRAME_NO, NUMBER_OF_ANGLES, 1)
        Joint Velocity    : Tensor(FRAME_NO, NUMBER_OF_JOINTS, 3)
        Joint Acceleration: Tensor(FRAME_NO, NUMBER_OF_JOINTS, 3)
      }

      then do torch.save(dict_of_features, BIOMECHANICAL_FEATURES_SAVE_PATH)


-------------------------------

So, the above described save idea stores the computed biomechanical features appropriately such that:
  Features are extendable, specializable(removal of unnecessary joints), faster to load than json?
------------------------------- 

Now how to compute these features?
  Joint Position is already known

  Joint Position -> Known
  Joint Velocity -> Computable by v_x_t+1 = (J_x_t+1 - J_x_t) / STEP
                                  v_y_t+1 = (J_y_t+1 - J_y_t) / STEP
                                  v_z_t+1 = (J_z_t+1 - J_z_t) / STEP

                    Here STEP = 1/fps # BUT FPS is not exactly computable for skeleton data
                    Maybe STEP = 1/TOTAL FRAME 
                    or simply STEP = 1? But what problem can arise if we do this)
                    J_x_t+1 represents joint position in x coord at frame_no t+1

                    Init velocity with (0,0,0)?

  Joint Acceleration -> Computed in the same way instead of joint position use v_x                                  
  Joint Angle -> We take 3 related joints then compute
                 two vector and compute cosine angle between
                 them?

                Challenge is to note down which 3 joint to take.
                WORK required.

  NOTE: JOINT VELOCITY and JOINT ACCELERATION computation method might be incorrect.


---------------------------------  
Highest Level Overview
--------------------------------
Input Feature -> Biomechanical Feature Extractor -> Biomechnical Features
-------------------------------
  Operations with Biomechanical Features:
    1. Offline storage 
    2. Biomechanical Feature displayable/Visualization of the features (Visualization helps
    in identifying any incorrect calculation of joint features)
    3. Standard Format
-------------------------------
## Format of storage or Standard Format

  ### Types of Biomechanical Features
      -------------------------------------------------
      | 1. Static  Features (angles, positions)       |
      | 2. Dynamic Features (velocities, acceleration)|
      | 3. Balance Metric ? Undefined?
      | 4. Relative Alignment ? Undefined?
      -------------------------------------------------

      `Related Activities`
      - Phase Detection 
        > Entry Phase (Motion/Energy heavy phase)
        > Holding Phase (Static Pose/Unchanging activity)
        > Exit Phase (No clue or no idea what to do here)
      

HOW TO SAVE the biomechanical features
## Naive Idea:
    ```txt

        Body Joint_i = Position {},
        Body Joint_ijk_angle = angle {},
        Velocity Joint_i_wrt_prev_frame = velocity{}
    ```      
    So, in naive idea we compute these and store the features as a JSON file
    this information along with `sequence id` can be stored.
    So like this
    ```JSON
      {
      Sequence ID: 1000(id number),
      Total_Feature_Vector: [Feature_Vector_FRAME1, ..., Feature_Vector_FRAMEK]
      (N dimensional where N is the total number of biomechanical features)
      (K is total number of frames of captured data, K is different for different sample)
      },
      {
        Sequence ID: 1001,
        Total_Feature_Vector: [Feature_Vector_FRAME1, ..., Feature_Vector_FRAMEK]
        (Here again K is Different form sequence id 1000)
      },
      ...
    ```


## Maybe a better idea:
  -------------------------------
  Input Features -> Biomehcanical Feature Extractor -> Output Feature = Tensor
  -------------------------------

  Save the output feature as a `.pt\.pth` file with sequence id as the name of the file.
    > SEQUENCE_ID.pt
  So, the feature vector is processed and stored as a Tensor
   So, how will the Tensor look:

        So first look at how biomechanical features are represented mathematically:

          To represent a single joint position you need 3 floats (x,y,z)
          So for a single joint position this is the max(aka 3) numbers are required.

          Joint Position 
            (FRAME_NO, NUMBER_OF_JOINTS, 3)
          Joint Angles
            (FRAME_NO, NUMBER_OF_TOTAL_ANGLES, 1)
          Joint Velocity
            ~~(FRAME_NO, NUMBER_OF_JOINTS=33, 1)~~
            (FRAME_NO, NUMBER_OF_JOINTS=33, 3)
              > (v_x, v_y, v_z) is required
          Joint Acceleration
            ~~(FRAME_NO, NUMBER_OF_JOINTS=33, 1)~~
            (FRAME_NO, NUMBER_OF_JOINTS=33, 3)
              > (a_x, a_y, a_z) is required

          Joint Position, Velocity and Acceleration can be stored in a single Tensor but for 
          simplicity and modularity let's not do that
                    
      That means for each class of the features, (Here class includes
       (Joint Position, Joint Angles, Joint Velocity, Joint Acceleration)),
      we create a dictionary
      {
        Joint Position    : Tensor(FRAME_NO, NUMBER_OF_JOINTS, 3)
        Joint Angles      : Tensor(FRAME_NO, NUMBER_OF_ANGLES, 1)
        Joint Velocity    : Tensor(FRAME_NO, NUMBER_OF_JOINTS, 3)
        Joint Acceleration: Tensor(FRAME_NO, NUMBER_OF_JOINTS, 3)
      }

      then do torch.save(dict_of_features, BIOMECHANICAL_FEATURES_SAVE_PATH)


-------------------------------

So, the above described save idea stores the computed biomechanical features appropriately such that:
  Features are extendable, specializable(removal of unnecessary joints), faster to load than json?
------------------------------- 

Now how to compute these features?
  Joint Position is already known

  Joint Position -> Known
  Joint Velocity -> Computable by v_x_t = (J_x_t+1 - J_x_t) / STEP
                                  v_y_t = (J_y_t+1 - J_y_t) / STEP
                                  v_z_t = (J_z_t+1 - J_z_t) / STEP

                    Here STEP = 1/fps # BUT FPS is not exactly computable for skeleton data
                    Maybe STEP = 1/TOTAL FRAME 
                    or simply STEP = 1? But what problem can arise if we do this)
                    J_x_t+1 represents joint position in x coord at frame_no t+1

                    Init velocity with (0,0,0)?

  Joint Acceleration -> Computed in the same way instead of joint position use v_x                                  
  Joint Angle -> We take 3 related joints then compute
                 two vector and compute cosine angle between
                 them?

                Challenge is to note down which 3 joint to take.
                WORK required.

  NOTE: JOINT VELOCITY and JOINT ACCELERATION computation method might be incorrect.


---------------------------------  

# Visualization

Visualizaiton helps identify errors in the computation of the biomechanical features.

Joint Position -> Already Visualizable
Joint Angles -> ?? Not needed maybe because there should be much error here,
               but displaying the angle on screen as the pose changes might be
               a good idea? like highlighting the common joint in the joint angle and
              displaying a legend with Joint_NAME and Corresponding angle in each frame?
Joint Velocity -> Showing a curve from previous position to new position? or maybe displaying
                  a vector with appropriate magnitude might provide better visuals?
Joint Acceleration -> No need to visualize if velocity is computed properly


Visualization maybe done later along with the web app front.

--------------------------------


