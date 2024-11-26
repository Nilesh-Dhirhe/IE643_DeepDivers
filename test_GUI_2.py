flag_model_loaded = False

if not flag_model_loaded:
    from PIL import Image 
    from streamlit_drawable_canvas import st_canvas
    from tqdm import tqdm
    from ultralytics import YOLO
    import numpy as np
    import streamlit as st
    import cv2
    import tempfile
    import os
    import matplotlib.pyplot as plt
    import math
    import tempfile
    from io import BytesIO
    import supervision as sv
    print("supervision.__version__:", sv.__version__)

# Set up the Streamlit app title
st.title("Speed & Vehicle Detection")

# FREEEZE Functions & FLAGs that allow certain pieces of code to run only once
# Initialize session states to keep track of which widgets are frozen
if "freeze_model_input" not in st.session_state:
    st.session_state.freeze_model_input = False
if "freeze_file_input" not in st.session_state:
    st.session_state.freeze_file_input = False
if "freeze_output_file_input" not in st.session_state:
    st.session_state.freeze_output_file_input = False
if "freeze_dist_x_input" not in st.session_state:
    st.session_state.freeze_dist_x_input = False
if "freeze_dist_y_input" not in st.session_state:
    st.session_state.freeze_dist_y_input = False

# Functions to freeze widget
def freeze_model_input():
    st.session_state.freeze_model_input = True
def freeze_file_input():
    st.session_state.freeze_file_input = True
def freeze_output_file_input():
    st.session_state.freeze_output_file_input = True
def freeze_dist_x_input():
    st.session_state.freeze_dist_x_input = True
def freeze_dist_y_input():
    st.session_state.freeze_dist_y_input = True

model = None
COCO_classes = ['background',
                'person',
                'bicycle',
                'car',
                'motorbike',
                'aeroplane',
                'bus',
                'train',
                'truck']
st.subheader("Upload pre-trained model")
# File picker for model
uploaded_model = st.file_uploader("Upload YOLO model file", type=["pt"], key = 0, disabled = st.session_state.freeze_model_input)

st.subheader("Upload video")
# File picker for video upload
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], key = 1, disabled = st.session_state.freeze_file_input)
temp_video_path = None
# Check if a video is uploaded
if uploaded_video is not None:
    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name
    st.write(temp_video_path)
    video_file = open(temp_video_path, "rb")
    video_bytes = video_file.read()
    st.text("Playing uploaded video...")
    st.video(video_bytes)

    max_display_height = 400 # default
    max_display_width = 600 # default
    display_height = 0
    display_width = 0

    # load YOLO model
    if uploaded_model is not None and not flag_model_loaded:
         # Save the file temporarily and pass its path to YOLO()
        with open("uploaded_model.pt", "wb") as f:
            f.write(uploaded_model.getbuffer())

        # Initialize YOLO with the saved model path
        model = YOLO("uploaded_model.pt")
        model.fuse()
        # Verify model loading (example prediction or summary)
        st.write("Model loaded successfully.")
        flag_model_loaded = True
    elif uploaded_model is None and not flag_model_loaded:
        MODEL = r"E:\IITB\IE 643\Trained Models\results\yolov8m.pt"
        model = YOLO(MODEL)
        model.fuse()
        flag_model_loaded = True
        
    freeze_model_input()
    freeze_file_input()
    
    # Create a text input box
    output_path = st.text_input("Path with file name to save output video", value=None, disabled = st.session_state.freeze_output_file_input)
    if output_path is not None:
        freeze_output_file_input()
        st.write(output_path)
        # SOURCE_VIDEO_PATH = file_entry.get()
        # dist_x = int(perpendicular_entry.get())
        # dist_y = int(along_entry.get())

        # Predict and annotate single frame
        # dict maping class_id to class_name
        CLASS_NAMES_DICT = model.model.names
        # class_ids of interest - car, motorcycle, bus and truck
        selected_classes = [2, 3, 5, 7]
        # create frame generator
        generator = sv.get_video_frames_generator(temp_video_path)
        # create instance of BoxAnnotator
        box_annotator = sv.BoxCornerAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

        # acquire first video frame
        iterator = iter(generator)
        frame = next(iterator)

        # Display the first frame from video
        st.subheader("Known Region")
        # Create a canvas with the uploaded image
        height = frame.shape[0]
        width = frame.shape[1]
        print('shape', width, height)

        # choosing appropriate display width and height
        if (height/width > max_display_height/max_display_width):
            # height is the limiting dimension
            display_height = max_display_height
            display_width = width * max_display_height/height # scaling display width down
        else:
            # width is the limiting dimension
            display_width = max_display_width
            display_height = height * max_display_width/width # scaling display width down
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Color for the fill
            stroke_width=3,
            stroke_color="red",
            background_image=Image.fromarray(frame),  # Set the uploaded image as background
            update_streamlit=True,
            height = display_height,
            width = display_width,
            drawing_mode="point",
            point_display_radius=5,
            key="canvas",
        )

        clicks = []
        # Check for points (clicks)
        if canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "circle":  # Points are stored as circles
                    # print(obj)
                    x, y = (obj["left"]+obj["width"]/2), obj["top"] # the origin of x is left, origin of y is center
                    # does it mean the coordinates are of the left-center of the square formed by the point?? assumed to be so here.
                    scaled_x, scaled_y = x / display_width * width, y / display_height * height
                    clicks.append((scaled_x, scaled_y))

            # Display the coordinates of clicks
            if len(clicks)==4:
                st.write(f"{len(clicks)} clicks recorded")
                # print([click[0] for click in clicks])
                # print([click[1] for click in clicks])
            else:
                st.warning(f"{len(clicks)} clicks recorded" if len(clicks)>0 else "No clicks recorded")

        # Create a text input box
        dist_x = st.text_input("Distance perpendicular to the road: (metres)", value=None, disabled = st.session_state.freeze_dist_x_input)
        # Create a text input box
        dist_y = st.text_input("Distance along the road: (metres)", value=None, disabled = st.session_state.freeze_dist_y_input)

        # model prediction on single frame and conversion to supervision Detections
        results = model(frame, verbose=False)[0]

        # convert to Detections
        detections = sv.Detections.from_ultralytics(results)
        # only consider class id from selected_classes define above
        detections = detections[np.isin(detections.class_id, selected_classes)]

        # annotate and display frame
        anotated_frame=box_annotator.annotate(scene=frame, detections=detections)
        anotated_frame_2=label_annotator.annotate(scene=anotated_frame, detections=detections)

        if len(clicks)>=4 and dist_x is not None and dist_y is not None:
            freeze_dist_x_input()
            freeze_dist_y_input()
            # dist_x = float(input("Distance perpendicular to the road "))
            # dist_y = float(input("Distance along the road "))
            dist_x = int(dist_x)
            dist_y = int(dist_y)
            st.write("Input distances:",dist_x,dist_y)
            cap = cv2.VideoCapture(temp_video_path) # cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            selected_points = clicks
            destination_points = np.array([
                [0,0],
                [dist_x*100,0],
                [dist_x*100,dist_y*100],
                [0,dist_y*100]]
            )
            selected_points = np.float32(selected_points)
            destination_points = np.float32(destination_points)
            # assuming a metre in dist_x or dist_y would result in 100 pixels for practical usage without loss of information affter transform

            transform_matrix = cv2.getPerspectiveTransform(selected_points, destination_points)
            height = frame.shape[0]
            width = frame.shape[1]
            # print(width,height)

            # Perform the perspective transformation
            transformed_image = cv2.warpPerspective(anotated_frame_2, transform_matrix, (int(dist_x*100),int(dist_y*100)))

            # Display the transformed image using Matplotlib
            # plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
            # plt.axis('off')  # Hide the axes
            # plt.show(block=False)  # Show the image without blocking
            # plt.pause(8)  # Display for 8 seconds
            # plt.close()  # Close the image window
            print("reached1")
            tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=15, minimum_matching_threshold=0.4, frame_rate=25)
            bounding_box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            video_info = sv.VideoInfo.from_video_path(video_path=temp_video_path)
            smoother = sv.DetectionsSmoother()
            print("reached2")
            # Dictionary to store the tracking path (coordinates) of each object, transformed tracking path (coordinates) and speed
            object_paths = {}
            transformed_object_paths = {}
            speeds = {}
            window = 0.5 # the number seconds, over which average speed is calculated
            window_frames = window * video_info.fps# averaging window in frames
            vehicle_images = {}
            print("reached3")
            # Get the total frame count of the video
            video_capture = cv2.VideoCapture(temp_video_path)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()
            print("reached5")
            # the function returns the 0th index value of a tuple, set of which exist in a last, 
            # if the timestamp (index 1 element is timestamp) is equal to the specified time stamp 
            # or is the closest imestamp lesser than the specified timestamp
            def find_latest_value(my_dict, specific_timestamp):
                latest_value = None
                # print(my_dict)
                # Check for value at a specific timestamp
                for key, (value, timestamp) in my_dict.items():
                    if timestamp == specific_timestamp:
                        latest_value = value
                        break
                
                # If no value found at specific timestamp, check for the latest value before specific_timestamp
                if latest_value is None:
                    # Find the latest value that is less than specific_value
                    latest_item = max((v[0] for v in my_dict.values() if v[1] < specific_timestamp), 
                                    default=0)  # Max based on timestamp
                return latest_value
            print("reached")

            st.subheader("Annotating Video")
            # Initialize progress bar
            progress_bar = st.progress(0)
            # Use tqdm to update the progress bar
            with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
                # Define callback function with progress tracking
                def callback_with_progress(frame: np.ndarray, index: int) -> np.ndarray:
                                
                        def callback(frame: np.ndarray, index: int) -> np.ndarray:
                            results = model(frame)[0]
                            detections = sv.Detections.from_ultralytics(results)
                            detections = tracker.update_with_detections(detections)
                            detections = smoother.update_with_detections(detections)

                            labels = []
                            for i, tracker_id in enumerate(detections.tracker_id):
                                latest_speed = find_latest_value(speeds.get(tracker_id, {}), int(int(index/(video_info.fps/2))*video_info.fps/2))
                                # Draw bounding boxes and labels; update speed every half second
                                # print(latest_speed)
                                if not latest_speed == None: 
                                    latest_speed = math.ceil(latest_speed*3.6)
                                labels.append(f"#{tracker_id} {CLASS_NAMES_DICT[detections.class_id[i]]} {latest_speed} km/h" )

                            annotated_frame = bounding_box_annotator.annotate(
                                scene=frame.copy(), detections=detections)
                            annotated_frame = label_annotator.annotate(
                                scene=annotated_frame, detections=detections, labels=labels)

                            # Update paths for each detected object and draw trailing lines
                            for i, tracker_id in enumerate(detections.tracker_id):
                                # Initialize a path list for a new tracker ID
                                x1, y1, x2, y2 = detections.xyxy[i]
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                if tracker_id not in object_paths:
                                    object_paths[tracker_id] = []
                                    transformed_object_paths[tracker_id] = {} # dictionary with frame number as key
                                    speeds[tracker_id] = {} # dictionary with frame number as key
                                    # print(y1,y2,x1,x2)
                                    vehicle_images[tracker_id] = frame[round(y1):round(y2),round(x1):round(x2)]
                                
                                # Get the center of the current bounding box


                                # Append the current center to the object's path
                                object_paths[tracker_id].append((int(x2), int(y2))) # using x2,y2 as the coordinates above the vehicle get projected unevenly with distance from the camera
                                transformed_object_paths[tracker_id][index] = cv2.perspectiveTransform(np.array([np.array([[(x2),(y2)]], dtype=np.float32)]), transform_matrix)
                                # transformed_object_paths[tracker_id][index] = (cv2.perspectiveTransform(np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2), transform_matrix)[0])
                                # print(transformed_object_paths[tracker_id][index])  
                                speeds[tracker_id][index] = (0,index)
                                # Check for valid index values starting from index - window_frames
                                for offset in range(int(window_frames)): # to ensure a coordinate exists at speeds[tracker_id][index-window_frames]
                                    check_index = index - window_frames + offset
                                    if check_index in transformed_object_paths[tracker_id]:
                                        speeds[tracker_id][index] = (np.linalg.norm(transformed_object_paths[tracker_id][index] - transformed_object_paths[tracker_id][check_index]) * video_info.fps/(index-check_index)/100,
                                        index) # dictionary key and tuple element on index 1 are the same, bad naming, future fix
                                        
                                        # print(speeds[tracker_id][index])
                                        # result in speeds[tracker_id] is stored in m/s
                                        break

                                # Draw trailing line by connecting points in the path
                                # limited to last 30 points
                                for j in range(1, len(object_paths[tracker_id])):
                                    # print(annotated_frame.shape)
                                    cv2.line(
                                        annotated_frame,
                                        object_paths[tracker_id][j - 1],  # Previous point
                                        object_paths[tracker_id][j],      # Current point
                                        (0, 255, 0),                      # Line color (green)
                                        2                                 # Line thickness
                                    )
                                
                                # Optional: Limit the trail length for each object
                                if len(object_paths[tracker_id]) > 30:  # Keep the last 30 positions
                                    object_paths[tracker_id].pop(0)
                            
                            # cv2.imshow('Annotations', annotated_frame)
                            # if cv2.waitKey(500) & 0xFF == ord('q'):  # Press 'q' to quit early
                            #     pass
                            # Update tqdm bar
                            pbar.update(1) # redundant old way of progress showing
                            progress_bar.progress(index/total_frames)
                            return annotated_frame
                        # Call the wrapped callback function
                        return callback(frame, index)
                    
                sv.process_video(
                    source_path=temp_video_path,
                    target_path=output_path,
                    callback=callback_with_progress
                )

            # After processing the video, print the paths of unique objects
            print("Paths of unique objects detected:")
            for tracker_id, path in object_paths.items():
                print(f"Object {tracker_id} path:", path)


            # # List of points to transform
            # object_paths = object_paths.reshape(-1, 1, 2)  # Reshape to (N, 1, 2) as required by perspectiveTransform

            # # Apply the transformation to the points
            # transformed_points = cv2.perspectiveTransform(object_paths, transform_matrix)

            # Assuming 'vehicles' and 'speeds' dictionaries are defined
            # Example structure: vehicles = {tracker_id: [image1, image2, ...]}, speeds = {tracker_id: [speed1, speed2, ...]}

            # Define the number of rows
            num_rows = len(vehicle_images)

            # Create a figure with a 2-column layout
            fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))

            # Flatten the axes array if there is more than one row
            if num_rows > 1:
                axes = axes.reshape(num_rows, 2)

            # Loop over each tracker_id and plot the image and speeds
            for idx, tracker_id in enumerate(vehicle_images.keys()):
                
                # fill missing indices
                for index in range(total_frames):
                    if index not in speeds[tracker_id]:
                        speeds[tracker_id][index] = (0,index)
                
                if tracker_id not in speeds:
                    print(f"Tracker ID {tracker_id} is missing in speeds dictionary")
                    continue

                # Retrieve the first image and speed data up to index 760
                vehicle_image = vehicle_images[tracker_id]  # First image for the tracker_id
                speed_values = [speeds[tracker_id][key][0] for key in range(760)]

                # Display the image in the first column, images for different tracking IDs
                ax_image = axes[idx, 0] if num_rows > 1 else axes[0]
                ax_image.imshow(vehicle_image)
                ax_image.axis('off')
                ax_image.set_title(f"Vehicle Image: {tracker_id})")

                # Plot the speed values in the second column, speed v/s frame index plots for different tracking IDs
                ax_speed = axes[idx, 1] if num_rows > 1 else axes[1]
                ax_speed.plot(speed_values, color="blue")
                ax_speed.set_title(f"Speeds: {tracker_id}")
                ax_speed.set_xlabel("Index")
                ax_speed.set_ylabel("Speed")
            # Display fig using pyplot widget in Streamlit
            st.pyplot(fig)

            # Header
            st.subheader("Output Video")
            # Subheader
            st.text(f"Success | Video processed and saved as {output_path}")
            st.text("Playing output video...")
            video_file_output = open(output_path, "rb")
            video_bytes_output = video_file_output.read()
            st.video(video_bytes_output)

            video_file_output.close()
            del model
            os.remove(temp_video_path)
            os.remove('uploaded_model.pt')
    else:
        st.warning("Please set an output video path")
else:
    st.warning("Please upload a video file to play it back")