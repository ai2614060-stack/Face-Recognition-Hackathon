# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import av
# import cv2
# import numpy as np

# from ultralytics import YOLO
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------
# # Load Models
# # -----------------------------

# yolo_model = YOLO("yolov8n.pt")

# face_app = FaceAnalysis()
# face_app.prepare(ctx_id=0)

# # -----------------------------
# # Memory
# # -----------------------------

# known_faces = []
# known_ids = []
# next_id = 0

# SIMILARITY_THRESHOLD = 0.6

# # -----------------------------
# # Video Processor
# # -----------------------------

# class FaceProcessor(VideoProcessorBase):

#     def recv(self, frame):

#         global next_id

#         img = frame.to_ndarray(format="bgr24")

#         # YOLO detection
#         results = yolo_model(img)

#         boxes = results[0].boxes.xyxy.cpu().numpy()

#         for box in boxes:

#             x1, y1, x2, y2 = map(int, box)

#             face_crop = img[y1:y2, x1:x2]

#             if face_crop.size == 0:
#                 continue

#             faces = face_app.get(face_crop)

#             if len(faces) == 0:
#                 continue

#             embedding = faces[0].embedding.reshape(1, -1)

#             matched_id = None

#             if len(known_faces) > 0:

#                 similarities = cosine_similarity(
#                     embedding,
#                     np.array(known_faces)
#                 )[0]

#                 best_match = np.argmax(similarities)

#                 if similarities[best_match] > SIMILARITY_THRESHOLD:
#                     matched_id = known_ids[best_match]

#             # New identity
#             if matched_id is None:

#                 matched_id = next_id

#                 known_faces.append(embedding[0])
#                 known_ids.append(next_id)

#                 next_id += 1

#             # Draw
#             cv2.rectangle(
#                 img,
#                 (x1, y1),
#                 (x2, y2),
#                 (0, 255, 0),
#                 2
#             )

#             cv2.putText(
#                 img,
#                 f"ID: {matched_id}",
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0, 255, 0),
#                 2
#             )

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # -----------------------------
# # Streamlit UI
# # -----------------------------

# st.title("Realtime Face Recognition")

# webrtc_streamer(
#     key="face-recognition",
#     video_processor_factory=FaceProcessor,
#     media_stream_constraints={
#         "video": True,
#         "audio": False
#     },
#     async_processing=True
# )



import streamlit as st
import cv2

st.title("Face Detection")

camera = st.camera_input("Take a picture")

if camera:

    file_bytes = camera.getvalue()

    import numpy as np

    npimg = np.frombuffer(file_bytes, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        4
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

    st.image(img, channels="BGR")
