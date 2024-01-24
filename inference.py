import numpy as np
import onnxruntime as rt
import mediapipe as mp
import cv2
import os
import time
from skimage.transform import SimilarityTransform


class Inference():
    def __init__(self, model_path = "./model/FaceTransformerOctupletLoss.onnx") -> None:
        # Target landmark coordinates for alignment (used in training)
        self.LANDMARKS_TARGET = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        # Initialize Face Detector (For Example Mediapipe)
        self.FACE_DETECTOR = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1
        )

        try:
            # Initialize the Face Recognition Model (FaceTransformerOctupletLoss)
            self.FACE_RECOGNIZER = rt.InferenceSession(model_path, providers=rt.get_available_providers())
        except:
            print('problem to load model')
            exit()



    def inf(self,img_path):
        img = cv2.imread(img_path)

        # ---------------------------------------------------------------------------------------------------------------------
        # FACE DETECTION

        # Process the image with the face detector
        result = self.FACE_DETECTOR.process(img)

        if result.multi_face_landmarks:
            # Select 5 Landmarks (Eye Centers, Nose Tip, Left Mouth Corner, Right Mouth Corner)
            five_landmarks = np.asarray(result.multi_face_landmarks[0].landmark)[[470, 475, 1, 57, 287]]

            # Extract the x and y coordinates of the landmarks of interest
            landmarks = np.asarray(
                [[landmark.x * img.shape[1], landmark.y * img.shape[0]] for landmark in five_landmarks]
            )

            # Extract the x and y coordinates of all landmarks
            all_x_coords = [landmark.x * img.shape[1] for landmark in result.multi_face_landmarks[0].landmark]
            all_y_coords = [landmark.y * img.shape[0] for landmark in result.multi_face_landmarks[0].landmark]

            # Compute the bounding box of the face
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            bbox = [[x_min, y_min], [x_max, y_max]]

        else:
            print("No faces detected")
            return [], False


        # ---------------------------------------------------------------------------------------------------------------------
        # FACE ALIGNMENT

        # Align Image with the 5 Landmarks
        tform = SimilarityTransform()
        tform.estimate(landmarks, self.LANDMARKS_TARGET)
        tmatrix = tform.params[0:2, :]
        img_aligned = cv2.warpAffine(img, tmatrix, (112, 112), borderValue=0.0)

        # ---------------------------------------------------------------------------------------------------------------------
        # FACE RECOGNITION

        # Inference face embeddings with onnxruntime
        input_image = (np.asarray([img_aligned]).astype(np.float32)).clip(0.0, 255.0).transpose(0, 3, 1, 2)
        embedding = self.FACE_RECOGNIZER.run(None, {"input_image": input_image})[0][0]


        return embedding, True