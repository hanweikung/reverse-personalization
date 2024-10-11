import cv2
import insightface
import numpy as np
import torch
from diffusers.utils import load_image
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis


class FaceEmbeddingExtractor:
    """
    A class to extract identity embeddings of the largest face from an image using InsightFace.
    """

    def __init__(self, ctx_id=0, det_thresh=0.5, det_size=(640, 640)):
        """
        Initializes the InsightFace app for facial analysis.

        Args:
        ctx_id (int): GPU context ID (use -1 for CPU).
        det_size (tuple): The size for face detection model input.
        """

        # Initialize InsightFace app
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)

        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
        )

        # Initialize the antelopev2 model using the InsightFace model_zoo API
        snapshot_download("DIAMONIK7777/antelopev2", local_dir="models/antelopev2")
        self.antelopev2_model = insightface.model_zoo.get_model(
            "models/antelopev2/glintr100.onnx"
        )
        self.antelopev2_model.prepare(ctx_id=0)  # ctx_id=0 for GPU, -1 for CPU

    def get_face_embeddings(
        self, image_path, scale_factor=1.0, dtype=torch.float32, device="cuda"
    ):
        """
        Extract the identity embeddings of the largest face from an image file, scale it, and concatenate it with a black image representing a negative embedding.

        Args:
        image_path (str): The file path of the image.
        scale_factor (float): A factor to scale the embedding.
        dtype (torch.dtype): The desired data type of the resulting embedding.
        device (str or torch.device): The device on which to place the resulting embedding.

        Returns:
        concatenated_embedding (torch.Tensor): The embedding of the largest detected face, or None if no face is detected.
        """

        # Load image
        image = load_image(image_path)
        ref_images_embeds = []

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.app.get(image)

        if len(faces) == 0:
            # raise ValueError(f"No face detected in {image_path}.")

            # return None  # No faces detected

            self.face_helper.clean_all()

            self.face_helper.read_image(cv2.imread(image_path))

            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=True)

            # align and warp each face
            self.face_helper.align_warp_face()

            # Raise an error if no faces are detected
            if len(self.face_helper.cropped_faces) == 0:
                raise ValueError(f"No face detected in {image_path}.")

            cropped_face = self.face_helper.cropped_faces[0]
            normed_embedding = self.normalize_embedding(
                self.antelopev2_model.get_feat(cropped_face)[0]
            )
        else:
            # Find the largest face by bounding box area
            largest_face = max(
                faces,
                key=lambda face: (face.bbox[2] - face.bbox[0])
                * (face.bbox[3] - face.bbox[1]),
            )
            normed_embedding = largest_face.normed_embedding

        # Get the embedding of the largest face
        embedding = torch.from_numpy(normed_embedding)

        # Scale the embedding by the provided scale factor
        scaled_embedding = embedding * scale_factor
        ref_images_embeds.append(scaled_embedding.unsqueeze(0))
        ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
        neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
        concatenated_embedding = torch.cat(
            [neg_ref_images_embeds, ref_images_embeds]
        ).to(dtype=dtype, device=device)

        return concatenated_embedding

    def normalize_embedding(self, embedding):
        """
        Normalize the face embedding using L2 normalization.
        Raises an error if the norm is zero.

        Args:
        embedding (np.ndarray): The face embedding to be normalized.

        Returns:
        np.ndarray: The normalized embedding.

        Raises:
        ValueError: If the L2 norm of the embedding is zero.
        """
        # Calculate the L2 norm of the embedding
        l2_norm = np.linalg.norm(embedding)

        # Raise an error if the norm is zero
        if l2_norm == 0:
            raise ValueError("The L2 norm of the embedding is zero, cannot normalize.")

        # Normalize the embedding
        normalized_embedding = embedding / l2_norm

        return normalized_embedding
