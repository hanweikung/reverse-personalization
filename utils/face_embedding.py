import cv2
import numpy as np
import torch
from diffusers.utils import load_image
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
        concat_emb_inv (torch.Tensor): The embedding of the largest detected face for performing inversion
        concat_emb (torch.Tensor): The embedding of the largest detected face
        """

        # Load image
        image = load_image(image_path)
        ref_images_embeds = []

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.app.get(image)

        if len(faces) == 0:
            raise ValueError(f"No face detected in {image_path}.")
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
        concat_emb_inv = torch.cat(
            [neg_ref_images_embeds, neg_ref_images_embeds]
        ).to(dtype=dtype, device=device)
        concat_emb = torch.cat(
            [neg_ref_images_embeds, ref_images_embeds]
        ).to(dtype=dtype, device=device)

        return concat_emb_inv, concat_emb
