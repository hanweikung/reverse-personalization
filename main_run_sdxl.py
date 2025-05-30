import argparse
from pathlib import Path

import face_alignment
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from sdxl.leditspp.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline as StableDiffusionPipelineXL_LEDITS,
)
from sdxl.leditspp.scheduling_dpmsolver_multistep_inject import (
    DPMSolverMultistepSchedulerInject,
)
from utils.extractor import extract_faces
from utils.face_embedding import FaceEmbeddingExtractor
from utils.merger import paste_foreground_onto_background


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demonstrate how to use LEDITS++ with SDXL and the IP adapter"
    )
    parser.add_argument(
        "--sd_model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to the Stable Diffusion XL model",
    )
    parser.add_argument(
        "--insightface_model_path",
        type=str,
        default="~/.insightface",
        help="Path to the InsightFace model",
    )
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="The input image that will be edited",
    )
    parser.add_argument(
        "--skip",
        type=float,
        default=0.7,
        help="Controlling the adherence to the input image.",
    )
    parser.add_argument(
        "--id_emb_scale",
        type=float,
        default=1.0,
        help="Scale for the identity embedding. The default value is 1.0.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=-7.5,
        help="CFG scale for guidance.",
    )
    parser.add_argument(
        "--num_inversion_steps",
        default=100,
        type=int,
        help="The number of inversion steps",
    )
    parser.add_argument(
        "--face_image_size",
        type=int,
        default=1024,
        help=("The resolution for output images."),
    )
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.1,
        help="Set your desired threshold for face detection.",
    )
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default="1.0",
        help=(
            "Controls the amount of text or image conditioning to apply to the model."
            "A value of 1.0 means the model is only conditioned on the image prompt."
        ),
    )
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="The size for face detection model input",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible inference."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dtype = torch.float16
    device = f"cuda:{args.device_num}"

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype
    )

    anon_image = image = Image.open(args.input_image)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )
    face_images, image_to_face_matrices = extract_faces(fa, image, args.face_image_size)

    pipe = StableDiffusionPipelineXL_LEDITS.from_pretrained(
        args.sd_model_path,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        args.sd_model_path,
        subfolder="scheduler",
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
    )
    pipe = pipe.to("cuda")

    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin",
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    # Initialize FaceEmbeddingExtractor instance
    extractor = FaceEmbeddingExtractor(
        ctx_id=0,
        det_thresh=args.det_thresh,
        det_size=(args.det_size, args.det_size),
        model_path=args.insightface_model_path,
    )  # Use GPU (ctx_id=0), or CPU with ctx_id=-1

    for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
        id_embs_inv, id_embs = extractor.get_face_embeddings(
            image_path=face_image,
            seed=args.seed,
            scale_factor=args.id_emb_scale,
            dtype=dtype,
            device=device,
        )

        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        reconstructed_image = pipe.invert(
            image=face_image,
            num_inversion_steps=args.num_inversion_steps,
            skip=args.skip,
            source_guidance_scale=args.guidance_scale,
            ip_adapter_image_embeds=[id_embs_inv],
            generator=generator,
        ).vae_reconstruction_images[0]

        anon_face_image = pipe(
            prompt="",
            ip_adapter_image_embeds=[id_embs],
            num_images_per_prompt=1,
            generator=generator,
            guidance_scale=args.guidance_scale,
            timesteps=pipe.scheduler.timesteps,
            latents=pipe.init_latents,
            num_inference_steps=args.num_inversion_steps,
        ).images[0]

        anon_image = paste_foreground_onto_background(
            anon_face_image, anon_image, image_to_face_mat
        )

    input_filename_wo_ext = Path(args.input_image).stem
    # Replace dots with underscores
    filename_wo_ext = (
        f"{input_filename_wo_ext}-skip-{args.skip}-id-{args.id_emb_scale}-cfg-{args.guidance_scale:05.2f}-ip-{args.ip_adapter_scale:04.2f}"
    ).replace(".", "_")

    anon_image.save(filename_wo_ext + ".png")
