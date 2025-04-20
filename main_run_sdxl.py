import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from custom_ip_adapter_loader import load_ip_adapter, set_ip_adapter_scale
from sdxl.diffusers.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline as LEditsPPPipelineStableDiffusionXL,
)
from sdxl.leditspp.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline as StableDiffusionPipelineXL_LEDITS,
)
from sdxl.leditspp.scheduling_dpmsolver_multistep_inject import (
    DPMSolverMultistepSchedulerInject,
)
from utils.face_embedding import FaceEmbeddingExtractor


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
        "--face_images",
        type=str,
        nargs="*",
        default=None,
        help="One or more image paths for face embeddings. Defaults to input_image if not provided.",
    )
    parser.add_argument(
        "--skip",
        type=float,
        default=0.1,
        help="Controlling the adherence to the input image.",
    )
    parser.add_argument(
        "--id_emb_scale",
        type=float,
        default=1.0,
        help="Scale for the identity embedding. The default value is 1.0.",
    )
    parser.add_argument(
        "--shift_neg_id_emb_scale",
        action="store_true",
        help="Negative id_emb_scale value will be shifted to positive in the output filename when the flag is True.",
    )
    parser.add_argument(
        "--shift_neg_cfg",
        type=float,
        default=0.0,
        help="Add this value to the classifier-free guidance to make negative values positive in the output filename.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="CFG scale for guidance. The default value is 3.0.",
    )
    parser.add_argument(
        "--inversion",
        default="leditspp",
        choices=["leditspp", "diffusers"],
        type=str,
        help="The inversion mode. The default value is 'leditspp'.",
    )
    parser.add_argument(
        "--num_inversion_steps",
        default=100,
        type=int,
        help="The number of inversion steps",
    )
    parser.add_argument(
        "--resolution",
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
        default=1024,
        help="The size for face detection model input",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible inference."
    )
    parser.add_argument(
        "--max_angle",
        type=float,
        default=0.0,
        help="The maximum allowed angle (in degrees) between the generated face embedding and the input face embedding.",
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

    image = Image.open(args.input_image)
    image = image.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)

    if args.face_images is None:
        args.face_images = [args.input_image]

    pipeline_class = (
        StableDiffusionPipelineXL_LEDITS
        if args.inversion == "leditspp"
        else LEditsPPPipelineStableDiffusionXL
    )
    pipe = pipeline_class.from_pretrained(
        args.sd_model_path,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    if args.inversion == "leditspp":
        pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
            args.sd_model_path,
            subfolder="scheduler",
            algorithm_type="sde-dpmsolver++",
            solver_order=2,
        )
    pipe = pipe.to("cuda")

    # Initialize FaceEmbeddingExtractor instance
    extractor = FaceEmbeddingExtractor(
        ctx_id=0,
        det_thresh=args.det_thresh,
        det_size=(args.det_size, args.det_size),
        model_path=args.insightface_model_path,
    )  # Use GPU (ctx_id=0), or CPU with ctx_id=-1

    id_embs_inv_list = []
    id_embs_list = []
    for face_image in args.face_images:
        id_embs_inv, id_embs = extractor.get_face_embeddings(
            image_path=face_image,
            max_angle=args.max_angle,
            is_opposite=False,
            seed=args.seed,
            scale_factor=args.id_emb_scale,
            dtype=dtype,
            device=device,
        )
        id_embs_inv_list.append(id_embs_inv)
        id_embs_list.append(id_embs)

    weight_names = ["ip-adapter-faceid_sdxl.bin"] * len(args.face_images)
    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name=weight_names,
        image_encoder_folder=None,
    )

    ip_adapter_scales = [args.ip_adapter_scale] * len(args.face_images)
    pipe.set_ip_adapter_scale(ip_adapter_scales)

    generator = torch.Generator(device="cpu").manual_seed(42)

    reconstructed_image = pipe.invert(
        image=image,
        num_inversion_steps=args.num_inversion_steps,
        skip=args.skip,
        source_guidance_scale=args.guidance_scale,
        ip_adapter_image_embeds=id_embs_inv_list,
        generator=generator,
    ).vae_reconstruction_images[0]

    image = pipe(
        prompt="",
        ip_adapter_image_embeds=id_embs_list,
        num_images_per_prompt=1,
        generator=generator,
        guidance_scale=args.guidance_scale,
        timesteps=pipe.scheduler.timesteps,
        latents=pipe.init_latents,
        num_inference_steps=args.num_inversion_steps,
    ).images[0]

    id_emb_scale = (
        args.id_emb_scale + 1.0 if args.shift_neg_id_emb_scale else args.id_emb_scale
    )
    shifted_cfg_scale_tar = args.guidance_scale + args.shift_neg_cfg
    input_filename_wo_ext = Path(args.input_image).stem
    # Replace dots with underscores
    filename_wo_ext = (
        f"{input_filename_wo_ext}-skip-{args.skip}-id-{id_emb_scale}-cfg-{shifted_cfg_scale_tar:05.2f}-ip-{args.ip_adapter_scale:04.2f}"
    ).replace(".", "_")

    image.save(filename_wo_ext + ".png")
