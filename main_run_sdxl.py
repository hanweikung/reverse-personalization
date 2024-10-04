import argparse

import cv2
import numpy as np
import torch
from diffusers.utils import load_image, make_image_grid
from insightface.app import FaceAnalysis
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from custom_ip_adapter_loader import load_ip_adapter, set_ip_adapter_scale
from sdxl.diffusers.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline as LEditsPPPipelineStableDiffusionXL,
)
from sdxl.leditspp.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline as StableDiffusionPipelineXL_LEDITS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demonstrate how to use LEDITS++ with SDXL and the IP adapter"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="The input image that will be edited",
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
        default="ledits++",
        choices=["ledits++", "diffusers"],
        type=str,
        help="The inversion mode. The default value is 'ledits++'.",
    )

    args = parser.parse_args()
    return args


def extract_id_embeddings(image_path, id_emb_scale):
    image = load_image(image_path)
    ref_images_embeds = []
    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    faces = app.get(image)
    image = torch.from_numpy(faces[0].normed_embedding)
    image = image * id_emb_scale
    ref_images_embeds.append(image.unsqueeze(0))
    ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
    neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
    id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(
        dtype=torch.float16, device="cuda"
    )
    return id_embeds


if __name__ == "__main__":
    args = parse_args()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
    )

    image = Image.open(args.input_image)
    # image = Image.open("example_images/yann-lecun.jpg")
    image = image.resize((768, 768), Image.Resampling.LANCZOS)

    pipeline_class = (
        StableDiffusionPipelineXL_LEDITS
        if args.inversion == "ledits++"
        else LEditsPPPipelineStableDiffusionXL
    )
    pipe = pipeline_class.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    id_embeds = extract_id_embeddings(args.input_image, args.id_emb_scale)
    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin",
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(1.0)

    generator = torch.Generator(device="cpu").manual_seed(42)

    reconstructed_image = pipe.invert(
        image=image,
        num_inversion_steps=100,
        skip=args.skip,
        source_guidance_scale=args.guidance_scale,
        ip_adapter_image_embeds=[id_embeds],
        generator=generator,
    ).vae_reconstruction_images[0]

    image = pipe(
        prompt="",
        ip_adapter_image_embeds=[id_embeds],
        num_images_per_prompt=1,
        generator=generator,
        guidance_scale=args.guidance_scale,
        timesteps=pipe.scheduler.timesteps,
        latents=pipe.init_latents,
    ).images[0]

    id_emb_scale = (
        args.id_emb_scale + 1.0 if args.shift_neg_id_emb_scale else args.id_emb_scale
    )
    shifted_cfg_scale_tar = args.guidance_scale + args.shift_neg_cfg

    # Replace dots with underscores
    filename_wo_ext = (
        f"skip-{args.skip}-id-{id_emb_scale}-cfg-{shifted_cfg_scale_tar:05.2f}"
    ).replace(".", "_")

    image.save(filename_wo_ext + ".png")
