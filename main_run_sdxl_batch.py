import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from accelerate import Accelerator
from datasets import load_dataset
from diffusers.utils import make_image_grid
from tqdm import tqdm
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
    parser.add_argument(
        "--dataset_loading_script_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset loading script file.",
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
        help="Scaling factor for the identity embedding, with a default value of -1.0 for anonymization purposes.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="CFG scale for guidance. The default value is 2.5.",
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
        help="The number of inversion steps.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for output images.",
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
        "--seed", type=int, default=None, help="A seed for reproducible inference."
    )
    parser.add_argument(
        "--vis_input",
        action="store_true",
        help="If set, save the input and generated images together as a single output image for easy visualization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test-infer/",
        help="The output directory where generated images are saved.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="log.txt",
        help="The output text file records the images in which faces could not be detected.",
    )
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.1,
        help="Set your desired threshold for face detection.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Truncate the number of test examples to this value if set.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="The size for face detection model input",
    )
    parser.add_argument(
        "--max_angle",
        type=float,
        default=0.0,
        help="The maximum allowed angle (in degrees) between the generated face embedding and the input face embedding.",
    )
    args = parser.parse_args()
    return args


def make_test_dataset(args):
    ds = load_dataset(
        path=args.dataset_loading_script_path, split="test", trust_remote_code=True
    )

    # Preprocessing the datasets.
    image_transforms = transforms_v2.Compose(
        [
            transforms_v2.Resize(
                args.resolution, interpolation=transforms_v2.InterpolationMode.BILINEAR
            ),
            transforms_v2.CenterCrop(args.resolution)
            if args.center_crop
            else transforms_v2.RandomCrop(args.resolution),
        ]
    )

    def preprocess_test(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        images = [image_transforms(image) for image in images]
        examples["image"] = images
        return examples

    if args.max_test_samples is not None:
        max_test_samples = min(args.max_test_samples, len(ds))
        ds = ds.select(range(max_test_samples))

    test_dataset = ds.with_transform(preprocess_test)
    return test_dataset


def collate_fn(examples):
    images = [example["image"] for example in examples]
    image_paths = [example["image_path"] for example in examples]

    return {
        "images": images,
        "image_paths": image_paths,
    }


def preprocess_image(pil_image, device):
    """
    Preprocess a PIL image for PyTorch model input.

    Args:
    pil_image (PIL.Image): The input image in PIL format.
    device (torch.device): The device to which the tensor should be moved (e.g., 'cpu' or 'cuda').

    Returns:
    torch.Tensor: The preprocessed image tensor, ready for model input (shape: (1, 3, H, W)).
    """
    # Convert the image to RGB and then to a NumPy array
    image = np.array(pil_image.convert("RGB"))[:, :, :3]

    # Convert the NumPy array to a PyTorch tensor and normalize to [-1, 1]
    image = torch.from_numpy(image).float() / 127.5 - 1

    # Permute dimensions to (C, H, W) and add a batch dimension
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)

    return image


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()

    # Specify the device
    device = accelerator.device

    os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_input:
        output_vis_dir = Path(args.output_dir, "vis")
        output_vis_dir.mkdir(parents=True, exist_ok=True)

    generator = None
    dtype = torch.float16

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype
    )

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

    # Load the test dataset
    test_dataset = make_test_dataset(args)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    test_dataloader = accelerator.prepare(test_dataloader)

    # Open the output file in write mode
    with open(args.log_file, "w") as f:
        for step, batch in enumerate(tqdm(test_dataloader)):
            # Group corresponding items from each key together
            grouped_items = list(
                zip(
                    batch["images"],
                    batch["image_paths"],
                )
            )

            for (
                image,
                image_path,
            ) in grouped_items:
                filename = f"{Path(image_path).stem}.png"
                save_to = Path(args.output_dir, filename)

                if save_to.is_file():
                    continue

                try:
                    id_embs_inv, id_embs = extractor.get_face_embeddings(
                        image_path=image_path,
                        max_angle=args.max_angle,
                        is_opposite=False,
                        seed=args.seed,
                        scale_factor=args.id_emb_scale,
                        dtype=dtype,
                        device=device,
                    )
                except ValueError as e:
                    f.write(f"{e}\n")
                else:
                    if args.seed is not None:
                        # create a generator for reproducibility; notice you don't place it on the GPU!
                        generator = torch.Generator(device="cpu").manual_seed(args.seed)

                    reconstructed_image = pipe.invert(
                        image=image,
                        num_inversion_steps=args.num_inversion_steps,
                        skip=args.skip,
                        source_guidance_scale=args.guidance_scale,
                        ip_adapter_image_embeds=[id_embs_inv],
                        generator=generator,
                    ).vae_reconstruction_images[0]

                    pil_image = pipe(
                        prompt="",
                        ip_adapter_image_embeds=[id_embs],
                        num_images_per_prompt=1,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        timesteps=pipe.scheduler.timesteps,
                        latents=pipe.init_latents,
                        num_inference_steps=args.num_inversion_steps,
                    ).images[0]

                    pil_image.save(save_to)

                    if args.vis_input:
                        save_vis_to = Path(output_vis_dir, filename)
                        if not save_vis_to.is_file():
                            combined_image = make_image_grid(
                                [image, pil_image], rows=1, cols=2
                            )
                            combined_image.save(save_vis_to)
