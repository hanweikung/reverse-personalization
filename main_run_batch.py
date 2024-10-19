import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import make_image_grid
from torchvision.transforms.v2 import ToPILImage
from tqdm import tqdm

from ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
from face_embedding_utils import FaceEmbeddingExtractor


def parse_arguments():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Load image files from a JSONL file and anonymize the faces present in those images."
    )
    parser.add_argument(
        "--dataset_loading_script_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset loading script file.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the test dataset will be resized to this resolution.",
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
        "--ip_adapter_scale",
        type=float,
        default="1.0",
        help=(
            "Controls the amount of text or image conditioning to apply to the model."
            "A value of 1.0 means the model is only conditioned on the image prompt."
        ),
    )
    parser.add_argument(
        "--id_emb_scale",
        type=float,
        default=-1.0,
        help="Scaling factor for the identity embedding, with a default value of -1.0 for anonymization purposes.",
    )

    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help=(
            "A higher guidance scale value encourages the model to generate images closely linked to the text"
            "prompt at the expense of lower image quality."
        ),
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=70,
        help="Controlling the adherence to the input image.",
    )
    parser.add_argument("--eta", type=float, default=1)
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
        "--vis_input",
        action="store_true",
        help="If set, save the input and generated images together as a single output image for easy visualization.",
    )
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.1,
        help="Set your desired threshold for face detection.",
    )
    parser.add_argument(
        "--det_size",
        type=int,
        default=512,
        help="The size for face detection model input",
    )

    # Parse the arguments and return them
    return parser.parse_args()


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


def preprocess_image(pil_image, dtype, device):
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
    image = image.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype, device=device)

    return image


def main():
    # Parse the command-line arguments
    args = parse_arguments()

    accelerator = Accelerator()

    # Specify the device
    device = accelerator.device

    os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_input:
        output_vis_dir = Path(args.output_dir, "vis")
        output_vis_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the LDM Stable Diffusion pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "/data/han-wei/models/stable-diffusion-v1-5"  # load local save of model (for internet problems)

    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    ldm_stable.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin",
        image_encoder_folder=None,
    )
    ldm_stable.set_ip_adapter_scale(args.ip_adapter_scale)
    ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)
    dtype = ldm_stable.dtype

    # Initialize the ToPILImage transform
    to_pil = ToPILImage()

    # Initialize FaceEmbeddingExtractor instance
    extractor = FaceEmbeddingExtractor(
        ctx_id=0, det_thresh=args.det_thresh, det_size=(args.det_size, args.det_size)
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

            for image, image_path in grouped_items:
                filename = f"{Path(image_path).stem}.png"
                save_to = Path(args.output_dir, filename)

                if save_to.is_file():
                    continue

                try:
                    id_embs_inv, id_embs = extractor.get_face_embeddings(
                        image_path=image_path,
                        scale_factor=args.id_emb_scale,
                        dtype=dtype,
                        device=device,
                    )
                except ValueError as e:
                    f.write(f"{e}\n")
                else:
                    # Preprocess the image
                    x0 = preprocess_image(pil_image=image, dtype=dtype, device=device)

                    # vae encode image
                    w0 = (
                        ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215
                    ).to(dtype=dtype)

                    wt, zs, wts = inversion_forward_process(
                        ldm_stable,
                        w0,
                        etas=args.eta,
                        prompt="",
                        cfg_scale=args.guidance_scale,
                        prog_bar=True,
                        num_inference_steps=args.num_diffusion_steps,
                        ip_adapter_image_embeds=[id_embs_inv],
                    )

                    w0, _ = inversion_reverse_process(
                        ldm_stable,
                        xT=wts[args.num_diffusion_steps - args.skip],
                        etas=args.eta,
                        prompts=[""],
                        cfg_scales=[args.guidance_scale],
                        prog_bar=True,
                        zs=zs[: (args.num_diffusion_steps - args.skip)],
                        controller=None,
                        ip_adapter_image_embeds=[id_embs],
                    )

                    # vae decode image
                    x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample

                    # The tensor values should be in the range [0, 1] for float tensors or [0, 255] for integer tensors.
                    image_tensor = (x0_dec / 2 + 0.5).clamp(0, 1)[0]

                    # Convert the tensor to a PIL image
                    pil_image = to_pil(image_tensor)
                    pil_image.save(save_to)

                    if args.vis_input:
                        save_vis_to = Path(output_vis_dir, filename)
                        if not save_vis_to.is_file():
                            combined_image = make_image_grid(
                                [image, pil_image], rows=1, cols=2
                            )
                            combined_image.save(save_vis_to)


if __name__ == "__main__":
    main()
