import argparse
from pathlib import Path

import torch
from diffusers import (
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from diffusers.utils import load_image
from PIL import Image
from torch import autocast, inference_mode
from tqdm import tqdm

from custom_ip_adapter_loader import load_ip_adapter, set_ip_adapter_scale
from ddm_inversion.ddim_inversion import ddim_inversion
from ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
from ddm_inversion.utils import dataset_from_yaml, image_grid
from utils.face_embedding import FaceEmbeddingExtractor
from prompt_to_prompt.ptp_classes import (
    AttentionRefine,
    AttentionReplace,
    AttentionStore,
    EmptyControl,
    load_512,
)
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_model_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Path to the Stable Diffusion 1.5 model",
    )
    parser.add_argument(
        "--insightface_model_path",
        type=str,
        default="~/.insightface",
        help="Path to the InsightFace model",
    )
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=7.0)
    parser.add_argument("--cfg_tar", type=float, default=7.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--dataset_yaml", default="test.yaml")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument(
        "--mode", default="our_inv", help="modes: our_inv,p2pinv,p2pddim,ddim"
    )
    parser.add_argument("--skip", type=int, default=36)
    parser.add_argument("--xa", type=float, default=0.6)
    parser.add_argument("--sa", type=float, default=0.2)
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
        "--output_file",
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
        "--det_size",
        type=int,
        default=512,
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
    parser.add_argument(
        "--mask_delay_steps",
        type=int,
        default=0,
        help="The number of diffusion steps to wait before applying the mask.",
    )

    args = parser.parse_args()
    full_data = dataset_from_yaml(args.dataset_yaml)

    # create scheduler
    # load diffusion model
    sd_model_path = args.sd_model_path

    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_tar_list = [args.cfg_tar]
    eta = args.eta  # = 1
    skip_zs = [args.skip]
    xa_sa_string = f"_xa_{args.xa}_sa{args.sa}_" if args.mode == "p2pinv" else "_"

    # load/reload model:
    ldm_stable = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model_path, torch_dtype=torch.float16
    ).to(device)
    ldm_stable.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin",
        image_encoder_folder=None,
    )
    ldm_stable.set_ip_adapter_scale(args.ip_adapter_scale)
    dtype = ldm_stable.dtype

    # Initialize FaceEmbeddingExtractor instance
    extractor = FaceEmbeddingExtractor(
        ctx_id=0,
        det_thresh=args.det_thresh,
        det_size=(args.det_size, args.det_size),
        model_path=args.insightface_model_path,
    )  # Use GPU (ctx_id=0), or CPU with ctx_id=-1

    # Open the output file in write mode
    with open(args.output_file, "w") as f:
        for i in tqdm(range(len(full_data))):
            current_image_data = full_data[i]
            source_image_path = current_image_data["source_image"]
            target_image_path = current_image_data["target_image"]
            mask_image_path = current_image_data["mask_image"]
            prompt_src = current_image_data.get(
                "source_prompt", ""
            )  # default empty string
            prompt_tar_list = current_image_data["target_prompts"]

            prompt_src = ""
            prompt_tar_list = [""]

            do_anonymization = source_image_path == target_image_path
            # Extract embedding for the largest face with scaling
            try:
                id_embs_inv, id_embs = extractor.get_face_embeddings(
                    image_path=source_image_path,
                    max_angle=args.max_angle,
                    is_opposite=do_anonymization,
                    seed=args.seed,
                    scale_factor=args.id_emb_scale,
                    dtype=dtype,
                    device=device,
                )
            except ValueError as e:
                # Write the filename to the text file
                f.write(f"{e}\n")
            else:
                if args.mode == "p2pddim" or args.mode == "ddim":
                    scheduler = DDIMScheduler(
                        beta_start=0.00085,
                        beta_end=0.012,
                        beta_schedule="scaled_linear",
                        clip_sample=False,
                        set_alpha_to_one=False,
                    )
                    ldm_stable.scheduler = scheduler
                else:
                    ldm_stable.scheduler = DDIMScheduler.from_config(
                        sd_model_path, subfolder="scheduler"
                    )

                ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

                # load image
                offsets = (0, 0, 0, 0)
                x0 = load_512(target_image_path, *offsets, device).to(dtype=dtype)

                # Check if the target mask path exists. If it does, load the mask image.
                # Otherwise, create a new black image with the same size as the target image.
                if mask_image_path and Path(mask_image_path).is_file():
                    mask_image = load_image(mask_image_path)
                else:
                    print(f"Error: The file '{mask_image_path}' was not found.")
                    # width and height are the dimensions of the target image
                    height, width = x0.shape[-2:]
                    # Create a new image with a white background
                    mask_image = Image.new("RGB", (width, height), "white")

                # vae encode image
                with autocast("cuda"), inference_mode():
                    w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).to(
                        dtype=dtype
                    )

                # find Zs and wts - forward process
                if args.mode == "p2pddim" or args.mode == "ddim":
                    wT = ddim_inversion(ldm_stable, w0, prompt_src, cfg_scale_src)
                else:
                    wt, zs, wts = inversion_forward_process(
                        ldm_stable,
                        w0,
                        etas=eta,
                        prompt=prompt_src,
                        cfg_scale=cfg_scale_src,
                        prog_bar=True,
                        num_inference_steps=args.num_diffusion_steps,
                        ip_adapter_image_embeds=[id_embs_inv],
                    )

                # iterate over decoder prompts
                for k in range(len(prompt_tar_list)):
                    prompt_tar = prompt_tar_list[k]

                    # Check if number of words in encoder and decoder text are equal
                    src_tar_len_eq = len(prompt_src.split(" ")) == len(
                        prompt_tar.split(" ")
                    )

                    for cfg_scale_tar in cfg_scale_tar_list:
                        for skip in skip_zs:
                            generator = torch.manual_seed(args.seed)
                            if args.mode == "our_inv":
                                # reverse process (via Zs and wT)
                                # controller = AttentionStore()
                                controller = None
                                # register_attention_control(ldm_stable, controller)
                                w0, _ = inversion_reverse_process(
                                    ldm_stable,
                                    xT=wts[args.num_diffusion_steps - skip],
                                    etas=eta,
                                    prompts=[prompt_tar],
                                    cfg_scales=[cfg_scale_tar],
                                    prog_bar=True,
                                    zs=zs[: (args.num_diffusion_steps - skip)],
                                    controller=controller,
                                    ip_adapter_image_embeds=[id_embs],
                                    init_image=x0,
                                    mask_image=mask_image,
                                    generator=generator,
                                    mask_delay_steps=args.mask_delay_steps,
                                )

                            elif args.mode == "p2pinv":
                                # inversion with attention replace
                                cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                                prompts = [prompt_src, prompt_tar]
                                if src_tar_len_eq:
                                    controller = AttentionReplace(
                                        prompts,
                                        args.num_diffusion_steps,
                                        cross_replace_steps=args.xa,
                                        self_replace_steps=args.sa,
                                        model=ldm_stable,
                                    )
                                else:
                                    # Should use Refine for target prompts with different number of tokens
                                    controller = AttentionRefine(
                                        prompts,
                                        args.num_diffusion_steps,
                                        cross_replace_steps=args.xa,
                                        self_replace_steps=args.sa,
                                        model=ldm_stable,
                                    )

                                register_attention_control(ldm_stable, controller)
                                w0, _ = inversion_reverse_process(
                                    ldm_stable,
                                    xT=wts[args.num_diffusion_steps - skip],
                                    etas=eta,
                                    prompts=prompts,
                                    cfg_scales=cfg_scale_list,
                                    prog_bar=True,
                                    zs=zs[: (args.num_diffusion_steps - skip)],
                                    controller=controller,
                                    ip_adapter_image_embeds=[id_embs],
                                )
                                w0 = w0[1].unsqueeze(0)

                            elif args.mode == "p2pddim" or args.mode == "ddim":
                                # only z=0
                                if skip != 0:
                                    continue
                                prompts = [prompt_src, prompt_tar]
                                if args.mode == "p2pddim":
                                    if src_tar_len_eq:
                                        controller = AttentionReplace(
                                            prompts,
                                            args.num_diffusion_steps,
                                            cross_replace_steps=0.8,
                                            self_replace_steps=0.4,
                                            model=ldm_stable,
                                        )
                                    # Should use Refine for target prompts with different number of tokens
                                    else:
                                        controller = AttentionRefine(
                                            prompts,
                                            args.num_diffusion_steps,
                                            cross_replace_steps=0.8,
                                            self_replace_steps=0.4,
                                            model=ldm_stable,
                                        )
                                else:
                                    controller = EmptyControl()

                                register_attention_control(ldm_stable, controller)
                                # perform ddim inversion
                                cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                                w0, latent = text2image_ldm_stable(
                                    ldm_stable,
                                    prompts,
                                    controller,
                                    args.num_diffusion_steps,
                                    cfg_scale_list,
                                    None,
                                    wT,
                                )
                                w0 = w0[1:2]
                            else:
                                raise NotImplementedError

                            # vae decode image
                            with autocast("cuda"), inference_mode():
                                x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                            if x0_dec.dim() < 4:
                                x0_dec = x0_dec[None, :, :, :]
                            img = image_grid(x0_dec)

                            id_emb_scale = (
                                args.id_emb_scale + 1.0
                                if args.shift_neg_id_emb_scale
                                else args.id_emb_scale
                            )
                            shifted_cfg_scale_tar = cfg_scale_tar + args.shift_neg_cfg

                            # Replace dots with underscores.
                            # Format cfg to have at least 4 characters in total, including one digit after the decimal point, and pad it with leading zeros if necessary.
                            filename_wo_ext = f"{Path(source_image_path).stem}-{Path(target_image_path).stem}-cfg-tar-{shifted_cfg_scale_tar:04.1f}-skip-{skip}-id-{id_emb_scale}".replace(
                                ".", "_"
                            )

                            img.save(filename_wo_ext + ".png")
