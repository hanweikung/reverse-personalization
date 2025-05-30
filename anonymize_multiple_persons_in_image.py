import torch
from PIL import Image
import face_alignment
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


def anonymize_multiple_persons_in_image(
    input_image,
    sd_model_path="stabilityai/stable-diffusion-xl-base-1.0",
    insightface_model_path="~/.insightface",
    device_num=0,
    skip=0.7,
    id_emb_scale=1.0,
    guidance_scale=-10.0,
    num_inversion_steps=100,
    face_image_size=1024,
    det_thresh=0.1,
    ip_adapter_scale=1.0,
    det_size=640,
    seed=0,
):
    dtype = torch.float16
    device = f"cuda:{device_num}"

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype
    )

    anon_image = image = Image.open(input_image)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )
    face_images, image_to_face_matrices = extract_faces(fa, image, face_image_size)

    pipe = StableDiffusionPipelineXL_LEDITS.from_pretrained(
        sd_model_path,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        sd_model_path,
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
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    extractor = FaceEmbeddingExtractor(
        ctx_id=0,
        det_thresh=det_thresh,
        det_size=(det_size, det_size),
        model_path=insightface_model_path,
    )

    for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
        try:
            id_embs_inv, id_embs = extractor.get_face_embeddings(
                image_path=face_image,
                seed=seed,
                scale_factor=id_emb_scale,
                dtype=dtype,
                device=device,
            )
        except ValueError as e:
            print(e)
            print("Consider using a lower det_thresh or a different det_size.")
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

            _ = pipe.invert(
                image=face_image,
                num_inversion_steps=num_inversion_steps,
                skip=skip,
                source_guidance_scale=guidance_scale,
                ip_adapter_image_embeds=[id_embs_inv],
                generator=generator,
            )

            anon_face_image = pipe(
                prompt="",
                ip_adapter_image_embeds=[id_embs],
                num_images_per_prompt=1,
                generator=generator,
                guidance_scale=guidance_scale,
                timesteps=pipe.scheduler.timesteps,
                latents=pipe.init_latents,
                num_inference_steps=num_inversion_steps,
            ).images[0]

            anon_image = paste_foreground_onto_background(
                anon_face_image, anon_image, image_to_face_mat
            )

    return anon_image
