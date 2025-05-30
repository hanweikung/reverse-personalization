# Reverse Personalization

This repository provides tools for anonymizing multiple persons in a given image using diffusion models and face embeddings.

## Environment Setup

To install all required dependencies, create a new conda environment using the provided `environment.yml` file:

```sh
conda env create -f environment.yml
```

Then activate the environment:

```sh
conda activate <your_env_name>
```

## Example Usage

You can use the `anonymize_multiple_persons_in_image` function to anonymize faces in an image. Here is an example of how to use it in your own script:

```python
from anonymize_multiple_persons_in_image import anonymize_multiple_persons_in_image

def main():
    input_image_path = "your_input_image.jpg"
    anon_image = anonymize_multiple_persons_in_image(
        input_image=input_image_path,
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
    )
    # anon_image.show()  # Uncomment to display the anonymized image

if __name__ == "__main__":
    main()
```

---
