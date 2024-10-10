from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput


@dataclass
class LEditsPPInversionPipelineOutput(BaseOutput):
    """
    Output class for LEdits++ Diffusion pipelines.

    Args:
        vae_reconstruction_images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of VAE reconstruction of all input images as PIL images of length `batch_size` or NumPy array of shape
            ` (batch_size, height, width, num_channels)`.
    """

    vae_reconstruction_images: Union[List[PIL.Image.Image], np.ndarray]
