from torch import Tensor

import torch
import torch.nn as nn

from TorchJaekwon.Model.Diffusion.External.diffusers.DiffusersWrapper import DiffusersWrapper
from TorchJaekwon.Model.Diffusion.External.diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from guided_diffusion_modules.GuidedDiffusionUnet import GuidedDiffusionUnet
from MelDDPM import MelDDPM

sample_rate:int = 16000
duration_sec:float = 4.0
nfft:int = 1024
hop_size:int = nfft//4
mel_size:int = 64
frequency_min:float = 0
frequency_max:float = sample_rate//2
frame_size:int = int((sample_rate * duration_sec) // hop_size)

diffusion_denoiser:nn.Module = GuidedDiffusionUnet(
    image_size = None,
    in_channel = 1,
    inner_channel = 64,
    out_channel = 1,
    res_blocks = 2,
    attn_res = [8]
)

mel_ddpm:nn.Module = MelDDPM(
    model = diffusion_denoiser,
    model_output_type = 'v_prediction',
    unconditional_prob = 0.1,
    nfft = nfft,
    hop_size = hop_size,
    sample_rate = sample_rate,
    mel_size = mel_size,
    frequency_min = frequency_min,
    frequency_max = frequency_max,
    frame_size=frame_size
)

audio:Tensor = torch.rand(16, 1, int(sample_rate * duration_sec))
condition = {"class_label": torch.randint(0, 10, (16, 1))}
            
diffusion_loss = mel_ddpm( x_start = audio, cond = condition, is_cond_unpack = True)

audio:Tensor = DiffusersWrapper.infer(
    ddpm_module = mel_ddpm,
    diffusers_scheduler_class = DPMSolverMultistepScheduler,
    x_shape = None,
    cond = {"class_label": torch.tensor([[1]])},
    is_cond_unpack = True,
    num_steps = 8,
    cfg_scale = 3.5
)

print('Good luck! CCTV!!')