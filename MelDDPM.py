from typing import Optional, Tuple, Union
from torch import Tensor

import torch

from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM
from TorchJaekwon.Util.UtilAudioMelSpec import UtilAudioMelSpec

class MelDDPM(DDPM):
    def __init__(
            self, 
            nfft: int, 
            hop_size: int, 
            sample_rate:int,
            mel_size:int,
            frequency_min:float,
            frequency_max:float,
            frame_size:int,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)
        self.mel_size:int = mel_size
        self.frame_size:int = frame_size
        self.mel_spec_util = UtilAudioMelSpec(nfft, hop_size, sample_rate, mel_size, frequency_min, frequency_max)
        self.mel_max:Tensor = torch.tensor([100 for _ in range(mel_size)]).view(1,1,-1,1)
        self.mel_min:Tensor = torch.tensor([-100 for _ in range(mel_size)]).view(1,1,-1,1)
        self.vocoder = lambda x: torch.rand(x.shape[0], 1,  x.shape[-1] * hop_size)
    
    def preprocess(
            self, 
            x_start:Tensor, 
            cond:Optional[Union[dict, Tensor]] = None,
        ) -> Tuple[Tensor,Tensor]: 
        if x_start is not None:
            mel_spec:Tensor = self.mel_spec_util.get_hifigan_mel_spec(x_start)
            mel_spec = self.normalize_mel(mel_spec)
        else:
            mel_spec = None
        additional_data_dict = None
        return mel_spec, cond, additional_data_dict
    
    def normalize_mel(self, mel_spec:Tensor) -> Tensor:
        return (mel_spec - self.mel_min) / (self.mel_max - self.mel_min) * 2 - 1
    
    def denormalize_mel(self, mel_spec:Tensor) -> Tensor:
        return (mel_spec + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min
        
    def postprocess(
            self, 
            x: Tensor, #[batch, 1, mel, time]
            _
        ) -> Tensor:
        mel_spec:Tensor = self.denormalize_mel(x)
        pred_audio:Tensor = self.vocoder(mel_spec)
        return pred_audio
    
    def get_x_shape(self, cond) -> tuple:
        return (cond['class_label'].shape[0], 1, self.mel_size, self.frame_size)
    
    def get_unconditional_condition(self,
                                    cond:Optional[Union[dict,Tensor]] = None, 
                                    cond_shape:Optional[tuple] = None,
                                    condition_device:Optional[torch.device] = None
                                    ) -> dict:
        return {'class_label': torch.tensor([[11] for _ in range(cond["class_label"].shape[0])]).to(condition_device)}
    
    