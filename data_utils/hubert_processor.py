from transformers import Wav2Vec2Processor, HubertModel
import torch
import librosa
import soundfile as sf
import numpy as np

class HubertProcessor:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HubertProcessor, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        print("Initializing the HuBERT Processor and Model...")
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        print("Initialization Finished")

    @torch.no_grad()
    def get_hubert_from_16k_speech(self, speech, device="cuda:0"):
        self.hubert_model = self.hubert_model.to(device)
        if speech.ndim ==2:
            speech = speech[:, 0] # [T, 2] ==> [T,]
        input_values_all = self.wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
        input_values_all = input_values_all.to(device)
        # For long audio sequence, due to the memory limitation, we cannot process them in one run
        # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
        # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
        # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
        # We have the equation to calculate out time step: T = floor((t-k)/s)
        # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
        # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
        res_lst = []
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            input_values = input_values_all[:, start_idx: end_idx]
            hidden_states = self.hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
            res_lst.append(hidden_states[0])
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
        # if input_values.shape[1] != 0:
        if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
            hidden_states = self.hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
            res_lst.append(hidden_states[0])
        ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
        # assert ret.shape[0] == expected_T
        assert abs(ret.shape[0] - expected_T) <= 1
        if ret.shape[0] < expected_T:
            ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
        else:
            ret = ret[:expected_T]
        return ret

    def make_even_first_dim(self, tensor):
        size = list(tensor.size())
        if size[0] % 2 == 1:
            size[0] -= 1
            return tensor[:size[0]]
        return tensor
    
    def process_audio(self, wav_path):
        
        speech, sr = sf.read(wav_path)

        speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)  
        print(f"Processing {wav_path}...")

        hubert_hidden = self.get_hubert_from_16k_speech(speech_16k)

        hubert_hidden = self.make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)
        np.save(wav_path.replace('.wav', '_hu.npy'), hubert_hidden.detach().numpy())
        print(f"Saved features to {wav_path.replace('.wav', '_hu.npy')}")
        print(hubert_hidden.detach().numpy().shape)