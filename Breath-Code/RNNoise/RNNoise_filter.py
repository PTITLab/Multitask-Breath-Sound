from rnnoise_wrapper import RNNoise
from glob import glob

denoiser = RNNoise()

list_path_wavs = glob('../../Breath-Data/*.wav')
root_output = '../../Breath-Data/output_filter/'

for path_wav in list_path_wavs:
    audio = denoiser.read_wav(path_wav)
    denoised_audio = denoiser.filter(audio, sample_rate=8000, voice_prob_threshold=0.0, save_source_sample_rate=True)
    path_output = root_output + path_wav.split('/')[-1]
    denoiser.write_wav(path_output, denoised_audio)