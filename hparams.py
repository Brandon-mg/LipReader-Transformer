random_seed = 0
num_workers = 0
batch_size = 24
T = 90
img_size = 96
mel_step_size = 240
fps = 30
num_mels = 80
bce_weights = 7
outputs_per_step =1
num_hidden = 384
emb_size = 512
nhead = 4
nlayers = 3
max_db = 100
ref_db = 20
lr = 1e-4
warmup_steps = 0.2
epochs = 100000
save_step = 1
image_step = 1000
data_root = '/content'
checkpoint_path = './logs/checkpoint'
mel_path = './logs/mel'
wav_path = './logs/wavs'
plot_dir = './logs/plots'

#Encoder Conv3D
num_init_filters = 24
encoder_n_convolutions = 5

# Mel spectrogram
n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
sample_rate = 16000  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

# Mel and Linear spectrograms normalization/scaling and clipping
use_lws=False
signal_normalization = True
# Whether to normalize mel spectrograms to some predefined range (following below parameters)
allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
symmetric_mels = True
# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
# faster and cleaner convergence)
max_abs_value = 4.
# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
# be too big to avoid gradient explosion,
# not too small for fast convergence)
normalize_for_wavenet = True
# whether to rescale to [0, 1] for wavenet. (better audio quality)
clip_for_wavenet = True
# whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)

# Contribution by @begeekmyfriend
# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
# levels. Also allows for better G&L phase reconstruction)
preemphasize = True # whether to apply filter
preemphasis = 0.97  # filter coefficient.

# Limits
min_level_db = -100
ref_level_db = 20
fmin = 55
# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
fmax = 7600  # To be increased/reduced depending on data.

# Griffin Lim
power = 1.5
# Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
griffin_lim_iters = 60
# Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.