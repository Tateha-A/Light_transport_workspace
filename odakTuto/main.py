from odak.learn.wave import stochastic_gradient_descent, \
    calculate_amplitude, calculate_phase
import torch
from odak.learn.wave import gerchberg_saxton
from odak.learn.tools import save_image, load_image
from odak.learn.wave import linear_grating
from odak.tools import resize_image
import numpy

wavelength = 0.000000532
# Pixel pitch and resolution of the phase-only hologram
# or a phase-only spatial light modulator
dx = 0.0000064
resolution = [1080, 1920]

# distance that the light will travel from optimized hologram
distance = 0.15

# target image

data_loader = load_image("target/usaf1951.png")

target = data_loader.type(torch.FloatTensor)
target = target.view(resolution)

# nomalize
print(torch.max(target))
if torch.max(target) > 1:
    target -= target.min(1, keepdim=True)[0]
    target /= target.max(1, keepdim=True)[0]

# target = torch.zeros(resolution[0], resolution[1])
# target[300:700, 400:450] = 1.
print(target)

iteration_number = 300
learning_rate = 0.01
cuda = True
# Propagation type: transfer function Fresnel approach
propagation_type = 'TR Fresnel'


hologram, reconstructed = stochastic_gradient_descent(
        target,
        wavelength,
        distance,
        dx,
        resolution,
        'TR Fresnel',
        iteration_number,
        learning_rate=learning_rate,
        cuda=cuda
    )


reconstructed_intensity = calculate_amplitude(reconstructed)**2
save_image('reconstructed_image.png', reconstructed_intensity, cmin=0., cmax=1.)


slm_range = 2 * 3.14
dynamic_range = 255
phase_hologram = calculate_phase(hologram)
phase_only_hologram = (phase_hologram % slm_range) / slm_range * dynamic_range

save_image('phase_only_hologram.png', phase_only_hologram)

grating = linear_grating(resolution[0], resolution[1], axis='y')\
    .to(phase_hologram.device)
phase_only_hologram_w_grating = phase_hologram + calculate_phase(grating)

phase_only_hologram_w_grating = \
    (phase_only_hologram_w_grating % slm_range) / slm_range * dynamic_range
save_image('phase_only_hologram_w_grating.png', phase_only_hologram_w_grating)