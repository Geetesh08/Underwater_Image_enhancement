# Underwater-image-enhancement
## Abstract
Underwater images find application in various fields, like marine research, inspection of
aquatic habitat, underwater surveillance, identification of minerals, and more. However,
underwater shots are affected a lot during the acquisition process due to the absorption
and scattering of light. As depth increases, longer wavelengths get absorbed by water;
therefore, the images appear predominantly bluish-green, and red gets absorbed due to
higher wavelength. These phenomenons result in significant degradation of images due to
which images have low contrast, color distortion, and low visibility. Hence, underwater
images need enhancement to improve the quality of images to be used for various
applications while preserving the valuable information contained in them.
This project aims to provide a comprehensive toolbox for enhancing underwater images. Underwater photography often faces challenges such as color distortion, low contrast, and reduced visibility. This toolbox offers various image processing techniques to address these issues and improve the overall quality of underwater images.

## Features

- **Color Correction**: Functions for compensating red and blue channels to correct color distortion commonly found in underwater images.
- **White Balancing**: Implementation of the Gray World algorithm for white balancing to restore natural color tones in underwater scenes.
- **Contrast Enhancement**: Tools for enhancing contrast using HSV global equalization, which improves the visibility of objects in low-contrast underwater images.
- **Image Fusion**: Provides both average fusion and Principal Component Analysis (PCA) fusion methods for combining information from multiple underwater images to create a single enhanced image.
- **Evaluation Metrics**: Includes functions for evaluating the quality of enhanced images using metrics such as Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).

