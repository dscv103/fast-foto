## Kodak Portra 400 Film Emulation: Technical Specifications for Python Implementation

Based on extensive research into Kodak's official technical documentation and color science analysis, I've compiled comprehensive specifications for implementing a faithful Portra 400 film emulation in Python. The specifications document has been created and saved for your AI coding agent.

### Core Characteristics of Portra 400

**Color Signature**

Kodak Portra 400 is renowned for its exceptional skin tone rendering and natural color reproduction. The film's defining characteristics include:[1][2]

**Skin Tones**: The most critical aspect is warm, peachy skin rendering with smooth transitions. The color falls primarily in the orange-red spectrum (RGB: approximately 80% red, 70% green, 60% blue). This creates flattering, natural-looking portraits without harsh blemishes or color shifts.[3][4]

**Color Behavior by Hue**:

- **Greens**: Shift toward olive/yellow tones with 15-30% reduced saturation compared to digital. Hue shift of +15° to +25° toward yellow is essential[5][6]
- **Blues**: Soft, slightly cyan-shifted with reduced saturation [85-95% of digital intensity](7)[5]
- **Reds/Oranges**: Enhanced brightness (+5-10% luminance) and increased saturation [105-115%](8)[9]

**Tone Curve and Contrast**

Portra 400 exhibits low contrast with a gentle S-curve. Key specifications:[10][11]

- **Black point**: Lifted to approximately 0.05 (not pure 0), preserving shadow detail
- **Highlight rolloff**: Soft compression starting at 0.9, topping at 0.98 to avoid clipping
- **Mid-tone handling**: Slight lift (+2-4%) maintaining smooth transitions
- **Overall gamma**: The characteristic S-curve comes from the positive conversion, not the negative itself[11]

### Technical Implementation Pipeline

**Processing Steps**:

1. **Image Loading**: Support JPEG, PNG, JPG, RAW, and DNG formats. RAW files processed via rawpy library with linear output
2. **Color Space**: Process in linear RGB (float, 0-1 range) for accurate color math
3. **White Balance**: Apply warm shift with multipliers: R×1.02, G×1.0, B×0.92 [equivalent to ~5200-5500K](12)[3]
4. **HSL Adjustments**: Critical step applying per-hue modifications to achieve Portra's signature color rendering
5. **RGB Channel Curves**: Per-channel tone curves with blue reduction being most dramatic[13][7]
6. **Master Tone Curve**: Apply gentle S-curve with lifted blacks and compressed highlights
7. **Saturation**: Global reduction to 92-98% for "rich but subdued" look[14][10]
8. **Grain Application**: Fine, color-variant T-GRAIN simulation with gaussian noise [σ=0.01, blur radius 2.5px](1)[11]
9. **Output Conversion**: Linear to sRGB gamma correction and export

### Spectral Sensitivity Data

From Kodak's official technical sheet, Portra 400 uses a three-layer emulsion structure:[2][1]

- **Yellow-forming layer** (blue-sensitive): 400-500nm peak
- **Magenta-forming layer** (green-sensitive): 500-600nm peak  
- **Cyan-forming layer** (red-sensitive): 600-700nm peak

The film incorporates T-GRAIN optimized emulsions with antenna dye sensitization in cyan and magenta layers, plus proprietary DIR (Development Inhibitor Releasing) couplers for enhanced sharpness.[1]

### Implementation Libraries

**Core Requirements**:

- **Pillow (PIL)**: Image I/O and basic operations
- **NumPy**: Array processing and mathematical operations
- **pillow-lut**: 3D LUT application support[15][16]
- **rawpy**: RAW/DNG file handling
- **colour-science** (optional): Advanced color space transformations

### Validation Methodology

To ensure faithful emulation, validate against:

1. **Color Chart Testing**: ColorChecker SG comparison with target ΔE < 2.0[11]
2. **Skin Tone Verification**: Vectorscope check at ~135° [orange-red line](17)
3. **Dynamic Range Testing**: Gradient ramps, overexposed and underexposed scenes
4. **Characteristic Testing**: Verify green olive shift, blue softness, and low contrast

### Edge Cases

**Underexposure**: Portra 400 shifts yellow-green in shadows when underexposed—this is authentic behavior[18][19]

**Overexposure**: Film is extremely forgiving with overexposure; highlight detail and color saturation are retained exceptionally well[20][14]

**Purple/Magenta Desaturation**: Certain purple hues (280-320°) may desaturate—this matches actual film behavior[21]

### Configuration Parameters

All transformation parameters are provided as configurable values including:

- White balance multipliers
- HSL adjustments per hue range
- Tone curve control points
- Grain characteristics
- Global saturation levels
- Exposure compensation

The complete technical specification document includes detailed curve points, color matrices, implementation architecture, code structure recommendations, and success criteria. This specification is designed to be unambiguous and directly implementable by an AI coding agent, with all numerical values, formulas, and processing steps explicitly defined.

[1](https://125px.com/docs/film/kodak/e4050-Portra-400.pdf)
[2](https://business.kodakmoments.com/sites/default/files/files/products/e4050_portra_400.pdf)
[3](https://imagen-ai.com/valuable-tips/kodak-portra-400-lightroom-preset/)
[4](https://pixelationblog.wordpress.com/2009/07/15/correcting-skin-color-skin-tones-in-lightroom/)
[5](https://imagen-ai.com/valuable-tips/portra-400-lightroom-preset/)
[6](https://www.vsco.co/features/film-filters/kodak-presets)
[7](https://medialoot.com/blog/how-to-emulate-the-portra-400-film-look-in-lightroom/)
[8](https://www.tonywodarck.com/education/2023/6/18/kodak-portra-400-film-guide)
[9](https://www.shopmoment.com/articles/kodak-portra-400-review-the-film-stock-everyone-loves)
[10](https://casualphotophile.com/2015/06/12/film-profile-kodak-portra-400/)
[11](https://www.youtube.com/watch?v=0YC6YzmXmD0)
[12](https://fujixweekly.com/2022/12/16/kodak-portra-400-v2-fujifilm-x-t5-x-trans-v-film-simulation-recipe/)
[13](https://www.youtube.com/watch?v=TIZFooNHu94)
[14](https://timlaytonfineart.com/portra/)
[15](https://stackoverflow.com/questions/73341263/apply-3d-luts-cube-files-into-an-image-using-python)
[16](https://pillow-lut-tools.readthedocs.io)
[17](https://gofilmnow.com/blog/get-good-skin-tones/)
[18](https://www.youtube.com/watch?v=970RZ44Dq8w)
[19](https://www.reddit.com/r/analog/comments/11ouaee/first_time_film_what_is_causing_this_greenish/)
[20](https://www.alexburkephoto.com/blog/2019/5/16/metering-and-shooting-kodak-portra-film)
[21](https://www.photrio.com/forum/threads/is-it-possible-for-a-particular-color-to-fall-outside-of-the-portra-400-gamut.200515/)
[22](http://link.springer.com/10.1007/s12221-019-1223-8)
[23](https://www.semanticscholar.org/paper/15ba4f0144839076b722fcfa4b00bffa87ab8ff5)
[24](http://www.jstage.jst.go.jp/article/jlve/5/2/5_2_2_7/_article)
[25](https://www.semanticscholar.org/paper/6080d890e2b39e36e51376016094cca85212be8c)
[26](http://ieeexplore.ieee.org/document/1370397/)
[27](https://www.semanticscholar.org/paper/56195029fef2c528c8bb5e7a5e29b0b29950f6d5)
[28](https://onlinelibrary.wiley.com/doi/10.1002/9780470994375.app7)
[29](https://meetingorganizer.copernicus.org/EPSC2020/EPSC2020-1030.html)
[30](http://jacow.org/rupac2016/doi/JACoW-RuPAC2016-THPSC068.html)
[31](https://library.imaging.org/jist/articles/45/4/art00003)
[32](https://arxiv.org/pdf/2207.03649.pdf)
[33](https://www.shopmoment.com/products/kodak-professional-portra-400-35mm-film)
[34](https://www.youtube.com/watch?v=x5wtC9Ud5oI)
[35](https://pixls.us/articles/color-curves-matching/)
[36](https://www.reddit.com/r/AnalogCommunity/comments/1gha5yl/portra_400_digital_simulation_vs_analog/)
[37](https://smashandgrabphoto.wordpress.com/2014/04/09/the-science-and-pseudoscience-of-icc-profiles-and-scanning-colour-negative-film/)
[38](https://reggiebphotography.com/blog/the-most-versatile-fujifilm-x-trans-iv-film-simulation-recipe-reggies-portra)
[39](https://film.recipes/2024/12/16/kodak-portra-400-classic-film-recipe/)
[40](https://www.reddit.com/r/AnalogCommunity/comments/1ekran1/scanning_color_negative_film_with_rgb_light/)
[41](http://arxiv.org/pdf/2112.03536.pdf)
[42](https://www.reddit.com/r/Python/comments/4u7qlu/pillow_vs_opencv/)
[43](https://filtergrade.com/product/kodak-portra-400-lut-pack/)
[44](https://learningdaily.dev/what-is-the-difference-between-opencv-and-pillow-457e37b7d530)
[45](https://realpython.com/image-processing-with-the-python-pillow-library/)
[46](https://www.youtube.com/watch?v=JIejmwjXWYo)
[47](https://www.geeksforgeeks.org/python/python-pillow-tutorial/)
[48](https://www.cobalt-image.com/product/cobalt-elite-portra-video/)
[49](https://www.youtube.com/watch?v=DWymRxeR-tk)
[50](https://www.reddit.com/r/AnalogCommunity/comments/y4y59l/odd_colors_from_portra_400_details_in_comments/)
[51](https://www.semanticscholar.org/paper/c62e1ea16a2e429a36713385514af85a713547a0)
[52](http://arxiv.org/pdf/2407.19921.pdf)
[53](https://arxiv.org/pdf/2401.01569.pdf)
[54](https://joss.theoj.org/papers/10.21105/joss.07120)
[55](https://arxiv.org/pdf/2209.01749.pdf)
[56](https://arxiv.org/html/2404.10133v1)
[57](https://arxiv.org/abs/2412.15438)
[58](http://arxiv.org/pdf/2311.03943v2.pdf)
[59](https://peerj.com/articles/453)
[60](https://www.youtube.com/watch?v=Pkr1vlf_GgM)
[61](https://www.reddit.com/r/ColorGrading/comments/1n0r48j/film_emulation_of_kodak_ektachrome_100_with/)
[62](https://www.youtube.com/watch?v=CjnvM1TUU3I)
[63](https://www.youtube.com/watch?v=gXGSJKxu-VU)
[64](https://www.reddit.com/r/comfyui/comments/1dthmqg/how_do_i_create_a_cube_file_for_the_image_apply/)
[65](https://pypi.org/project/pycubelut/)
[66](https://lisyarus.github.io/blog/posts/transforming-colors-with-matrices.html)
[67](https://stackoverflow.com/questions/49792632/ideas-on-generating-3d-luts-programmatically)
[68](https://github.com/phyng/lutlib)
[69](https://www.reddit.com/r/colorists/comments/p0iube/i_was_going_through_juan_mularas_blogi_came/)
[70](https://colab.research.google.com/github/steveseguin/color-grading/blob/master/colab.ipynb)
[71](https://www.reddit.com/r/photography/comments/6dhfq1/for_those_familiar_with_programming_is_there_a/)
[72](https://mononodes.com/film-emulation/)
[73](https://vitek-fkh.uwks.ac.id/index.php/jv/article/view/225)
[74](https://journals.ashs.org/view/journals/hortsci/49/4/article-p460.xml)
[75](http://www.atlantis-press.com/php/paper-details.php?id=25861122)
[76](https://www.semanticscholar.org/paper/66773ff78e3e25fc4757265cb188408a1decaf60)
[77](https://www.semanticscholar.org/paper/7264f4cc9ea0026fa17bcd1261e49909fd2fddb6)
[78](http://medrxiv.org/cgi/content/short/2025.05.08.25327244v1?rss=1)
[79](https://escholarship.org/content/qt9hn6m6vv/qt9hn6m6vv.pdf?t=pmekqe)
[80](http://arxiv.org/pdf/2410.21005.pdf)
[81](https://zenodo.org/record/4220825/files/Ruili_2020_Assessing%20skin%20tone%20heterogeneity%20under%20various%20light%20sources_Ingenta.pdf)
[82](https://arxiv.org/html/2406.15848v1)
[83](https://pmc.ncbi.nlm.nih.gov/articles/PMC11235144/)
[84](https://onlinelibrary.wiley.com/doi/10.1111/srt.13486)
[85](https://www.pythoninformer.com/python-libraries/pillow/imageops-colour-effects/)
[86](https://alchemycolor.com/film-emulation/)
[87](https://www.geeksforgeeks.org/python/python-pil-imageops-colorize-method/)
[88](https://noamkroll.com/how-to-make-digital-footage-look-like-film-camera-choice-color-workflow-film-grain-more/)
[89](https://www.tutorialspoint.com/python_pillow/python_pillow_imageops_colorize_function.htm)
[90](https://www.youtube.com/watch?v=r9vAr9LLang)
[91](https://www.photrio.com/forum/threads/need-advise-portra400-green-color-shift-while-developing-at-home.208421/)
[92](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html)
[93](https://chasejarvis.com/blog/four-great-ways-to-get-the-look-of-film-in-the-digital-darkroom/)
[94](https://www.tutorialspoint.com/python_pillow/python_pillow_correcting_color_balance.htm)
[95](https://forum.dxo.com/t/integration-of-film-curve-profiles-in-dxo-pureraw-for-enhanced-dng-pre-processing-workflow/36546)
[96](https://library.imaging.org/jist/articles/57/6/art00005)
[97](https://www.mdpi.com/2073-4352/15/12/1035)
[98](https://czasopisma.up.lublin.pl/asphc/article/view/5363)
[99](https://www.mdpi.com/2075-163X/12/11/1344)
[100](https://link.springer.com/10.1007/s10904-025-03612-y)
[101](https://www.mdpi.com/2673-6373/4/3/25)
[102](https://www.trjfas.org/uploads/pdf_15116.pdf)
[103](http://link.springer.com/10.1007/s00344-012-9316-2)
[104](https://www.mdpi.com/2073-4352/15/9/775)
[105](http://ieeexplore.ieee.org/document/7952404/)
[106](https://arxiv.org/pdf/1903.06490.pdf)
[107](https://www.degruyter.com/document/doi/10.1515/secm-2022-0189/pdf)
[108](https://library.imaging.org/cic/articles/28/1/art00044)
[109](https://www.jstatsoft.org/index.php/jss/article/view/v096i01/v96i01.pdf)
[110](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-23/issue-2/029901/Color-Appearance-Models/10.1117/1.JEI.23.2.029901.pdf)
[111](https://downloads.hindawi.com/journals/am/2012/273723.pdf)
[112](https://arxiv.org/html/2405.17725v1)
[113](https://www.mdpi.com/2571-9408/6/8/300/pdf?version=1691223391)
[114](https://tinker.koraks.nl/photography/flipped-doing-color-negative-inversions-manually/)
[115](https://discuss.pixls.us/t/spectral-film-simulations-from-scratch/48209)
[116](https://www.photrio.com/forum/threads/how-do-scanners-color-correct-c-41-negatives.169634/)
[117](https://www.alexburkephoto.com/blog/2019/10/16/manual-inversion-of-color-negative-film)
[118](https://kevinmartinjose.com/2021/04/27/film-simulations-from-scratch-using-python/)
[119](https://www.pixl-latr.com/three-step-and-one-step-conversion-of-colour-film-negatives-with-affinity-photo/)
[120](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/advs.202103309)
[121](https://pmc.ncbi.nlm.nih.gov/articles/PMC3948926/)
[122](http://arxiv.org/pdf/2211.16076.pdf)
[123](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15252)
[124](https://akjournals.com/downloadpdf/journals/606/14/1/article-p3.pdf)
[125](https://www.youtube.com/watch?v=xdLNMpYyP4k)
[126](https://www.youtube.com/watch?v=cesKJdP3o2s)
[127](https://www.reddit.com/r/AskPhotography/comments/1m9yit6/how_to_achieve_the_contrast_of_film_cameras/)
[128](https://colorfinale.com/blog/post/cf-skin-tones-two-02-25)
