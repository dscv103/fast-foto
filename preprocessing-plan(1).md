# Photo Preprocessing for Portra 400 Film Emulation - Complete Python Implementation Plan

## Overview

**Goal**: Automatically optimize digital photos (JPEG/PNG/RAW/DNG) to create the perfect base for Kodak Portra 400 film emulation.

**Principle**: Preprocessing fixes technical issues (exposure, white balance, clipping) while film emulation adds aesthetic character.

---

## 7-Phase Preprocessing Pipeline

### Phase 1: Analysis & Diagnostics

**Purpose**: Detect exposure problems, clipping, and color balance issues.

**Histogram Analysis Code**:
```python
def analyze_histogram(rgb_linear):
    """
    Analyze histogram to detect exposure problems
    
    Args:
        rgb_linear: Linear RGB array (H x W x 3), float, range [0, 1]
    
    Returns:
        dict: Statistics including brightness, shadows, highlights ratios
    """
    # Convert to luminance (ITU-R BT.709 weights)
    luminance = 0.2126 * rgb_linear[:,:,0] + \
                0.7152 * rgb_linear[:,:,1] + \
                0.0722 * rgb_linear[:,:,2]
    
    # Compute statistics
    brightness_mean = luminance.mean()
    brightness_median = np.median(luminance)
    brightness_min = luminance.min()
    brightness_max = luminance.max()
    
    # Percentiles for exposure detection
    shadows_ratio = (luminance < 0.2).sum() / luminance.size
    highlights_ratio = (luminance > 0.8).sum() / luminance.size
    
    # Clipping detection
    is_underexposed = shadows_ratio > 0.30  # > 30% in shadows
    is_overexposed = highlights_ratio > 0.20  # > 20% blown
    
    # Dynamic range
    dynamic_range = brightness_max - brightness_min
    
    # Histogram
    histogram, _ = np.histogram(luminance, bins=256, range=(0, 1))
    
    return {
        'brightness_mean': brightness_mean,
        'brightness_median': brightness_median,
        'shadows_ratio': shadows_ratio,
        'highlights_ratio': highlights_ratio,
        'dynamic_range': dynamic_range,
        'is_underexposed': is_underexposed,
        'is_overexposed': is_overexposed,
        'histogram': histogram,
        'brightness_min': brightness_min,
        'brightness_max': brightness_max
    }
```

**Target Values**:
- **brightness_mean 0.18-0.25**: Well-exposed (18% gray, film standard)
- **brightness_mean < 0.15**: Underexposed, needs lift
- **brightness_mean > 0.50**: Overexposed or bright scene
- **shadows_ratio > 0.40**: Very dark image
- **highlights_ratio > 0.15**: Risk of blown highlights

---

### Phase 2: Automatic Exposure Correction

**Target Brightness**: 0.22 (18% gray standard for film cinematography)

**Exposure Correction Algorithm**:
```python
def calculate_exposure_correction(histogram_stats):
    """
    Calculate exposure adjustment in EV (exposure value) stops
    
    EV = log2(exposure_factor)
    1 stop = 2x brightness (EV = 1)
    -1 stop = 0.5x brightness (EV = -1)
    
    Args:
        histogram_stats: Output from analyze_histogram()
    
    Returns:
        dict: EV correction, exposure factor, and rationale
    """
    brightness_mean = histogram_stats['brightness_mean']
    highlights_ratio = histogram_stats['highlights_ratio']
    shadows_ratio = histogram_stats['shadows_ratio']
    
    # Target: mean brightness around 0.22 (18% gray reference)
    target_brightness = 0.22
    
    # Calculate correction factor
    current_ratio = brightness_mean / target_brightness if brightness_mean > 0 else 1.0
    
    # Clamp exposure correction to safe range: -2 to +1 EV
    ev_correction = np.clip(
        np.log2(1.0 / current_ratio),
        -2.0,  # Don't darken more than 2 stops
        +1.0   # Don't brighten more than 1 stop
    )
    
    # Penalties for extreme clipping
    if highlights_ratio > 0.15:  # > 15% blown
        ev_correction -= 0.5
    
    if shadows_ratio > 0.40:  # > 40% in deep shadows
        ev_correction += 0.3
    
    exposure_factor = 2.0 ** ev_correction
    
    return {
        'ev_correction': ev_correction,
        'exposure_factor': exposure_factor,
        'reason': f"Current brightness: {brightness_mean:.3f}, Target: {target_brightness:.3f}"
    }
```

**Apply Exposure with Soft Clipping**:
```python
def apply_exposure_correction(rgb_linear, exposure_factor):
    """
    Apply linear exposure adjustment with soft highlight compression
    
    Args:
        rgb_linear: Linear RGB array
        exposure_factor: Multiplication factor (from calculate_exposure_correction)
    
    Returns:
        ndarray: Exposure-corrected RGB
    """
    rgb_corrected = rgb_linear * exposure_factor
    
    # Soft clipping: preserve highlights gradually
    rgb_corrected = soft_clip_highlights(rgb_corrected, max_value=1.0)
    
    return np.clip(rgb_corrected, 0, 1)

def soft_clip_highlights(rgb, max_value=1.0, toe_strength=0.1):
    """
    Gently compress highlights without hard clipping (like film)
    Uses log-based toe function
    """
    threshold = 0.8
    
    mask = rgb > threshold
    
    # Log-based toe: gradual compression
    toe = np.where(
        mask,
        threshold + toe_strength * np.log1p((rgb[mask] - threshold) / toe_strength),
        rgb
    )
    
    return np.clip(toe, 0, max_value)
```

**Exposure Decision Logic**:
```
IF brightness_mean < 0.15:
  → Underexposed: Apply +0.5 to +1.5 EV
  
ELIF brightness_mean < 0.22:
  → Slightly underexposed: Apply +0.3 to +1.0 EV
  
ELIF brightness_mean 0.22-0.35:
  → Optimally exposed: Apply 0 to +0.3 EV (minimal)
  
ELIF brightness_mean 0.35-0.50:
  → Slightly overexposed: Apply -0.3 to 0 EV
  
ELIF brightness_mean > 0.50:
  → Heavily overexposed: Apply -0.5 to -1.5 EV
```

---

### Phase 3: White Balance Correction

**Method 1: Gray World Algorithm**:
```python
def auto_white_balance_gray_world(rgb_linear):
    """
    Gray World: Assumes average color in scene is neutral gray
    Fast, works in 80% of cases
    
    Args:
        rgb_linear: Linear RGB array
    
    Returns:
        ndarray: White-balanced RGB
    """
    # Compute channel means
    r_mean = rgb_linear[:, :, 0].mean()
    g_mean = rgb_linear[:, :, 1].mean()
    b_mean = rgb_linear[:, :, 2].mean()
    
    # Target: average of all means
    average_mean = (r_mean + g_mean + b_mean) / 3.0
    
    # Calculate gains (avoid division by zero)
    r_gain = average_mean / r_mean if r_mean > 0.001 else 1.0
    g_gain = average_mean / g_mean if g_mean > 0.001 else 1.0
    b_gain = average_mean / b_mean if b_mean > 0.001 else 1.0
    
    # Apply gains
    rgb_wb = rgb_linear.copy()
    rgb_wb[:, :, 0] *= r_gain
    rgb_wb[:, :, 1] *= g_gain
    rgb_wb[:, :, 2] *= b_gain
    
    return np.clip(rgb_wb, 0, 1)
```

**Method 2: White Patch Algorithm**:
```python
def auto_white_balance_white_patch(rgb_linear, percentile=99.5):
    """
    White Patch: Treats brightest area as white reference
    More robust, handles color casts better
    
    Args:
        rgb_linear: Linear RGB array
        percentile: Percentile to use as "white" (default 99.5)
    
    Returns:
        ndarray: White-balanced RGB
    """
    # Find percentile value for each channel
    r_max = np.percentile(rgb_linear[:, :, 0], percentile)
    g_max = np.percentile(rgb_linear[:, :, 1], percentile)
    b_max = np.percentile(rgb_linear[:, :, 2], percentile)
    
    # Calculate scaling (avoid division by zero)
    r_scale = 1.0 / r_max if r_max > 0.001 else 1.0
    g_scale = 1.0 / g_max if g_max > 0.001 else 1.0
    b_scale = 1.0 / b_max if b_max > 0.001 else 1.0
    
    # Apply scaling
    rgb_wb = rgb_linear.copy()
    rgb_wb[:, :, 0] *= r_scale
    rgb_wb[:, :, 1] *= g_scale
    rgb_wb[:, :, 2] *= b_scale
    
    return np.clip(rgb_wb, 0, 1)
```

**Method 3: Hybrid Approach (RECOMMENDED)**:
```python
def auto_white_balance_hybrid(rgb_linear):
    """
    Hybrid: Combine Gray World + White Patch for best results
    
    Process:
    1. Apply white patch first (normalize highlights)
    2. Apply gray world (neutralize remaining cast)
    3. Blend 70% gray world, 30% white patch
    
    Args:
        rgb_linear: Linear RGB array
    
    Returns:
        ndarray: White-balanced RGB
    """
    # Step 1: White patch (handles color casts)
    rgb_wp = auto_white_balance_white_patch(rgb_linear, percentile=98.0)
    
    # Step 2: Gray world (neutralizes remaining cast)
    rgb_gw = auto_white_balance_gray_world(rgb_wp)
    
    # Step 3: Blend (more gray world for neutrality)
    blend_factor = 0.7  # 70% gray world, 30% white patch
    rgb_hybrid = (1.0 - blend_factor) * rgb_wp + blend_factor * rgb_gw
    
    return np.clip(rgb_hybrid, 0, 1)
```

---

### Phase 4: Highlight Recovery

**Detect Clipping**:
```python
def detect_clipping(rgb_linear, tolerance=0.98):
    """
    Detect channel-specific clipping
    
    Args:
        rgb_linear: Linear RGB array
        tolerance: Threshold for "clipped" (default 0.98)
    
    Returns:
        dict: Clipping statistics and recommendations
    """
    # Count pixels above threshold per channel
    clipped_r = (rgb_linear[:,:,0] > tolerance).sum()
    clipped_g = (rgb_linear[:,:,1] > tolerance).sum()
    clipped_b = (rgb_linear[:,:,2] > tolerance).sum()
    total_pixels = rgb_linear.shape[0] * rgb_linear.shape[1]
    
    channel_clipping = {
        'red': (clipped_r / total_pixels) * 100,
        'green': (clipped_g / total_pixels) * 100,
        'blue': (clipped_b / total_pixels) * 100
    }
    
    overall_clipping = max(channel_clipping.values())
    action_needed = overall_clipping > 5.0  # > 5% clipping
    
    return {
        'overall_clipping_percent': overall_clipping,
        'channel_clipping': channel_clipping,
        'action_needed': action_needed,
        'recommendation': 'Apply highlight recovery' if action_needed else 'No action needed'
    }
```

**Recover Highlight Detail**:
```python
def recover_highlight_detail(rgb_linear, strength=1.0):
    """
    Preserve clipped highlight detail using adaptive compression
    
    Args:
        rgb_linear: Linear RGB array
        strength: Recovery strength (0.0-1.0, default 1.0)
    
    Returns:
        ndarray: Highlight-recovered RGB
    """
    # Compute luminance
    luminance = 0.2126 * rgb_linear[:,:,0] + \
                0.7152 * rgb_linear[:,:,1] + \
                0.0722 * rgb_linear[:,:,2]
    
    # Highlight threshold
    threshold = 0.8
    
    # Compute compression mask (areas above threshold get compressed)
    highlight_mask = np.maximum(0, (luminance - threshold) / (1.0 - threshold))
    
    # Apply recovery curve (log compression for gradual rolloff)
    recovery = 1.0 - (strength * 0.3 * highlight_mask)  # Max 30% compression
    
    # Apply per-channel
    rgb_recovered = rgb_linear.copy()
    for c in range(3):
        rgb_recovered[:,:,c] *= recovery
    
    return np.clip(rgb_recovered, 0, 1)
```

**Settings**: Use strength=0.7 for portraits, 0.6 for landscapes

---

### Phase 5: Shadow Recovery & Lifting

**Detect Shadow Crush**:
```python
def detect_shadow_crush(rgb_linear, threshold=0.05):
    """
    Detect shadow detail loss (crushed blacks)
    
    Args:
        rgb_linear: Linear RGB array
        threshold: Below this value = crushed (default 0.05)
    
    Returns:
        dict: Shadow crush statistics
    """
    luminance = 0.2126 * rgb_linear[:,:,0] + \
                0.7152 * rgb_linear[:,:,1] + \
                0.0722 * rgb_linear[:,:,2]
    
    crushed = (luminance < threshold).sum()
    total_pixels = rgb_linear.shape[0] * rgb_linear.shape[1]
    crushed_percent = (crushed / total_pixels) * 100
    
    # > 5% pure black = shadows crushed
    detail_lost = crushed_percent > 5.0
    
    return {
        'crushed_pixels_percent': crushed_percent,
        'detail_lost': detail_lost,
        'recommendation': 'Apply shadow lift' if detail_lost else 'Shadows look good'
    }
```

**Lift Shadows**:
```python
def lift_shadows(rgb_linear, strength=0.5):
    """
    Lift shadow detail using luminance-based adaptive approach
    
    Args:
        rgb_linear: Linear RGB array
        strength: Lift strength (0.0-1.0, default 0.5)
    
    Returns:
        ndarray: Shadow-lifted RGB
    """
    # Compute luminance
    luminance = 0.2126 * rgb_linear[:,:,0] + \
                0.7152 * rgb_linear[:,:,1] + \
                0.0722 * rgb_linear[:,:,2]
    
    # Shadow threshold
    shadow_threshold = 0.25
    
    # Create lift curve using log function
    # Stronger in deep shadows, gentle in near-midtones
    lift_amount = np.where(
        luminance < shadow_threshold,
        strength * 0.25 * np.log1p(shadow_threshold / np.maximum(luminance, 0.01)),
        0.0
    )
    
    # Clamp lift to reasonable range (0 to +0.3)
    lift_amount = np.clip(lift_amount, 0, 0.3 * strength)
    
    # Apply lift to all channels
    rgb_lifted = rgb_linear.copy()
    for c in range(3):
        rgb_lifted[:,:,c] = np.minimum(1.0, rgb_linear[:,:,c] + lift_amount)
    
    return rgb_lifted
```

**Settings**: Use strength=0.4-0.5 for Portra 400 (gentle, film-like)

---

### Phase 6: Contrast & Tone Normalization

**Normalize Midtones**:
```python
def normalize_midtones(rgb_linear):
    """
    Normalize midtones to neutral position (median = 0.5)
    
    Args:
        rgb_linear: Linear RGB array
    
    Returns:
        ndarray: Midtone-normalized RGB
    """
    # Compute luminance
    luminance = 0.2126 * rgb_linear[:,:,0] + \
                0.7152 * rgb_linear[:,:,1] + \
                0.0722 * rgb_linear[:,:,2]
    
    # Find median luminance
    median_lum = np.median(luminance)
    
    # Target midtone level
    target_midtone = 0.5
    
    if median_lum < 0.001:
        return rgb_linear
    
    # Linear scaling factor
    midtone_scale = target_midtone / median_lum
    
    # Clamp to reasonable range (0.7 to 1.4 = -0.5 to +0.5 EV)
    midtone_scale = np.clip(midtone_scale, 0.7, 1.4)
    
    # Apply scaling
    rgb_normalized = rgb_linear * midtone_scale
    
    return np.clip(rgb_normalized, 0, 1)
```

**Enhance Contrast (Mild)**:
```python
def enhance_contrast_mild(rgb_linear, strength=0.5):
    """
    Apply mild S-curve to enhance contrast before film emulation
    
    Args:
        rgb_linear: Linear RGB array
        strength: Contrast strength (0.0-1.0, default 0.5)
    
    Returns:
        ndarray: Contrast-enhanced RGB
    """
    from scipy.interpolate import CubicSpline
    
    # Define gentle S-curve control points
    input_pts = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    output_pts = np.array([0.0, 0.15, 0.5, 0.85, 1.0])
    
    # Create spline curve
    curve = CubicSpline(input_pts, output_pts, bc_type='natural')
    
    # Apply per-channel
    rgb_contrast = rgb_linear.copy()
    for c in range(3):
        channel = rgb_linear[:,:,c].ravel()
        channel_curved = curve(channel)
        
        # Blend with original based on strength
        channel_blended = (1.0 - strength) * channel + strength * channel_curved
        
        rgb_contrast[:,:,c] = channel_blended.reshape(rgb_linear[:,:,c].shape)
    
    return np.clip(rgb_contrast, 0, 1)
```

---

### Phase 7: Noise Reduction (Optional)

**Bilateral Filter** (for high-ISO images):
```python
def reduce_noise_bilateral(rgb_linear, kernel_size=5, sigma_spatial=10, sigma_range=0.1):
    """
    Apply bilateral filter for edge-preserving noise reduction
    
    Args:
        rgb_linear: Linear RGB array
        kernel_size: Filter neighborhood (5, 7, 9)
        sigma_spatial: Spatial distance decay
        sigma_range: Color/luminance decay
    
    Returns:
        ndarray: Noise-reduced RGB
    """
    import cv2
    
    # Convert to 8-bit for OpenCV
    rgb_8bit = (rgb_linear * 255).astype(np.uint8)
    
    # Apply bilateral filter per channel
    rgb_filtered = cv2.bilateralFilter(
        rgb_8bit,
        d=kernel_size,
        sigmaColor=sigma_range * 255,
        sigmaSpace=sigma_spatial
    )
    
    # Convert back to float
    rgb_filtered = rgb_filtered.astype(np.float32) / 255.0
    
    return rgb_filtered
```

**When to Use**:
- High-ISO images (ISO 3200+): strength 0.6-0.8
- Low-ISO images (ISO 100-400): strength 0.2-0.4
- Well-lit images: No noise reduction needed

---

## Complete Preprocessing Pipeline

### Main Function

```python
def preprocess_for_portra_emulation(
    input_path,
    output_path=None,
    auto_exposure=True,
    auto_white_balance=True,
    highlight_recovery_strength=0.7,
    shadow_lift_strength=0.4,
    contrast_enhancement=0.5,
    noise_reduction=False,
    verbose=True
):
    """
    Complete preprocessing pipeline for Portra 400 emulation
    
    Parameters:
    -----------
    input_path : str
        Path to input image (JPEG, PNG, RAW, DNG)
    output_path : str, optional
        Path to save preprocessed image
    auto_exposure : bool
        Auto-correct exposure to optimal level
    auto_white_balance : bool
        Auto-correct white balance
    highlight_recovery_strength : float (0-1)
        Strength of highlight recovery
    shadow_lift_strength : float (0-1)
        Strength of shadow lifting
    contrast_enhancement : float (0-1)
        Strength of contrast enhancement
    noise_reduction : bool
        Apply bilateral noise reduction (slow)
    verbose : bool
        Print processing steps
    
    Returns:
    --------
    preprocessed_rgb : ndarray
        sRGB image ready for film emulation
    metadata : dict
        Processing metadata
    """
    
    # Step 1: Load image
    if verbose: print("[1/8] Loading image...")
    rgb_linear = load_image_linear(input_path)
    
    # Step 2: Analyze histogram
    if verbose: print("[2/8] Analyzing histogram...")
    hist_stats = analyze_histogram(rgb_linear)
    if verbose:
        print(f"    Mean brightness: {hist_stats['brightness_mean']:.3f}")
        print(f"    Shadows: {hist_stats['shadows_ratio']*100:.1f}%")
        print(f"    Highlights: {hist_stats['highlights_ratio']*100:.1f}%")
    
    # Step 3: Exposure correction
    if auto_exposure:
        if verbose: print("[3/8] Correcting exposure...")
        exp_corr = calculate_exposure_correction(hist_stats)
        rgb_linear = apply_exposure_correction(rgb_linear, exp_corr['exposure_factor'])
        if verbose:
            print(f"    EV correction: {exp_corr['ev_correction']:+.2f} stops")
    else:
        if verbose: print("[3/8] Skipping exposure correction")
        exp_corr = None
    
    # Step 4: White balance
    if auto_white_balance:
        if verbose: print("[4/8] Correcting white balance...")
        rgb_linear = auto_white_balance_hybrid(rgb_linear)
        if verbose: print("    Applied hybrid correction")
    else:
        if verbose: print("[4/8] Skipping white balance")
    
    # Step 5: Highlight recovery
    if verbose: print("[5/8] Recovering highlights...")
    clipping_info = detect_clipping(rgb_linear)
    if clipping_info['action_needed'] and highlight_recovery_strength > 0:
        rgb_linear = recover_highlight_detail(rgb_linear, strength=highlight_recovery_strength)
        if verbose:
            print(f"    Clipping: {clipping_info['overall_clipping_percent']:.1f}%")
            print(f"    Applied {highlight_recovery_strength:.1f} strength recovery")
    else:
        if verbose: print(f"    Clipping: {clipping_info['overall_clipping_percent']:.1f}% (OK)")
    
    # Step 6: Shadow lifting
    if verbose: print("[6/8] Lifting shadows...")
    shadow_info = detect_shadow_crush(rgb_linear)
    if shadow_info['detail_lost'] and shadow_lift_strength > 0:
        rgb_linear = lift_shadows(rgb_linear, strength=shadow_lift_strength)
        if verbose:
            print(f"    Crush: {shadow_info['crushed_pixels_percent']:.1f}%")
            print(f"    Applied {shadow_lift_strength:.1f} strength lift")
    else:
        if verbose: print(f"    Crush: {shadow_info['crushed_pixels_percent']:.1f}% (OK)")
    
    # Step 7: Noise reduction (optional)
    if noise_reduction:
        if verbose: print("[7/8] Reducing noise...")
        rgb_linear = reduce_noise_bilateral(rgb_linear)
    else:
        if verbose: print("[7/8] Skipping noise reduction")
    
    # Step 8: Contrast and tone normalization
    if verbose: print("[8/8] Normalizing tones...")
    rgb_linear = normalize_midtones(rgb_linear)
    if contrast_enhancement > 0:
        rgb_linear = enhance_contrast_mild(rgb_linear, strength=contrast_enhancement)
        if verbose: print(f"    Applied {contrast_enhancement:.1f} contrast")
    
    # Convert to sRGB for output
    if verbose: print("\nConverting to sRGB...")
    rgb_srgb = linear_to_srgb(rgb_linear)
    
    # Save if output path provided
    if output_path:
        if verbose: print(f"Saving to {output_path}...")
        save_image(output_path, rgb_srgb)
    
    if verbose: print("Preprocessing complete!")
    
    # Compile metadata
    metadata = {
        'histogram_stats': hist_stats,
        'exposure_correction': exp_corr,
        'clipping_info': clipping_info,
        'shadow_crush_info': shadow_info,
        'parameters_used': {
            'auto_exposure': auto_exposure,
            'auto_white_balance': auto_white_balance,
            'highlight_recovery_strength': highlight_recovery_strength,
            'shadow_lift_strength': shadow_lift_strength,
            'contrast_enhancement': contrast_enhancement,
            'noise_reduction': noise_reduction
        }
    }
    
    return rgb_srgb, metadata
```

---

## Helper Functions

### Color Space Conversion

```python
def load_image_linear(image_path):
    """
    Load image and convert to linear RGB
    
    Args:
        image_path: Path to JPEG/PNG/RAW/DNG
    
    Returns:
        ndarray: Linear RGB array (H x W x 3), float32, [0, 1]
    """
    import os
    from PIL import Image
    import rawpy
    
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in ['.dng', '.raw', '.cr2', '.nef', '.arw']:
        # RAW file
        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16
            )
        rgb_linear = (rgb / 65535.0).astype(np.float32)
    else:
        # JPEG/PNG
        img = Image.open(image_path).convert('RGB')
        rgb = np.array(img).astype(np.float32) / 255.0
        # Convert sRGB to linear
        rgb_linear = srgb_to_linear(rgb)
    
    return rgb_linear

def srgb_to_linear(rgb_srgb):
    """Convert sRGB to linear RGB"""
    return np.where(
        rgb_srgb <= 0.04045,
        rgb_srgb / 12.92,
        np.power((rgb_srgb + 0.055) / 1.055, 2.4)
    )

def linear_to_srgb(rgb_linear):
    """Convert linear RGB to sRGB"""
    return np.where(
        rgb_linear <= 0.0031308,
        rgb_linear * 12.92,
        1.055 * np.power(rgb_linear, 1/2.4) - 0.055
    )

def save_image(output_path, rgb_srgb):
    """Save sRGB image to file"""
    from PIL import Image
    rgb_8bit = (np.clip(rgb_srgb, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb_8bit, 'RGB')
    img.save(output_path, quality=95)
```

---

## Configuration Presets

### Portrait Photography
```python
PORTRAIT_CONFIG = {
    'auto_exposure': True,
    'auto_white_balance': True,
    'highlight_recovery_strength': 0.8,  # Protect skin
    'shadow_lift_strength': 0.5,         # Gentle fill
    'contrast_enhancement': 0.4,         # Don't over-sharpen
    'noise_reduction': False
}
```

### Landscape Photography
```python
LANDSCAPE_CONFIG = {
    'auto_exposure': True,
    'auto_white_balance': True,
    'highlight_recovery_strength': 0.6,  # Preserve sky
    'shadow_lift_strength': 0.3,         # Keep mood
    'contrast_enhancement': 0.7,         # Enhance detail
    'noise_reduction': False
}
```

### High-ISO Indoor
```python
HIGH_ISO_CONFIG = {
    'auto_exposure': True,
    'auto_white_balance': True,
    'highlight_recovery_strength': 0.5,
    'shadow_lift_strength': 0.6,         # Stronger lift
    'contrast_enhancement': 0.3,
    'noise_reduction': True              # Enable bilateral
}
```

---

## Integration with Film Emulation

```python
# Complete workflow
from preprocessing import preprocess_for_portra_emulation
from portra400_emulation import apply_portra_emulation

# Step 1: Preprocess
rgb_preprocessed, metadata = preprocess_for_portra_emulation(
    input_path="photo.jpg",
    auto_exposure=True,
    auto_white_balance=True,
    highlight_recovery_strength=0.7,
    shadow_lift_strength=0.4,
    contrast_enhancement=0.5,
    verbose=True
)

# Step 2: Apply film emulation
rgb_emulated = apply_portra_emulation(
    rgb_preprocessed,
    exposure_compensation=0.0,  # Already corrected
    grain_strength=0.012,
    scanner_profile='noritsu'
)

# Step 3: Save
save_image("output_portra400.jpg", rgb_emulated)
```

---

## Module Structure

```
preprocessing/
├── __init__.py
├── core.py              # Main pipeline
├── analysis.py          # Histogram, clipping detection
├── exposure.py          # Exposure correction
├── white_balance.py     # WB algorithms
├── highlights.py        # Highlight recovery
├── shadows.py           # Shadow lifting
├── contrast.py          # Tone curves
├── noise.py             # Noise reduction
├── color_space.py       # Linear/sRGB conversions
├── io.py                # Load/save, RAW handling
└── config.py            # Default parameters
```

---

## Success Criteria

After preprocessing, image should have:
1. ✓ Mean brightness 0.18-0.25 (proper exposure)
2. ✓ Neutral white balance (no color cast)
3. ✓ Highlight clipping < 5%
4. ✓ Shadow crush < 5%
5. ✓ Midtones centered (median ≈ 0.5)
6. ✓ Natural contrast (not flat, not harsh)

Ready for Portra 400 film emulation!
