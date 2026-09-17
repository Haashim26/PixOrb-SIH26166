"""SIH26166 Stage 2: multi-resolution coarse scale/rotation/translation alignment."""
import cv2
import numpy as np


def pyramid(img, levels=5):
    out = [img]
    for _ in range(1, levels):
        if min(out[-1].shape) < 128:
            break
        out.append(cv2.pyrDown(out[-1]))
    return out


def _fft_logmag(img):
    x = img.astype(np.float32)
    win = cv2.createHanningWindow((x.shape[1], x.shape[0]), cv2.CV_32F)
    f = np.fft.fftshift(np.fft.fft2(x * win))
    return np.log1p(np.abs(f)).astype(np.float32)


def _common_canvas(ref, src):
    h = max(ref.shape[0], src.shape[0])
    w = max(ref.shape[1], src.shape[1])

    def pad(x):
        y = np.zeros((h, w), np.uint8)
        yy = (h - x.shape[0]) // 2
        xx = (w - x.shape[1]) // 2
        y[yy:yy + x.shape[0], xx:xx + x.shape[1]] = x
        return y

    return pad(ref), pad(src)


def estimate_similarity(ref, src):
    """Estimate rotation+scale using Fourier-Mellin/log-polar phase correlation."""
    ref, src = _common_canvas(ref, src)
    h, w = ref.shape
    size = max(h, w)
    ds = min(1.0, 1024.0 / size)
    if ds < 1.0:
        ref = cv2.resize(ref, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        src = cv2.resize(src, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

    h, w = ref.shape
    center = (w / 2.0, h / 2.0)
    radius = max(2.0, min(center))
    a, b = _fft_logmag(ref), _fft_logmag(src)
    pa = cv2.warpPolar(a, (w, h), center, radius, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    pb = cv2.warpPolar(b, (w, h), center, radius, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), conf = cv2.phaseCorrelate(pb, pa, win)

    # warpPolar maps log(radius) linearly along x.
    log_base = max(w / np.log(max(radius, 2.0)), 1e-6)
    scale_est = float(np.exp(dx / log_base))
    angle_est = float(-dy * 360.0 / h)
    if not np.isfinite(scale_est) or not (0.02 <= scale_est <= 50.0):
        scale_est = 1.0
    if not np.isfinite(angle_est):
        angle_est = 0.0
    return scale_est, angle_est, float(conf)


def _center_transform(shape, scale, angle):
    h, w = shape
    # Maps source -> reference canvas around source centre.
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0 / scale)
    return np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)


def _translation(ref, aligned):
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), conf = cv2.phaseCorrelate(aligned.astype(np.float32), ref.astype(np.float32), win)
    return float(dx), float(dy), float(conf)


def coarse_align(reference, source, levels=5):
    """Return source warped to reference canvas and a source->reference 3x3 transform.

    Scale/rotation hypotheses are evaluated on the usable levels of a Gaussian
    pyramid; the highest-confidence hypothesis is selected. Translation is then
    refined by phase correlation on the resulting full-resolution alignment.
    """
    ref = reference.astype(np.uint8)
    src = source.astype(np.uint8)
    ref_pyr = pyramid(ref, levels)
    src_pyr = pyramid(src, levels)
    n = min(len(ref_pyr), len(src_pyr))

    hypotheses = []
    for level in range(n):
        # Extremely small levels are dominated by interpolation and lunar texture
        # loss; retain the useful levels for scale/rotation estimation.
        if min(ref_pyr[level].shape) < 160:
            continue
        sc, ang, conf = estimate_similarity(ref_pyr[level], src_pyr[level])
        hypotheses.append((float(conf), sc, ang, level))
    if not hypotheses:
        sc, ang, conf = estimate_similarity(ref, src)
        chosen_level = 0
    else:
        conf, sc, ang, chosen_level = max(hypotheses, key=lambda x: x[0])

    Hs = _center_transform(src.shape, sc, ang)
    warped = cv2.warpPerspective(src, Hs, (ref.shape[1], ref.shape[0]))
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), tconf = cv2.phaseCorrelate(warped.astype(np.float32), ref.astype(np.float32), win)
    Ht = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    H = Ht @ Hs
    aligned = cv2.warpPerspective(src, H, (ref.shape[1], ref.shape[0]))
    return aligned, H, {
        "scale": float(sc),
        "rotation_deg": float(ang),
        "scale_confidence": float(conf),
        "translation_dx": float(dx),
        "translation_dy": float(dy),
        "translation_confidence": float(tconf),
        "pyramid_levels": int(n),
        "estimation_level": int(chosen_level),
        "evaluated_hypotheses": int(len(hypotheses)),
    }
