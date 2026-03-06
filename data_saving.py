# v2_data_saving_aglsd.py
import re, math, os
import numpy as np
from urllib.request import urlopen

def save_data_aglsd_desert_dust(wavelength_nm: int = 488, out_path: str = None):
    """
    Save (mu, F11) TXT from the Amsterdam–Granada Light Scattering Database (AGLSD)
    average desert-dust phase function at 488 nm or 647 nm.
    - mu = cos(theta_deg)
    - Output: two columns: mu \t F11_norm

    Returns: out_path
    """
    urls = {
        488: "https://old-scattering.iaa.csic.es/site_media/download/matrix_averagedesert_488nm.txt",
        647: "https://old-scattering.iaa.csic.es/site_media/download/matrix_averagedesert_647nm.txt",
    }
    if wavelength_nm not in urls:
        raise ValueError("wavelength_nm must be 488 or 647")
    url = urls[wavelength_nm]

    # download text
    with urlopen(url) as r:
        text = r.read().decode("utf-8", errors="ignore")

    # pull all numeric tokens (the files put the entire table on one line)
    # per-row numeric stride in AGLSD 'matrix_averagedesert_*nm.txt' is 13:
    # [theta, F11, span, -F12/F11, span, F22/F11, span, F33/F11, span, F34/F11, span, F44/F11, span]
    nums = [float(m.group(0)) for m in re.finditer(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", text)]
    stride = 13
    if len(nums) < stride:
        raise ValueError("Unexpected file format — not enough numeric tokens.")

    thetas = np.array(nums[0::stride], dtype=float)
    f11    = np.array(nums[1::stride], dtype=float)

    # keep only physical theta in [0,180]
    mask = (thetas >= 0.0) & (thetas <= 180.0) & np.isfinite(f11)
    thetas = thetas[mask]; f11 = f11[mask]

    mus = np.cos(np.deg2rad(thetas))

    # sort by mu and drop duplicate mu
    order = np.argsort(mus)
    mus = mus[order]; f11 = f11[order]
    uniq = np.r_[True, np.diff(mus) != 0.0]
    mus = mus[uniq]; f11 = f11[uniq]

    # write two-column TXT
    if out_path is None:
        out_path = f"./aglsd_desert_dust_{wavelength_nm}nm_mu.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for mu, v in zip(mus, f11):
            f.write(f"{mu:.10f}\t{v:.8e}\n")

    dmu = (mus[-1] - mus[0]) / (len(mus) - 1) if len(mus) > 1 else float("nan")
    print(f"[data|AGLSD desert dust {wavelength_nm} nm] wrote {len(mus)} pts → {out_path}")
    print(f"[data|AGLSD] μ∈[{mus[0]:.3f},{mus[-1]:.3f}], Δμ≈{dmu:.6f}")
    return out_path


txt_path = save_data_aglsd_desert_dust(488, "./data/desert_dust_488_mu.txt")
txt_path = save_data_aglsd_desert_dust(647, "./data/desert_dust_647_mu.txt")