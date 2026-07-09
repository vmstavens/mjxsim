"""Local rough-terrain assets for the Spot RMA experiment."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


ROUGH_SPOT_XML = """
<mujoco model="spot rma rough terrain scene">
  <include file="spot_mjx_feetonly.xml"/>

  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>

  <visual>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <hfield name="hfield" file="hfield.png" size="10 10 .08 1.0"/>
  </asset>

  <worldbody>
    <geom name="floor" type="hfield" hfield="hfield" contype="1"
      conaffinity="0" priority="1" friction="1.0"/>
  </worldbody>

  <include file="sensor.xml"/>

  <keyframe>
    <key name="home" qpos="
    0 0 0.6
    1 0 0 0
    0.0 0.9 -1.8
    0.0 0.9 -1.8
    0.0 0.9 -1.8
    0.0 0.9 -1.8"
      ctrl="0.0 0.9 -1.8 0.0 0.9 -1.8 0.0 0.9 -1.8 0.0 0.9 -1.8"/>
  </keyframe>
</mujoco>
""".strip()


def make_hfield_png(size: int = 128, seed: int = 7) -> bytes:
    """Creates a deterministic grayscale hfield image."""
    rng = np.random.default_rng(seed)
    height = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32)

    for _ in range(5):
        height = (
            height
            + np.roll(height, 1, axis=0)
            + np.roll(height, -1, axis=0)
            + np.roll(height, 1, axis=1)
            + np.roll(height, -1, axis=1)
        ) / 5.0

    height -= height.min()
    height /= max(float(height.max()), 1e-6)
    image = Image.fromarray((height * 255).astype(np.uint8), mode="L")
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
