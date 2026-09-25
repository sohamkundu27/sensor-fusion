#!/usr/bin/env python3
"""Same fixed camera, three CARLA weather presets (clear / mid fog / max fog).

Uses the world the server already booted. Does not call load_world.
Expects: ./CarlaUE4.sh -RenderOffScreen -nosound
"""
import json
import queue
import subprocess
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image

OUT = Path("sim/carla_smoketest/outputs/weather_fog")
PRESETS = (
    ("clear", dict(fog_density=0.0, fog_distance=0.0, fog_falloff=0.2,
                   scattering_intensity=0.0, mie_scattering_scale=0.0,
                   rayleigh_scattering_scale=0.0331, sun_altitude_angle=45.0,
                   cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
                   wetness=0.0, wind_intensity=0.0)),
    ("fog_mid", dict(fog_density=50.0, fog_distance=0.0, fog_falloff=0.2,
                     scattering_intensity=1.0, mie_scattering_scale=1.0,
                     rayleigh_scattering_scale=0.0331, sun_altitude_angle=45.0,
                     cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0,
                     wetness=0.0, wind_intensity=0.0)),
    ("fog_max", dict(fog_density=100.0, fog_distance=0.0, fog_falloff=0.2,
                     scattering_intensity=1.0, mie_scattering_scale=1.0,
                     rayleigh_scattering_scale=0.0331, sun_altitude_angle=45.0,
                     cloudiness=40.0, precipitation=0.0, precipitation_deposits=0.0,
                     wetness=0.0, wind_intensity=0.0)),
)


def gpu():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().split(", ")
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    procs = [ln.strip(" |") for ln in table.split("Processes:", 1)[-1].splitlines() if "MiB" in ln]
    return {"used_mib": int(q[0]), "util_pct": int(q[1]), "process_rows": procs}


def weather_dict(w):
    keys = (
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
        "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
        "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm",
    )
    return {k: getattr(w, k) for k in keys if hasattr(w, k)}


def apply_weather(world, params):
    w = world.get_weather()
    for k, v in params.items():
        setattr(w, k, v)
    world.set_weather(w)
    return weather_dict(world.get_weather())


def grab(images, frame, timeout=10.0):
    img = images.get(timeout=timeout)
    while img.frame < frame:
        img = images.get(timeout=timeout)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    original = world.get_settings()
    original_weather = world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    spawn = world.get_map().get_spawn_points()[0]
    cam_tf = carla.Transform(
        spawn.location + carla.Location(z=2.2),
        spawn.rotation,
    )
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "1280")
    bp.set_attribute("image_size_y", "720")
    bp.set_attribute("fov", "90")
    camera = world.spawn_actor(bp, cam_tf)
    images = queue.Queue()
    camera.listen(images.put)
    world.tick()

    res = {
        "map": world.get_map().name,
        "camera_tf": {"x": cam_tf.location.x, "y": cam_tf.location.y,
                      "z": cam_tf.location.z, "yaw": cam_tf.rotation.yaw,
                      "pitch": cam_tf.rotation.pitch},
        "gpu_connect": gpu(),
        "presets": [],
    }

    try:
        for name, params in PRESETS:
            applied = apply_weather(world, params)
            last = None
            for _ in range(20):
                last = grab(images, world.tick())
            bgra = np.frombuffer(last.raw_data, dtype=np.uint8).reshape(last.height, last.width, 4)
            rgb = bgra[:, :, [2, 1, 0]].copy()
            png = OUT / f"{name}.png"
            Image.fromarray(rgb).save(png)
            res["presets"].append({
                "name": name,
                "requested": params,
                "applied": applied,
                "path": str(png),
                "frame": last.frame,
                "mean_rgb": [round(float(x), 2) for x in rgb.mean(axis=(0, 1))],
                "std_rgb": [round(float(x), 2) for x in rgb.std(axis=(0, 1))],
                "frac_pixels_all_zero": float((rgb.reshape(-1, 3).sum(axis=1) == 0).mean()),
                "unique_colors": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]),
                "gpu": gpu(),
            })
    finally:
        camera.stop()
        camera.destroy()
        world.set_weather(original_weather)
        world.apply_settings(original)
        res["gpu_after"] = gpu()
        (OUT / "result.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
