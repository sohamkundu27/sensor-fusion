#!/usr/bin/env python3
"""CARLA 0.9.16 simulator + Python API smoke test (no mesh export).

Expects a server that is already running, e.g.
    ./CarlaUE4.sh -RenderOffScreen -nosound

Loads the default town, switches to synchronous mode with a fixed time step,
spawns a few autopilot vehicles and a chase camera, ticks N times, and saves one
RGB frame plus a JSON summary.
"""
import argparse
import json
import queue
import subprocess
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image


def gpu_snapshot():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().split(", ")
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    procs = [ln.strip(" |") for ln in table.split("Processes:", 1)[-1].splitlines() if "MiB" in ln]
    return {"used_mib": int(q[0]), "total_mib": int(q[1]), "util_pct": int(q[2]),
            "t": time.time(), "process_rows": procs}


def attr_value(a):
    t = carla.ActorAttributeType
    if a.type == t.Bool:
        return a.as_bool()
    if a.type == t.Int:
        return a.as_int()
    if a.type == t.Float:
        return a.as_float()
    if a.type == t.RGBColor:
        c = a.as_color()
        return [c.r, c.g, c.b, c.a]
    return a.as_str()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default=None, help="default: whatever map the server booted with")
    ap.add_argument("--no-reload", action="store_true",
                    help="use the world the server booted with instead of calling load_world")
    ap.add_argument("--vehicles", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim/carla_smoketest/outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"args": vars(args), "gpu": {}, "timings_s": {}, "errors": []}

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    res["client_version"] = client.get_client_version()
    res["server_version"] = client.get_server_version()
    res["gpu"]["connected"] = gpu_snapshot()

    boot_map = client.get_world().get_map().name
    res["boot_map"] = boot_map
    res["available_maps"] = sorted(client.get_available_maps())
    town = args.town or boot_map.split("/")[-1]

    t0 = time.time()
    world = client.get_world() if args.no_reload else client.load_world(town)
    res["timings_s"]["get_world" if args.no_reload else "load_world"] = time.time() - t0
    res["loaded_map"] = world.get_map().name
    res["gpu"]["after_load_world"] = gpu_snapshot()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)
    applied = world.get_settings()
    res["applied_settings"] = {"synchronous_mode": applied.synchronous_mode,
                               "fixed_delta_seconds": applied.fixed_delta_seconds,
                               "no_rendering_mode": applied.no_rendering_mode}

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    actors = []
    camera = None
    try:
        rng = np.random.default_rng(args.seed)
        bp_lib = world.get_blueprint_library()
        vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*")
                       if int(bp.get_attribute("number_of_wheels")) == 4]
        spawn_points = world.get_map().get_spawn_points()
        res["n_spawn_points"] = len(spawn_points)
        res["n_vehicle_blueprints_4wheel"] = len(vehicle_bps)
        order = rng.permutation(len(spawn_points))

        for idx in order:
            if len(actors) >= args.vehicles:
                break
            bp = vehicle_bps[int(rng.integers(len(vehicle_bps)))]
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            v = world.try_spawn_actor(bp, spawn_points[int(idx)])
            if v is not None:
                actors.append(v)
        world.tick()
        for v in actors:
            v.set_autopilot(True, args.tm_port)
        res["spawned_vehicles"] = [{"id": v.id, "type": v.type_id} for v in actors]

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.width))
        cam_bp.set_attribute("image_size_y", str(args.height))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", "0.0")
        cam_tf = carla.Transform(carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12.0))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=actors[0])
        images = queue.Queue()
        camera.listen(images.put)
        res["camera"] = {"id": camera.id, "parent": actors[0].id,
                         "attributes": {a.id: attr_value(a) for a in cam_bp}}

        world.tick()
        start_locs = {v.id: v.get_location() for v in actors}
        start_snapshot = world.get_snapshot()

        tick_wall = []
        step_wall = []
        frames_matched = 0
        frames_mismatched = 0
        missing = 0
        kept = None
        for i in range(args.ticks):
            t = time.time()
            frame = world.tick()
            tick_wall.append(time.time() - t)
            try:
                img = images.get(timeout=10.0)
                while img.frame < frame:
                    img = images.get(timeout=10.0)
                if img.frame == frame:
                    frames_matched += 1
                else:
                    frames_mismatched += 1
                kept = img
            except queue.Empty:
                missing += 1
            step_wall.append(time.time() - t)
            if i + 1 == args.ticks // 2:
                res["gpu"]["mid_ticks"] = gpu_snapshot()
        res["gpu"]["end_ticks"] = gpu_snapshot()

        end_snapshot = world.get_snapshot()
        moved = {str(v.id): round(v.get_location().distance(start_locs[v.id]), 3) for v in actors}
        res["ticking"] = {
            "ticks": args.ticks,
            "frame_start": start_snapshot.frame,
            "frame_end": end_snapshot.frame,
            "sim_time_start_s": start_snapshot.timestamp.elapsed_seconds,
            "sim_time_end_s": end_snapshot.timestamp.elapsed_seconds,
            "sim_time_advanced_s": end_snapshot.timestamp.elapsed_seconds - start_snapshot.timestamp.elapsed_seconds,
            "tick_wall_s_median": float(np.median(tick_wall)),
            "tick_wall_s_mean": float(np.mean(tick_wall)),
            "tick_wall_s_max": float(np.max(tick_wall)),
            "tick_plus_image_wall_s_median": float(np.median(step_wall)),
            "tick_plus_image_wall_s_mean": float(np.mean(step_wall)),
            "tick_plus_image_wall_s_total": float(np.sum(step_wall)),
            "camera_frames_matched": frames_matched,
            "camera_frames_mismatched": frames_mismatched,
            "camera_frames_missing": missing,
            "vehicle_displacement_m": moved,
        }

        if kept is None:
            res["errors"].append("no camera image received")
        else:
            bgra = np.frombuffer(kept.raw_data, dtype=np.uint8).reshape(kept.height, kept.width, 4)
            rgb = bgra[:, :, [2, 1, 0]].copy()
            png = out / f"rgb_frame_{kept.frame}.png"
            Image.fromarray(rgb).save(png)
            flat = rgb.reshape(-1, 3)
            res["image"] = {
                "path": str(png),
                "frame": kept.frame,
                "sim_timestamp_s": kept.timestamp,
                "width": kept.width,
                "height": kept.height,
                "raw_bytes": len(kept.raw_data),
                "mean_rgb": [round(float(x), 2) for x in rgb.mean(axis=(0, 1))],
                "std_rgb": [round(float(x), 2) for x in rgb.std(axis=(0, 1))],
                "min": int(rgb.min()),
                "max": int(rgb.max()),
                "frac_pixels_all_zero": float((flat.sum(axis=1) == 0).mean()),
                "unique_colors": int(np.unique(flat, axis=0).shape[0]),
                "alpha_unique": np.unique(bgra[:, :, 3]).tolist()[:10],
            }
    except Exception as e:
        res["errors"].append(f"{type(e).__name__}: {e}")
        raise
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in actors], True)
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)
        res["gpu"]["after_cleanup"] = gpu_snapshot()
        (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k != "available_maps"}, indent=2, default=str))


if __name__ == "__main__":
    main()
