"""Render the smoke-test scene once and log wall-clock time and peak memory.

One render per process, so the memory peaks are per render:
  python render.py --sigma-t 0.0  --out outputs/clear.npz
  python render.py --sigma-t 0.15 --out outputs/fog.npz
"""
import argparse
import json
import os
import resource
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


class GpuMemPoller:
    """Samples NVML every `interval` s; tracks this process's peak GPU memory
    and the device-wide peak. NVML's device 'used' includes driver-reserved
    memory, so it reads higher than nvidia-smi; the per-process number is the
    one to quote."""

    def __init__(self, pid, interval=0.01):
        self.pid, self.interval = pid, interval
        self.peak_proc = 0
        self.peak_dev = 0
        self.baseline_dev = None
        self.samples = 0
        self._stop = threading.Event()
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.baseline_dev = pynvml.nvmlDeviceGetMemoryInfo(self.h).used
            self.ok = True
        except Exception as e:  # CPU-only machine or no NVML
            self.err = repr(e)

    def _run(self):
        while not self._stop.is_set():
            try:
                used = self.nv.nvmlDeviceGetMemoryInfo(self.h).used
                self.peak_dev = max(self.peak_dev, used)
                for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h):
                    if p.pid == self.pid and p.usedGpuMemory:
                        self.peak_proc = max(self.peak_proc, p.usedGpuMemory)
                self.samples += 1
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if self.ok:
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()

    def stop(self):
        if self.ok:
            self._stop.set()
            self.t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(HERE / "scene.xml"))
    ap.add_argument("--variant", default="cuda_ad_mono")
    ap.add_argument("--sigma-t", type=float, required=True)
    ap.add_argument("--albedo", type=float, default=0.95)
    ap.add_argument("--g", type=float, default=0.85)
    ap.add_argument("--spp", type=int, default=4096)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2,
                    help="render N times in-process; the first includes JIT compile")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_proc0 = time.perf_counter()
    poller = GpuMemPoller(os.getpid())
    poller.start()

    import numpy as np
    import mitsuba as mi
    import drjit as dr
    mi.set_variant(args.variant)
    import mitransient  # noqa: F401  (registers plugins)
    t_import = time.perf_counter() - t_proc0

    t0 = time.perf_counter()
    scene = mi.load_file(args.scene, sigma_t=args.sigma_t, albedo=args.albedo,
                         g=args.g, spp=args.spp, max_depth=args.max_depth)
    dr.sync_thread()
    t_load = time.perf_counter() - t0

    integrator = scene.integrator()
    sensor = scene.sensors()[0]
    render_times = []
    steady_np = transient_np = None
    for i in range(args.repeats):
        t0 = time.perf_counter()
        steady, transient = integrator.render(scene, sensor, seed=args.seed + i, spp=args.spp)
        dr.eval(steady, transient)
        dr.sync_thread()
        render_times.append(time.perf_counter() - t0)
        if i == 0:
            steady_np = np.array(steady)
            transient_np = np.array(transient)
        del steady, transient

    poller.stop()
    t_total = time.perf_counter() - t_proc0
    ru = resource.getrusage(resource.RUSAGE_SELF)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, steady=steady_np[..., 0], transient=transient_np[..., 0])

    import mitransient as mitr
    meta = {
        "variant": args.variant,
        "versions": {"mitsuba": mi.__version__, "drjit": dr.__version__,
                     "mitransient": mitr.__version__},
        "params": {"sigma_t": args.sigma_t, "albedo": args.albedo, "g": args.g,
                   "spp": args.spp, "max_depth": args.max_depth, "seed": args.seed},
        "shapes": {"steady": list(steady_np.shape), "transient": list(transient_np.shape)},
        "nonfinite_transient": int((~np.isfinite(transient_np)).sum()),
        "negative_transient": int((transient_np < 0).sum()),
        "time_s": {
            "python_import": t_import,
            "scene_load": t_load,
            "render_each": render_times,
            "render_cold_incl_jit": render_times[0],
            "render_warm": render_times[1] if len(render_times) > 1 else None,
            "process_total": t_total,
        },
        "memory": {
            "cpu_peak_rss_MiB": ru.ru_maxrss / 1024.0,
            "gpu_proc_peak_MiB": poller.peak_proc / 2**20 if poller.ok else None,
            "gpu_device_peak_MiB_nvml": poller.peak_dev / 2**20 if poller.ok else None,
            "gpu_device_baseline_MiB_nvml": (poller.baseline_dev / 2**20
                                             if poller.ok else None),
            "gpu_poll_samples": poller.samples,
            "gpu_poll_interval_s": poller.interval,
        },
        "cpu_time_s": {"user": ru.ru_utime, "sys": ru.ru_stime},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
