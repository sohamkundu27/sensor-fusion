#!/usr/bin/env python3
"""mmengine-API twin of bench_common.py (mmcv>=2 / mmdet3d>=1.x, e.g. SAMFusion).
Same measurements and result.json schema: 5 warmup + N timed iterations at
batch 1; wall-clock median (incl. data loading) as the headline, compute-only
and data-wait fraction alongside; peak torch allocated/reserved and
device-level used memory. Timing only -- no accuracy, no checkpoints."""
import argparse, json, time, traceback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--iters", type=int, default=30); ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ann-file", default=None, help="override train ann_file")
    ap.add_argument("--unwrap-cbgs", action="store_true")
    ap.add_argument("--drop-steps", default="")
    ap.add_argument("--mini-iters", type=int, default=323); ap.add_argument("--full-iters", type=int, default=28130)
    a = ap.parse_args()
    import torch
    res = {"method": a.method, "config": a.config, "ran": "N", "iters_requested": a.iters, "batch_size": 1}
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner
        from mmengine.utils import import_modules_from_strings
        from mmdet3d.registry import MODELS
        cfg = Config.fromfile(a.config)
        if cfg.get("custom_imports"):
            import_modules_from_strings(**cfg.custom_imports)
        init_default_scope("mmdet3d")
        dl = cfg.train_dataloader
        dl.batch_size = 1; dl.num_workers = 2; dl.persistent_workers = False
        ds = dl.dataset
        if a.unwrap_cbgs and ds.get("type") == "CBGSDataset":
            ds = ds.dataset; dl.dataset = ds
        if a.ann_file:
            ds.ann_file = a.ann_file
        drop = {d.strip() for d in a.drop_steps.split(",") if d.strip()}
        if drop:
            ds.pipeline = [st for st in ds.pipeline if st.get("type") not in drop]
            res["dropped_pipeline_steps"] = sorted(drop)
        loader = Runner.build_dataloader(dl)
        res["dataset_len"] = len(loader.dataset)
        model = MODELS.build(cfg.model).cuda(); model.init_weights(); model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        torch.cuda.reset_peak_memory_stats()
        times, walls = [], []; it = iter(loader); tw = time.perf_counter()
        for i in range(a.warmup + a.iters):
            try: batch = next(it)
            except StopIteration: it = iter(loader); batch = next(it)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            data = model.data_preprocessor(batch, True)
            losses = model(**data, mode="loss")
            loss, _ = model.parse_losses(losses)
            opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
            now = time.perf_counter()
            if i >= a.warmup: times.append(now - t0); walls.append(now - tw)
            tw = now
        free_b, total_b = torch.cuda.mem_get_info()
        times.sort(); walls.sort(); med = walls[len(walls)//2]; mc = times[len(times)//2]
        res.update(ran="Y", sec_per_iter=round(med, 4), sec_per_iter_mean=round(sum(walls)/len(walls), 4),
                   sec_per_iter_compute_only=round(mc, 4), data_wait_fraction=round(1 - mc/med, 3),
                   vram_gb=round(torch.cuda.max_memory_allocated()/1024**3, 3),
                   vram_reserved_gb=round(torch.cuda.max_memory_reserved()/1024**3, 3),
                   device_used_gb=round((total_b - free_b)/1024**3, 3), device_total_gb=round(total_b/1024**3, 3),
                   iters_timed=len(times), hrs_epoch_mini=round(med*a.mini_iters/3600, 3),
                   hrs_epoch_full=round(med*a.full_iters/3600, 2))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        oom = "out of memory" in msg.lower() or "error 2" in msg.lower()
        res.update(ran="N (OOM)" if oom else "N", oom=bool(oom),
                   blocker=(("Out of VRAM at batch size 1: " if oom else "") + msg)[:400],
                   frames=[f"{f.filename.split('/repo/')[-1]}:{f.lineno} {f.name}: {f.line}"
                           for f in traceback.extract_tb(e.__traceback__)][-8:],
                   traceback=traceback.format_exc()[-2000:])
        try:
            res["vram_gb_at_failure"] = round(torch.cuda.max_memory_allocated()/1024**3, 3)
        except Exception: pass
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=2))

if __name__ == "__main__":
    main()
