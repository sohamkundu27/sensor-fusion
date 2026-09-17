#!/usr/bin/env python3
"""Shared timing harness: builds a model+dataset from an mmdet3d/mmcv config,
runs N train iterations at batch size 1, records peak VRAM and sec/iter.

Timing only. No accuracy, no convergence, no checkpointing.
"""
import argparse, json, os, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--loader", default="mmcv", choices=["mmcv", "torchpack"])
    ap.add_argument("--drop-steps", default="",
                    help="Comma-separated pipeline step types to remove (e.g. ObjectPaste), "
                         "for steps whose prebuilt databases are unavailable.")
    ap.add_argument("--unwrap-cbgs", action="store_true",
                    help="Use the inner dataset instead of the CBGS resampling wrapper, "
                         "so one epoch equals the true keyframe count.")
    ap.add_argument("--mini-iters", type=int, default=323)
    ap.add_argument("--full-iters", type=int, default=28130)
    args = ap.parse_args()

    import torch
    res = {"method": args.method, "config": args.config, "ran": "N",
           "iters_requested": args.iters, "batch_size": 1}
    try:
        from mmcv import Config
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model
        from mmcv.parallel import MMDataParallel
        from mmdet3d.datasets import build_dataloader

        if args.loader == "torchpack":
            from torchpack.utils.config import configs as tp_configs
            from mmdet3d.utils import recursive_eval
            tp_configs.load(args.config, recursive=True)
            cfg = Config(recursive_eval(tp_configs), filename=args.config)
        else:
            cfg = Config.fromfile(args.config)
        cfg.data.samples_per_gpu = 1
        cfg.data.workers_per_gpu = 2

        train_cfg_data = cfg.data.train
        if args.unwrap_cbgs and "dataset" in train_cfg_data:
            train_cfg_data = train_cfg_data["dataset"]
        drop = {d.strip() for d in args.drop_steps.split(",") if d.strip()}
        if drop and "pipeline" in train_cfg_data:
            kept = [st for st in train_cfg_data["pipeline"]
                    if st.get("type") not in drop]
            res["dropped_pipeline_steps"] = sorted(drop)
            train_cfg_data["pipeline"] = kept
        dataset = build_dataset(train_cfg_data)
        res["dataset_len"] = len(dataset)
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                                  num_gpus=1, dist=False, shuffle=False, seed=0)
        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                            test_cfg=cfg.get("test_cfg"))
        model.init_weights()
        model = MMDataParallel(model.cuda(), device_ids=[0])
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

        torch.cuda.reset_peak_memory_stats()
        times = []
        it = iter(loader)
        for i in range(args.warmup + args.iters):
            try:
                data = next(it)
            except StopIteration:
                it = iter(loader); data = next(it)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            losses = model(return_loss=True, **data)
            loss = sum(v.mean() for k, v in losses.items() if "loss" in k and hasattr(v, "mean"))
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                times.append(dt)
        times.sort()
        med = times[len(times)//2]
        res.update(ran="Y",
                   sec_per_iter=round(med, 4),
                   sec_per_iter_mean=round(sum(times)/len(times), 4),
                   vram_gb=round(torch.cuda.max_memory_allocated()/1024**3, 3),
                   vram_reserved_gb=round(torch.cuda.max_memory_reserved()/1024**3, 3),
                   iters_timed=len(times),
                   hrs_epoch_mini=round(med*args.mini_iters/3600, 3),
                   hrs_epoch_full=round(med*args.full_iters/3600, 2))
    except Exception as e:
        import traceback
        msg = f"{type(e).__name__}: {e}"
        oom = ("out of memory" in msg.lower()
               or "cuda execution failed with error 2" in msg.lower()
               or "CUDA out of memory" in msg)
        res.update(ran="N (OOM)" if oom else "N",
                   oom=bool(oom),
                   blocker=("Out of VRAM at batch size 1: " + msg)[:400] if oom
                           else msg[:400],
                   traceback=traceback.format_exc()[-2000:])
        try:
            import torch as _t
            res["vram_gb_at_failure"] = round(_t.cuda.max_memory_allocated()/1024**3, 3)
            res["vram_reserved_gb_at_failure"] = round(_t.cuda.max_memory_reserved()/1024**3, 3)
        except Exception:
            pass
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=2))

if __name__ == "__main__":
    main()
