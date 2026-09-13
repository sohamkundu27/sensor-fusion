"""Summarize training logs and scheduled validation, with standalone learning curves."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(root):
    status = json.loads((root/'status.json').read_text())
    groups = defaultdict(list)
    path = root/'training/metrics.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live writer may not have finished the last line.
            groups[row['epoch']].append(row)
    epochs = []
    for epoch, rows in sorted(groups.items()):
        epochs.append(dict(epoch=epoch, steps=len(rows), mean_loss=statistics.mean(r['losses']['total'] for r in rows),
                           mean_seconds=statistics.mean(r['seconds'] for r in rows),
                           empty_target_fraction=statistics.mean(r['losses']['positive_anchors']==0 for r in rows),
                           skipped_updates=sum(r['skipped_optimizer_step'] for r in rows),
                           loss_components={k:statistics.mean(r['losses'][k] for r in rows)
                                            for k in ('logits','boxes','boxes3d','attributes')}))
    steps = sum(e['steps'] for e in epochs)
    summary = dict(state=status['state'], training_commit=status['git_commit'], started_utc=status['started_utc'],
                   finished_utc=status.get('finished_utc'), elapsed_seconds=status['elapsed_seconds'], total_steps=steps,
                   epochs=epochs, validation=status['validation'], best_epoch=status.get('best_epoch'),
                   best_mAP=status.get('best_mAP'),
                   peak_allocated_gib=max((r['peak_allocated_gib'] for rows in groups.values() for r in rows),default=0),
                   peak_reserved_gib=max((r['peak_reserved_gib'] for rows in groups.values() for r in rows),default=0))
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    rows = summary['epochs']
    axes[0].plot([r['epoch'] for r in rows], [r['mean_loss'] for r in rows], 'o-', color='#3158a6')
    axes[0].set(xlabel='Epoch', ylabel='Mean training loss', title='Mini training (six camera views)')
    for key, label, color in [('mAP', '3D mAP', '#16806a'), ('NDS', 'NDS', '#ad6630')]:
        validation = summary['validation']
        axes[1].plot([r['epoch'] for r in validation], [100*r[key] for r in validation], 'o-', label=label, color=color)
    axes[1].set(xlabel='Epoch', ylabel='Score (%)', title='Official mini-validation (81 keyframes)')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    summary = summarize(args.run)
    plot(summary, args.run/'learning_curves.png')
    print(json.dumps({k:v for k,v in summary.items() if k!='epochs'}, indent=2))


if __name__ == '__main__':
    main()
