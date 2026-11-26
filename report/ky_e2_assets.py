"""Build E2 numeric handover from real logs; no training or invented metrics."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'results/e2_vit_linear_probe/seed42'
    metrics = json.loads((run/'metrics.json').read_text())
    with (run/'log.csv').open(newline='', encoding='utf-8') as stream:
        log = list(csv.DictReader(stream))
    assert metrics['fake_data'] is False and metrics['seed'] == 42
    assert len(log) == metrics['epochs_run']
    selected = log[metrics['best_epoch']-1]
    assert abs(float(selected['val_loss'])-metrics['best_val_loss']) < 5.1e-5
    rows = [
        ['Best epoch (minimum val loss)', str(metrics['best_epoch'])],
        ['Validation loss at best epoch', f"{metrics['best_val_loss']:.6f}"],
        ['Validation accuracy at best epoch (%)', f"{100*metrics['best_val_acc']:.4f}"],
        ['Mean epoch time (s)', str(metrics['mean_epoch_time_s'])],
        ['Peak GPU allocated memory (MiB)', str(metrics['peak_gpu_memory_mb'])],
        ['Train loss epoch 1 (log)', log[0]['train_loss']],
        ['Train accuracy last epoch (%) (log)', f"{100*float(log[-1]['train_acc']):.2f}"],
        ['Validation accuracy last epoch (%) (log)', f"{100*float(log[-1]['val_acc']):.2f}"],
        ['Epochs completed', str(metrics['epochs_run'])],
        ['Total parameters', f"{metrics['parameters_total']:,}"],
        ['Trainable parameters', f"{metrics['parameters_trainable']:,}"],
        ['Trainable parameters (%)', f"{100*metrics['parameters_trainable']/metrics['parameters_total']:.6f}"],
        ['GPU', metrics['device']], ['Seed', str(metrics['seed'])],
    ]
    output = root / 'report/ky_assets/e2_results_summary'
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Metric', 'Value'])
        writer.writerows(rows)
    output.with_suffix('.md').write_text(
        '# E2 linear probe seed42\n\nKỳ hỗ trợ Trinh; validation, checkpoint selected by loss.\n\n'
        '| Metric | Value |\n|---|---|\n' + ''.join('| '+' | '.join(row)+' |\n' for row in rows)
        + '\nEpoch-end accuracy comes from4decimal logs; best accuracy from6decimal metrics. '
        'Report table number awaits latest verified caption.\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=['Metric', 'Value'], colWidths=[0.73, 0.27], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor('#dae3eb')
        cell.set_text_props(ha='left')
        if row == 0:
            cell.set_facecolor('#dbeaf6')
            cell.set_text_props(weight='bold')
        elif row % 2:
            cell.set_facecolor('#f4f8fc')
    ax.set_title('E2 linear probe — seed 42, validation', fontsize=17, pad=18)
    fig.text(0.5, 0.035, 'Real Tesla T4 run; group trainer unchanged; checkpoint selected by minimum validation loss.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for extension in ['png', 'svg']:
        fig.savefig(output.with_suffix('.'+extension), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
