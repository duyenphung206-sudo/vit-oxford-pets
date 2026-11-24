"""Architecture diagrams for Ky's support module; report numbering requires latest captions."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def diagram(path, title, nodes, footer):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set(xlim=(0, 13), ylim=(0, 4.5))
    ax.axis('off')
    ax.text(6.5, 4.1, title, ha='center', fontsize=18, weight='bold', color='#17334b')
    width = 1.62
    for index, (heading, detail) in enumerate(nodes):
        x = 0.25 + index * 1.84
        box = FancyBboxPatch((x, 1.55), width, 1.55, boxstyle='round,pad=0.04',
                             facecolor='#edf5fc', edgecolor='#39729d', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width / 2, 2.65, heading, ha='center', va='center', fontsize=10.5, weight='bold')
        ax.text(x + width / 2, 2.12, detail, ha='center', va='center', fontsize=10, linespacing=1.6)
        if index < len(nodes) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.03, 2.3), (x + 1.81, 2.3),
                                        arrowstyle='-|>', mutation_scale=14, color='#39729d'))
    ax.text(6.5, 0.75, footer, ha='center', va='center', fontsize=11, linespacing=1.6, color='#344f62')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(path.with_suffix('.'+ext), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['svg.hashsalt'] = 'ky-independent-vit-20261124'
    root = Path('figures/ky_support')
    diagram(root/'vit_architecture', 'Independent ViT-B/16 module', [
        ('Input', 'RGB image\n224 x 224'),
        ('Patch\nembedding', 'Conv2D 16 / 16\n196 x 768'),
        ('CLS +\nposition', '197 tokens\ndimension 768'),
        ('Encoder x 12', 'Pre-LN + MHSA\nResidual + MLP'),
        ('Final\nLayerNorm', '197 x 768\nepsilon 1e-6'),
        ('CLS\nselection', 'Select token 0\n768 features'),
        ('Linear head', '768 -> 37\nclass logits'),
    ], 'Per block: 12 attention heads, head dimension 64; MLP 768 -> 3072 -> 768, GELU.\n'
       'Architecture implementation does not imply training from random initialization.')
    diagram(root/'attention_flow', 'Independent self-attention: Q/K/V and 12 heads', [
        ('Input tokens', 'N = 197\nD = 768'),
        ('Q / K / V', 'One Linear\n768 -> 2304'),
        ('Split heads', 'H = 12\nd = 64'),
        ('Attention', 'softmax(QK^T\n/ sqrt(64))'),
        ('Apply to V', 'A @ V\nper head'),
        ('Concatenate', '197 x 768\n12 heads joined'),
        ('Projection', 'Linear 768\n-> 768'),
    ], 'Encoder block: x1 = x + Attention(LayerNorm(x));\n'
       'output = x1 + MLP(LayerNorm(x1)). Dropout = 0 for this matched architecture.')


if __name__ == '__main__':
    main()
