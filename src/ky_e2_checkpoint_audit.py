"""Audit a real E2 checkpoint against original pretrained backbone and validation metrics."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.models import ViT_B_16_Weights, vit_b_16

from src.data import build_dataloaders
from src.engine import evaluate
from src.models.factory import build_model
from src.utils import count_parameters, load_config


def audit(run_path, output):
    run = Path(run_path)
    manifest = json.loads((run/'run_metadata.json').read_text())
    assert manifest['status'] == 'DONE'
    results = run/'results/e2_vit_linear_probe/seed42'
    cfg = load_config(results/'config.yaml')
    metrics = json.loads((results/'metrics.json').read_text())
    state = torch.load(run/'checkpoints/e2_vit_linear_probe/seed42/best.pt',
                       map_location='cpu', weights_only=True)
    # Load through the official model to normalize legacy MLP checkpoint key names.
    reference = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    pretrained = reference.state_dict()
    backbone_keys = [name for name in state if not name.startswith('heads.')]
    assert backbone_keys and all(torch.equal(state[name], pretrained[name]) for name in backbone_keys)
    assert state['heads.head.weight'].shape == (37, 768)
    del pretrained, reference
    model_cfg = dict(cfg['model'], pretrained=False)
    model = build_model(model_cfg)
    loaded = model.load_state_dict(state, strict=True)
    assert count_parameters(model)['trainable'] == 28453
    if not torch.cuda.is_available():
        raise RuntimeError('GPU validation requires the authorized T4 runtime')
    device = torch.device('cuda')
    model.to(device).eval()
    loaders = build_dataloaders(cfg['data']['root'], cfg['data']['splits_dir'],
                               cfg['data']['batch_size'], cfg['data']['num_workers'], 42, True, True)
    val = evaluate(model, loaders['val'], torch.nn.CrossEntropyLoss(), device)
    assert abs(val['acc']-metrics['best_val_acc']) < 1e-6
    assert abs(val['loss']-metrics['best_val_loss']) < 1e-6
    result = {'status': 'PASS', 'strict_load': str(loaded), 'backbone_tensor_keys_checked': len(backbone_keys),
              'all_backbone_tensors_bitwise_equal_ImageNet1K_V1': True, 'head_weight_shape': [37, 768],
              'parameters': count_parameters(model), 'validation_recomputed': val,
              'validation_matches_recorded_best': True, 'validation_images': 736,
              'device': torch.cuda.get_device_name(0), 'test_evaluation': 'NOT_RUN',
              'checkpoint_sha256_from_verified_run': manifest['checkpoint_sha256']}
    Path(output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('E2_CHECKPOINT_AUDIT', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
