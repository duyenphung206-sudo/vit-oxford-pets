# Kỳ — verified E4 CPU demo

DONE: the group leader's real E4 seed44 checkpoint is available in release
[v0.9.0](https://github.com/duyenphung206-sudo/vit-oxford-pets/releases/tag/v0.9.0).
The user supplied the asset link on 2026-11-21. The earlier v1.0.0 reference was unavailable;
no personal-model checkpoint was substituted.

- Asset: `vit_b16_e4_seed44_best.pt`, 343,359,029 bytes.
- SHA256: `525435b0b0338e379fad8238906687a636c4a8ca4bf39539d67f2b1901d9ec77`.
- Downloaded bytes match the release asset digest exactly.
- Local weights: `C:/Users/Ken/Downloads/.ky-tools/checkpoints/vit_b16_e4_seed44_best.pt`.
- Raw state_dict loaded with `weights_only=True`, strict loading, CPU, ViT-B/16 and 37 classes.
- Labels come from the group's persisted `data/splits/train.csv`, verified against all
  official trainval label IDs. They are not sorted independently from model outputs.

The demo loader supports both raw group checkpoints and metadata bundles with the same
architecture/class order. It rejects smoke-only weights and mismatched class metadata.
Only `app/app.py` changed; the shared trainer and the existing UI were preserved.

Actual CPU inference and browser upload passed for `Abyssinian_100` and `beagle_100`
(both official trainval images). The UI displays five breeds and softmax scores.
Empty input displays `Upload a valid image`. These two illustrations are not an
accuracy estimate. No test-set evaluation or model selection was performed.

Evidence: `docs/KY_DEMO_VERIFICATION.json`; regression checks: `tests/test_ky_demo.py`.
After the user supplied the exact report PDF, figure3.11 required attention.
An optional `--attention` mode now adds real attention rollout while retaining the original
Top-5 default. Q/K are reconstructed from LayerNorm1 and existing projections; the model
forward and shared code remain unchanged. See `docs/KY_ATTENTION_VERIFICATION.json`.
Browser screenshots for figures3.10/3.11 use the first official test annotation, selected
before inference, as a single illustration; no aggregate test evaluation/tuning was done.

Run from repository root after downloading the verified release asset:

```powershell
python app/app.py --checkpoint <path-to-vit_b16_e4_seed44_best.pt> --device cpu --port 7860
```

Open http://127.0.0.1:7860. Checkpoint binaries are ignored and must not be committed.

Report mode: append `--attention` to the CLI (the verified live instance uses port7861).
