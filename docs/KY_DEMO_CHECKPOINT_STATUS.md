# Kỳ — E4 demo checkpoint status

BLOCKED: **waiting for E4 checkpoint**.

The release endpoint for `v1.0.0` returned HTTP 404 during this session. No group E4
checkpoint exists in this checkout. E4 metrics reference a remote Google Drive path,
which is not a local checkpoint.

The local `Downloads/ViT-B16-SOURCE-CODE/models/best.pt` has SHA256
`4c897b64829284a09bd9e85ba2984b97683cd074cb5711086bd698c4158331cb`.
Its embedded recipe uses 15 epochs, batch 16, backbone LR 3e-5, head LR 3e-4,
weight decay 0.05 and label smoothing 0.1. It is a personal-model checkpoint and
must not be presented as the group's E4 model.

Existing `app/app.py` already provides CPU inference, 37 classes, Top-5 softmax scores
and invalid-image handling. Its interface is preserved. Actual group E4 inference
and browser upload verification remain pending.

There is an interface mismatch to resolve when the real checkpoint is provided:
the shared trainer saves a raw `state_dict`, while the app expects a bundle with
`architecture`, `classes`, and `model`. Do not change the shared trainer. Supply the
correct 37-class ordering alongside the checkpoint, then adapt the demo loader or
package the existing weights without retraining. Do not invent a checkpoint.

CLI from repository root:
`python app/app.py --checkpoint <verified-E4-bundle.pt> --device cpu --port 7860`.
