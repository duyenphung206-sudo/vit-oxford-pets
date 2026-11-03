"""T37 partial demo, adapted from personal Predictor/demo; no attention yet."""
import argparse
import torch
from PIL import Image, ImageOps
from torchvision.models import vit_b_16, ViT_B_16_Weights

class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        self.device=torch.device(device)
        ckpt=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if ckpt.get("architecture")!="vit_b_16" or len(ckpt.get("classes",[]))!=37:
            raise ValueError("Unsupported checkpoint")
        if ckpt.get("smoke_only",False): raise ValueError("Reject smoke-only checkpoint")
        self.classes=ckpt["classes"]
        self.model=vit_b_16(weights=None)
        self.model.heads.head=torch.nn.Linear(768,37)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    @torch.inference_mode()
    def __call__(self,image):
        if not isinstance(image,Image.Image): raise ValueError("Upload a valid image")
        x=self.transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(self.device)
        probs=self.model(x).float().softmax(-1)[0]
        values,indices=probs.topk(5)
        return {self.classes[i]:float(v) for i,v in zip(indices.cpu().tolist(),values.cpu().tolist())}

def create_demo(predictor):
    import gradio as gr
    def classify(image):
        try: return predictor(image)
        except (ValueError,OSError) as exc: raise gr.Error(str(exc)) from exc
    return gr.Interface(fn=classify,inputs=gr.Image(type="pil",sources=["upload"]),
        outputs=gr.Label(num_top_classes=5),title="Oxford-IIIT Pet — Top 5",
        description="T37 partial: attention map pending verification. Softmax scores are not calibrated correctness probabilities.",flagging_mode="never")

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--device",default="cpu");p.add_argument("--port",type=int,default=7860)
    a=p.parse_args();create_demo(Predictor(a.checkpoint,a.device)).launch(server_name="127.0.0.1",server_port=a.port)
if __name__=="__main__":main()
