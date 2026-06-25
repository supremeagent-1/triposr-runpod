"""RunPod serverless handler — TripoSR image→3D (returns a GLB as base64).
input:  {"image": "<url or data-uri or base64>", "foreground_ratio": 0.85, "mc_resolution": 256}
output: {"mesh_base64": "<glb b64>", "format": "glb"}
"""
import base64, io, os, tempfile, traceback
import requests
from PIL import Image
import numpy as np
import torch
import runpod

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        from tsr.system import TSR
        m = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        m.renderer.set_chunk_size(8192)
        m.to("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = m
    return _MODEL


def _read_image(spec: str) -> Image.Image:
    if spec.startswith("http"):
        data = requests.get(spec, timeout=60).content
    elif spec.startswith("data:"):
        data = base64.b64decode(spec.split(",", 1)[1])
    else:
        data = base64.b64decode(spec)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _preprocess(img: Image.Image, ratio: float) -> Image.Image:
    # remove background + center on a neutral plate (TripoSR expects a clean foreground)
    try:
        import rembg
        from tsr.utils import remove_background, resize_foreground
        session = rembg.new_session()
        img = remove_background(img, session)
        img = resize_foreground(img, ratio)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        img = Image.fromarray((arr * 255.0).astype(np.uint8))
    except Exception:
        pass
    return img


def handler(job):
    try:
        inp = job.get("input", {})
        spec = inp.get("image")
        if not spec:
            return {"error": "input.image required (url / data-uri / base64)"}
        ratio = float(inp.get("foreground_ratio", 0.85))
        mc = int(inp.get("mc_resolution", 256))

        model = _load_model()
        img = _preprocess(_read_image(spec), ratio)
        with torch.no_grad():
            scene_codes = model([img], device=next(model.parameters()).device)
            meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc)
        mesh = meshes[0]

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            mesh.export(f.name)
            path = f.name
        b = open(path, "rb").read()
        os.unlink(path)
        return {"mesh_base64": base64.b64encode(b).decode(), "format": "glb",
                "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-1500:]}


runpod.serverless.start({"handler": handler})
