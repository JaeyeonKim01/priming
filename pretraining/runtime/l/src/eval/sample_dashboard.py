"""Fixed-input OST sampling and a browser that opens directly from disk."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .dashboard_data import ROOT, atomic_write, contract_hash, panel_contract, read_json, sha256, write_json


def shared_inputs(contract):
    import torch
    n = contract["samples_per_class"]
    size = contract["image_size"]
    return dict(
        pixel_noise=torch.randn(n, 3, size, size,
            generator=torch.Generator().manual_seed(contract["gaussian_seed"])),
        drift_noise_labels=torch.randint(0, 64, (n, 32),
            generator=torch.Generator().manual_seed(contract["discrete_seed"])),
        classes=torch.tensor([row["id"] for row in contract["classes"]]),
    )


def save_panel(output, pixels, metadata, contract):
    import hashlib
    import numpy as np
    import torch
    from PIL import Image
    output = Path(output)
    n, c, size = contract["samples_per_class"], len(contract["classes"]), contract["image_size"]
    if tuple(pixels.shape) != (n, c, 3, size, size) or not torch.isfinite(pixels).all():
        raise ValueError("Incomplete or nonfinite sample panel")
    if float(pixels.min()) < 0 or float(pixels.max()) > 1:
        raise ValueError("Samples must already be in [0,1]")
    output.mkdir(parents=True, exist_ok=True)
    raw = pixels.float().cpu().numpy()
    images = {}
    for index, row in enumerate(contract["classes"]):
        strip = np.concatenate([raw[i, index].transpose(1, 2, 0) for i in range(n)], axis=1)
        path = output / f'class_{row["id"]:04d}.png'
        temporary = path.with_suffix(".tmp.png")
        Image.fromarray((strip * 255 + 0.5).astype(np.uint8)).save(temporary)
        os.replace(temporary, path)
        images[str(row["id"])] = dict(file=path.name, sha256=sha256(path))
    record = dict(**metadata, contract=contract, contract_sha256=contract_hash(contract),
        sample_count=n*c, float_samples_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        images=images, complete=True)
    write_json(output / "manifest.json", record)
    return output / "manifest.json"


def export_ost_panel(args, model, ema, output, *, step, checkpoint, label=None):
    import torch
    from src.utils.sampling_util import generate_images
    contract = panel_contract()
    if model.in_channels != 3 or model.input_size != contract["image_size"]:
        raise ValueError("This fixed panel is for ImageNet-256 pixel generators")
    if args.num_sampling_steps != 1 or float(args.noise_scale) != 1.0:
        raise ValueError("The fixed panel requires one-step, unit-noise generation")
    bank = shared_inputs(contract)
    device = next(model.parameters()).device
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode(), ema.swap(model, label=contract["weights"]):
            samples = [generate_images(args, model, labels=bank["classes"].to(device),
                cfg=contract["cfg"],
                z_t=noise.to(device)[None].repeat(25, 1, 1, 1)).cpu()
                for noise in bank["pixel_noise"]]
    finally:
        model.train(was_training)
    return save_panel(output, torch.stack(samples), dict(
        model=label or args.exp_name, step=step, weights=contract["weights"],
        checkpoint=str(checkpoint), family="OST", cfg=contract["cfg"],
        sampling_steps=contract["sampling_steps"],
    ), contract)


def load_panel(path, contract):
    path = Path(path)
    record = read_json(path)
    if not record.get("complete") or record["sample_count"] != 200:
        raise ValueError(f"Incomplete panel: {path}")
    if record["contract_sha256"] != contract_hash(contract) or record["contract"] != contract:
        raise ValueError(f"Sample comparison contract mismatch: {path}")
    for row in record["images"].values():
        if sha256(path.parent / row["file"]) != row["sha256"]:
            raise ValueError(f"Sample checksum mismatch: {path}")
    return record


def build_sample_dashboard(output, manifests):
    """All model rows must have the same complete, checksummed input contract."""
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = panel_contract()
    models = []
    for path in map(Path, manifests):
        record = load_panel(path, contract)
        models.append(dict(model=record["model"], step=record["step"],
            weights=record["weights"], manifest=os.path.relpath(path, output),
            images={key: os.path.relpath(path.parent / row["file"], output)
                    for key, row in record["images"].items()}))
    if not models:
        raise ValueError("No sample panels supplied")
    # A direct PNG gives a compact view of the first predeclared input.
    tile, margin, header, row_height = 144, 160, 36, 166
    overview = Image.new("RGB", (margin + 5 * tile, header + 5 * (25 + len(models)*row_height)), "white")
    draw = ImageDraw.Draw(overview)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw.text((12, 9), "Fixed 25-class comparison | input 1 of 8 | EMA | CFG 1", fill="#20352e", font=title_font)
    for group in range(5):
        y = header + group * (25 + len(models)*row_height)
        classes = contract["classes"][group*5:(group+1)*5]
        for j, row in enumerate(classes):
            draw.text((margin + j*tile + 3, y+3), row["name"], fill="#444444", font=font)
        for i, model in enumerate(models):
            top = y + 25 + i*row_height
            draw.text((8, top+30), model["model"], fill="#20352e" if i == 0 else "#666666", font=font)
            draw.text((8, top+52), f'{model["step"]:,} updates', fill="#666666", font=font)
            for j, row in enumerate(classes):
                with Image.open(output / model["images"][str(row["id"])] ) as image:
                    sample = image.crop((0, 0, 256, 256)).resize((tile, tile), Image.Resampling.LANCZOS)
                overview.paste(sample, (margin+j*tile, top))
    temporary = output / "overview.tmp.png"
    overview.save(temporary)
    os.replace(temporary, output / "overview.png")
    data = json.dumps(dict(models=models, contract=contract)).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OST · fixed sample comparison</title><style>
body{margin:0;background:#f4f5f3;color:#22322c;font:15px system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:30px}p{line-height:1.6;max-width:1050px}a{color:#216449}nav{position:sticky;top:0;background:#f4f5f3ee;padding:16px 0;z-index:2}select{padding:9px;margin-right:20px}section{background:white;border:1px solid #dde4df;border-radius:10px;margin:20px 0;padding:18px}h2{font-size:19px}.row{display:grid;grid-template-columns:180px 1fr;align-items:center;margin:12px 0}.label{font-weight:600}.label small{display:block;color:#718078;font-weight:400}.strip{width:100%;display:block}.single{width:256px;height:256px;background-size:800% 100%;max-width:100%}.note{color:#647268}@media(max-width:700px){main{padding:15px}.row{display:block}.label{margin:12px 0}.strip{min-width:900px}.samples{overflow:auto}}
</style><main><h1>OST and pixel Drifting · fixed sample comparison</h1>
<p>25 predeclared classes · 20 animals and five structural or scene challenges · eight inputs per class · 200 samples per model. EMA, one step, CFG 1. No samples are selected by appearance. Shared noise does not require matching scenes.</p>
<p><a href="overview.png">Open overview PNG</a> · <a href="comparison.json">Sampling provenance</a></p>
<nav><label>Class <select id="class"><option value="all">All 25 classes</option></select></label><label>Input <select id="seed"><option value="all">All eight inputs</option></select></label></nav><div id="panels"></div>
<p class="note">Pixels are saved as PNG at native 256px resolution. The overview resizes input 1 for readability. FID and FDr5 come from separate 50,000-image evaluations, never from this panel.</p></main><script>
const data=__DATA__;
const classes=document.getElementById('class'), seed=document.getElementById('seed');
for(const c of data.contract.classes){const o=new Option(c.id+' · '+c.name,c.id);classes.add(o)}
for(let i=0;i<8;i++)seed.add(new Option('Input '+(i+1),i));
function render(){const root=document.getElementById('panels');root.replaceChildren();
for(const c of data.contract.classes){if(classes.value!=='all'&&classes.value!==String(c.id))continue;
const section=document.createElement('section'), h=document.createElement('h2');h.textContent=c.id+' · '+c.name;section.append(h);
for(const m of data.models){const row=document.createElement('div');row.className='row';const label=document.createElement('div');label.className='label';label.textContent=m.model;const sub=document.createElement('small');sub.textContent=m.step.toLocaleString()+' updates · EMA '+m.weights;label.append(sub);row.append(label);const wrap=document.createElement('div');wrap.className='samples';
if(seed.value==='all'){const img=document.createElement('img');img.className='strip';img.loading='lazy';img.src=m.images[c.id];img.alt=m.model+' · '+c.name+' · inputs 1–8';wrap.append(img)}else{const img=document.createElement('div');img.className='single';img.style.backgroundImage='url('+JSON.stringify(m.images[c.id])+')';img.style.backgroundPosition=(Number(seed.value)/7*100)+'% 0';img.setAttribute('role','img');img.setAttribute('aria-label',m.model+' · '+c.name+' · input '+(Number(seed.value)+1));wrap.append(img)}row.append(wrap);section.append(row)}root.append(section)}}
classes.addEventListener('change',render);seed.addEventListener('change',render);render();
</script></html>'''.replace("__DATA__", data)
    atomic_write(output / "index.html", page)
    write_json(output / "comparison.json", dict(contract=contract, models=models))
    return output / "index.html"
