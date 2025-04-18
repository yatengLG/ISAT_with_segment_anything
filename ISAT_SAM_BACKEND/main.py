# -*- coding: utf-8 -*-
# @Author  : LG
import os.path

import torch
from fastapi import FastAPI, Request, UploadFile, File, Body, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
import numpy as np
import json
from pathlib import Path
from ISAT_SAM_BACKEND.segment_any.segment_any import SegAny
from ISAT_SAM_BACKEND.segment_any.model_zoo import model_dict


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加载语言文件
LOCALES = {
    'en': json.loads(Path('static/locales/en.json').read_text()),
    'zh': json.loads(Path('static/locales/zh.json').read_text())
}

def get_locale(request: Request):
    # 获取语言参数，默认英语
    lang = request.query_params.get('lang', 'en')
    return LOCALES.get(lang, 'en')


# init sam
def sam_init(model_name='mobile_sam.pt', use_bfloat16=False):
    segany = SegAny(f'checkpoints/{model_name}', use_bfloat16=use_bfloat16)
    return segany

segany = sam_init()

@torch.no_grad()
async def sam_encode(image: np.ndarray):
    with torch.inference_mode(), torch.autocast(segany.device,
                                                dtype=segany.model_dtype,
                                                enabled=torch.cuda.is_available()):
        if 'sam2' in segany.model_type:
            _orig_hw = tuple([image.shape[:2]])
            input_image = segany.predictor_with_point_prompt._transforms(image)
            input_image = input_image[None, ...].to(segany.predictor_with_point_prompt.device)
            backbone_out = segany.predictor_with_point_prompt.model.forward_image(input_image)
            _, vision_feats, _, _ = segany.predictor_with_point_prompt.model._prepare_backbone_features(
                backbone_out)
            if segany.predictor_with_point_prompt.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[
                                       -1] + segany.predictor_with_point_prompt.model.no_mem_embed
            feats = [
                        feat.permute(1, 2, 0).view(1, -1, *feat_size)
                        for feat, feat_size in
                        zip(vision_feats[::-1], segany.predictor_with_point_prompt._bb_feat_sizes[::-1])
                    ][::-1]
            _features = {"image_embed": feats[-1], "high_res_feats": tuple(feats[:-1])}
            return _features, _orig_hw, _orig_hw
        else:
            input_image = segany.predictor_with_point_prompt.transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image,
                                                device=segany.predictor_with_point_prompt.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            original_size = image.shape[:2]
            input_size = tuple(input_image_torch.shape[-2:])

            input_image = segany.predictor_with_point_prompt.model.preprocess(input_image_torch)
            features = segany.predictor_with_point_prompt.model.image_encoder(input_image)
        return features, original_size, input_size


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, _: dict = Depends(get_locale)):

    return templates.TemplateResponse("index.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en'),
        "current_checkpoint": segany.checkpoint
    })

@app.get("/model", response_class=HTMLResponse)
async def model(request: Request, _: dict = Depends(get_locale)):
    model_dict_update = model_dict.copy()
    for model_name in model_dict_update:
        model_dict_update[model_name]['downloaded'] = os.path.exists(f'checkpoints/{model_name}')

    return templates.TemplateResponse("model.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en'),
        "model_dict": model_dict_update
    })

@app.get("/api", response_class=HTMLResponse)
async def api(request: Request, _: dict = Depends(get_locale)):
    return templates.TemplateResponse("api.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en')
    })

@app.post("/encode")
async def encode(file: bytes=File(...), shape: str=Form(...), dtype: str=Form(...)):
    try:
        #
        shape = tuple(map(int, shape.split(',')))
        image_data = np.frombuffer(file, eval(f'np.{dtype}')).reshape(shape)
        # process
        features, original_size, input_size = await sam_encode(image_data)
        features = features.detach().cpu().numpy()

        return {"features": features.tolist(), "original_size": original_size, "input_size": input_size}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise e


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)

