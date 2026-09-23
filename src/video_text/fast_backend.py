from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

from .types import Candidate


@dataclass(slots=True)
class ScoredComponent:
    bbox: np.ndarray
    score: float
    area: int

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        self.score = float(self.score)
        self.area = int(self.area)


def split_levels(components, high_score, low_score, high_min_area, low_min_area):
    high, low = [], []
    for comp in components:
        if comp.score >= high_score and comp.area >= high_min_area:
            high.append(Candidate(comp.bbox, comp.score, "HIGH", "fast"))
        elif comp.score >= low_score and comp.area >= low_min_area:
            low.append(Candidate(comp.bbox, comp.score, "LOW", "fast"))
    return high, low


class FastBackend:
    def __init__(
        self,
        repo,
        checkpoint,
        device="cuda",
        high_score=.88,
        low_score=.75,
        high_min_area=250,
        low_min_area=140,
    ):
        self.repo=Path(repo)
        if str(self.repo) not in sys.path:
            sys.path.insert(0,str(self.repo))
        from models import build_model
        from models.utils import rep_model_convert, fuse_module

        cfg=Config.fromfile(str(self.repo/"config/fast/ic15/fast_base_ic15_736_finetune_ic17mlt.py"))
        old=os.getcwd()
        try:
            os.chdir(self.repo)
            model=build_model(cfg.model)
        finally:
            os.chdir(old)
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        sd={k.replace("module.",""):v for k,v in ck["ema"].items()}
        model.load_state_dict(sd,strict=True)
        model=rep_model_convert(model)
        model=fuse_module(model)
        self.model=model.to(device).eval()
        self.device=device
        self.high_score=float(high_score)
        self.low_score=float(low_score)
        self.high_min_area=int(high_min_area)
        self.low_min_area=int(low_min_area)
        self.forward_calls=0

    @staticmethod
    def _resize_long(im,long_side=736):
        h,w=im.shape[:2]
        scale=float(long_side)/max(h,w)
        nh=max(32,int(round(h*scale)))
        nw=max(32,int(round(w*scale)))
        nh=((nh+31)//32)*32
        nw=((nw+31)//32)*32
        return cv2.resize(im,(nw,nh)), (h,w)

    def _prep(self,images):
        resized=[]; orig=[]
        for im in images:
            x,hw=self._resize_long(im)
            resized.append(x); orig.append(hw)
        H=max(x.shape[0] for x in resized)
        W=max(x.shape[1] for x in resized)
        batch=np.zeros((len(images),3,H,W),np.float32)
        mean=np.array([0.485,0.456,0.406],np.float32)[:,None,None]
        std=np.array([0.229,0.224,0.225],np.float32)[:,None,None]
        sizes=[]
        for i,x in enumerate(resized):
            rgb=cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            batch[i,:,:x.shape[0],:x.shape[1]]=(rgb.transpose(2,0,1)-mean)/std
            sizes.append(x.shape[:2])
        return torch.from_numpy(batch).to(self.device),orig,sizes

    def _forward_model(self,x):
        self.forward_calls += 1
        f=self.model.backbone(x)
        f=self.model.neck(f)
        return self.model.det_head(f)

    def _decode_scored_components(self,out,origs,sizes):
        B=out.shape[0]
        H,W=out.shape[-2]*4,out.shape[-1]*4
        texts=F.interpolate(out[:,0:1],size=(H//2,W//2),mode="nearest")
        pool=max(1,self.model.det_head.pooling_size//2+1)
        texts=F.max_pool2d(texts,kernel_size=pool,stride=1,padding=pool//2)
        scores=torch.sigmoid(texts)
        scores=F.interpolate(scores,size=(H,W),mode="nearest").squeeze(1).cpu().numpy()
        kernels=(out[:,0]>0).to(torch.uint8).cpu().numpy()
        results=[]
        for bi in range(B):
            ih,iw=sizes[bi]
            oh,ow=origs[bi]
            kh=max(1,int(round(ih*out.shape[-2]/H)))
            kw=max(1,int(round(iw*out.shape[-1]/W)))
            ker=kernels[bi,:kh,:kw]
            n,lab_small=cv2.connectedComponents(ker,connectivity=4)
            lab=torch.from_numpy(lab_small.astype(np.float32))[None,None]
            lab=F.interpolate(lab,size=(H//2,W//2),mode="nearest")
            lab=F.max_pool2d(lab,kernel_size=pool,stride=1,padding=pool//2)
            lab=F.interpolate(lab,size=(H,W),mode="nearest")[0,0].numpy().astype(np.int32)
            lab=lab[:ih,:iw]
            score=scores[bi,:ih,:iw]
            sx=ow/float(iw); sy=oh/float(ih)
            comps=[]
            for label_id in range(1,n):
                yy,xx=np.where(lab==label_id)
                area=len(xx)
                if area < self.low_min_area:
                    continue
                sc=float(score[yy,xx].mean())
                if sc < self.low_score:
                    continue
                pts=np.stack([xx,yy],1).astype(np.float32)
                rect=cv2.minAreaRect(pts)
                rw,rh=rect[1]
                if rw*rh<=1:
                    continue
                alpha=(len(pts)/max(rw*rh,1e-6))**0.25
                rect=(rect[0],(rw*alpha,rh*alpha),rect[2])
                quad=cv2.boxPoints(rect).astype(np.float32)
                quad[:,0]*=sx
                quad[:,1]*=sy
                bbox=np.array([quad[:,0].min(),quad[:,1].min(),quad[:,0].max(),quad[:,1].max()],np.float32)
                comps.append(ScoredComponent(bbox,sc,area))
            results.append(comps)
        return results

    @torch.inference_mode()
    def detect_batch(self,images):
        if not images:
            return []
        x,origs,sizes=self._prep(images)
        out=self._forward_model(x)
        components=self._decode_scored_components(out,origs,sizes)
        return [
            split_levels(
                comps,
                self.high_score,self.low_score,
                self.high_min_area,self.low_min_area,
            )
            for comps in components
        ]
