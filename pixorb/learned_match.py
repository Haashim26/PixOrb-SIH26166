"""SIH26166 Stage 3: SuperPoint + LightGlue primary matcher, classical fallback."""
import numpy as np
import cv2

class LearnedMatcher:
    def __init__(self, max_keypoints=4096, device=None):
        self.available=False; self.reason=""
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import rbd
            self.torch=torch; self.rbd=rbd
            self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.extractor=SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher=LightGlue(features="superpoint").eval().to(self.device)
            self.available=True
        except Exception as e:
            self.reason=str(e)

    def match(self, reference, source):
        if not self.available: raise RuntimeError(self.reason)
        torch=self.torch
        def tensor(img):
            x=torch.from_numpy(img.astype(np.float32)/255.0)[None,None]
            return x.to(self.device)
        with torch.inference_mode():
            f0=self.extractor.extract(tensor(reference))
            f1=self.extractor.extract(tensor(source))
            out=self.matcher({"image0":f0,"image1":f1})
        f0,f1,m=self.rbd(f0),self.rbd(f1),self.rbd(out)
        pairs=m["matches"].detach().cpu().numpy()
        scores=m.get("scores")
        if scores is None: scores=np.ones(len(pairs),np.float32)
        else: scores=scores.detach().cpu().numpy()
        k0=f0["keypoints"].detach().cpu().numpy(); k1=f1["keypoints"].detach().cpu().numpy()
        matches=[]
        for i,(a,b) in enumerate(pairs):
            matches.append((float(k1[b,0]),float(k1[b,1]),float(k0[a,0]),float(k0[a,1]),float(scores[i])))
        return matches

def classical_fallback(reference, source, max_features=4000):
    sift=cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.01)
    k0,d0=sift.detectAndCompute(reference,None); k1,d1=sift.detectAndCompute(source,None)
    if d0 is None or d1 is None: return []
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    knn=matcher.knnMatch(d1,d0,k=2)
    out=[]
    for pair in knn:
        if len(pair)!=2: continue
        m,n=pair
        if m.distance < 0.72*n.distance:
            p1=k1[m.queryIdx].pt; p0=k0[m.trainIdx].pt
            out.append((p1[0],p1[1],p0[0],p0[1],float(1.0-m.distance/(n.distance+1e-6))))
    return out

def loftr_fallback(reference, source, device=None):
    """Optional learned fallback matching using Kornia LoFTR.

    LoFTR is intentionally optional. It is used only when SuperPoint+LightGlue is
    unavailable or returns too few correspondences, matching the submitted hybrid
    learned-matching approach before the final classical fallback.
    """
    try:
        import torch
        from kornia.feature import LoFTR
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        matcher = LoFTR(pretrained="outdoor").to(dev).eval()
        t0 = torch.from_numpy(reference.astype(np.float32) / 255.0)[None, None].to(dev)
        t1 = torch.from_numpy(source.astype(np.float32) / 255.0)[None, None].to(dev)
        with torch.inference_mode():
            out = matcher({"image0": t0, "image1": t1})
        if not out or "keypoints0" not in out:
            return []
        k0 = out["keypoints0"].detach().cpu().numpy()
        k1 = out["keypoints1"].detach().cpu().numpy()
        conf = out.get("confidence")
        if conf is None:
            scores = np.ones(len(k0), np.float32)
        else:
            scores = conf.detach().cpu().numpy()
        return [(float(k1[i,0]), float(k1[i,1]), float(k0[i,0]), float(k0[i,1]), float(scores[i]))
                for i in range(len(k0))]
    except Exception:
        return []


def match_images(reference, source, min_learned=12):
    lm = LearnedMatcher()
    if lm.available:
        try:
            matches = lm.match(reference, source)
            if len(matches) >= min_learned:
                return matches, {"backend":"SuperPoint+LightGlue", "learned_available":True, "fallback_reason":"", "learned_backend_fallback":False}
        except Exception as e:
            lm.reason = str(e)

    loftr = loftr_fallback(reference, source)
    if len(loftr) >= min_learned:
        return loftr, {"backend":"LoFTR", "learned_available":True, "fallback_reason":lm.reason, "learned_backend_fallback":True}

    matches = classical_fallback(reference, source)
    return matches, {"backend":"SIFT fallback", "learned_available":False, "fallback_reason":lm.reason or "SuperPoint+LightGlue and LoFTR produced too few matches", "learned_backend_fallback":False}
