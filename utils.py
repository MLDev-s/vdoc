
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import cv2

_SESSION = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_CLASSES = ["documento_fisico", "pantalla"]
ID1_AR = 85.60 / 53.98  # ≈ 1.586

def _order_pts(pts):
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _four_point_warp(bgr, pts, target_ar=ID1_AR, height=640):
    tl, tr, br, bl = _order_pts(pts)
    h = int(height); w = int(round(h * target_ar))
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(bgr, M, (w, h))

def _quad_from_min_area_rect(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

def _largest_quads(bgr):
    H, W = bgr.shape[:2]
    scale = 1024 / max(H, W) if max(H, W) > 1024 else 1.0
    small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    quads = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.02 * edges.size:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = (approx.reshape(4,2) / scale).astype(np.float32)
            quads.append(("poly", q, area/edges.size))

    if not quads and cnts:
        c = cnts[0]
        area = cv2.contourArea(c)
        if area >= 0.02 * edges.size:
            q = (_quad_from_min_area_rect(c) / scale).astype(np.float32)
            quads.append(("minrect", q, area/edges.size))

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=140,
                            minLineLength=int(min(edges.shape[:2]) * 0.5), maxLineGap=20)
    if lines is not None and len(lines) >= 2:
        xs, ys = [], []
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = l
            xs += [x1,x2]; ys += [y1,y2]
        x_min, x_max = max(0,min(xs)), min(edges.shape[1]-1, max(xs))
        y_min, y_max = max(0,min(ys)), min(edges.shape[0]-1, max(ys))
        bw, bh = x_max-x_min, y_max-y_min
        if bw>0 and bh>0 and (bw*bh) >= 0.20*edges.size:
            rect = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], np.float32)
            rect = (rect / scale).astype(np.float32)
            quads.append(("hough", rect, (bw*bh)/edges.size))

    return quads

def _quad_confidence(q, kind, img_shape):
    H, W = img_shape[:2]
    area_img = float(H*W)
    hull = cv2.convexHull(q.astype(np.float32))
    area = cv2.contourArea(hull)
    area_rel = max(0.0, min(area/area_img, 1.0))
    p = q.reshape(4,2)
    p = _order_pts(p)
    w = (np.linalg.norm(p[1]-p[0]) + np.linalg.norm(p[2]-p[3]))/2.0
    h = (np.linalg.norm(p[3]-p[0]) + np.linalg.norm(p[2]-p[1]))/2.0
    ar = (w/h) if h>0 else ID1_AR
    ar_err = abs((ar - ID1_AR)/ID1_AR)
    ar_score = max(0.0, 1.0 - ar_err/0.5)
    kind_bonus = {"poly":0.15, "minrect":0.1, "hough":0.05}.get(kind,0.0)
    return max(0.0, min(area_rel*0.6 + ar_score*0.4 + kind_bonus, 1.0 )), float(ar)

def preprocess_image(image: Image.Image):
    """
    Devuelve:
      - tensor 1x3x224x224 float32 para ONNX
      - dbg: {..., 'src_quad': list[4x2] | None, 'src_quad_ar': float | None}
      - pil_roi: PIL.Image
    """
    dbg = {"localized": None, "ar_after_warp": None, "lap_var": None,
           "loc_kind": None, "loc_conf": 0.0, "src_quad": None, "src_quad_ar": None}
    pil_roi = image
    try:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        quads = _largest_quads(bgr)
        best = None
        best_conf = 0.0
        best_ar = None
        best_kind = None
        for kind, q, _ in quads:
            conf, ar = _quad_confidence(q, kind, bgr.shape)
            if conf > best_conf:
                best_conf, best_ar, best, best_kind = conf, ar, q, kind

        if best is not None and best_conf >= 0.60:
            roi = _four_point_warp(bgr, best, target_ar=ID1_AR, height=720)
            dbg["localized"] = True
            dbg["ar_after_warp"] = float(roi.shape[1] / roi.shape[0])
            dbg["loc_conf"] = float(best_conf)
            dbg["loc_kind"] = best_kind
            dbg["src_quad"] = best.astype(float).tolist()
            dbg["src_quad_ar"] = float(best_ar)
        else:
            roi = bgr
            dbg["localized"] = None
            dbg["ar_after_warp"] = None
            dbg["loc_conf"] = float(best_conf or 0.0)
            dbg["loc_kind"] = best_kind
            dbg["src_quad"] = None
            dbg["src_quad_ar"] = None

        if roi.shape[1] > roi.shape[0]:
            roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

        dbg["lap_var"] = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    except Exception:
        pass

    t = _TRANSFORM(pil_roi).unsqueeze(0).numpy().astype(np.float32)
    return t, dbg, pil_roi

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def predict(input_tensor: np.ndarray):
    inp_name = _SESSION.get_inputs()[0].name
    out = _SESSION.run(None, {inp_name: input_tensor})[0]  # (1,2) logits
    probs = _softmax(out)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    return _CLASSES[pred_idx], float(probs[0, pred_idx])
