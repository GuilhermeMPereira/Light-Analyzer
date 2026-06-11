import os
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import json
import traceback

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

# ─────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────

CALIBRATION_FACTOR  = 179.0
DEFAULT_SCALE_MAX   = 10000.0
DEFAULT_SCALE_MIN   = 1.0

ISOLINE_LEVELS_FIXED = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
ISOLINE_DENSITY_MAP = {'sparse': 5, 'normal': 8, 'dense': 14}


# ─────────────────────────────────────────────
#  RESPONSE CURVE (.rsp) — Radiance format
#
#  Suporta dois formatos comuns:
#   1. Radiance .rsp: linhas com pares (valor, expoente) por canal BGR
#   2. CSV simples: uma coluna de 256 floats (curva única aplicada aos 3 canais)
#
#  A curva mapeia DN (0-255) → fator de correção linear.
#  Quando fornecida no merge Debevec, substitui a calibração automática.
# ─────────────────────────────────────────────

def parse_rsp(rsp_bytes):
    """
    Lê uma curva de resposta .rsp do Radiance ou CSV simples.
    Retorna array (256, 3) float32 com os fatores de resposta para BGR,
    ou None se o parsing falhar.
    """
    try:
        text = rsp_bytes.decode('utf-8', errors='ignore')
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith('#')]

        floats = []
        for line in lines:
            parts = line.split()
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    pass

        n = len(floats)

        # Formato 1: 256×3 valores (BGR separados)
        if n >= 768:
            arr = np.array(floats[:768], np.float32).reshape(256, 3)
            # Garante valores positivos e normaliza para que DN=128 → 1.0
            arr = np.abs(arr)
            mid = arr[128].mean()
            if mid > 0:
                arr /= mid
            return arr

        # Formato 2: 256 valores (curva única aplicada a todos os canais)
        if n >= 256:
            col = np.abs(np.array(floats[:256], np.float32))
            mid = col[128]
            if mid > 0:
                col /= mid
            return np.stack([col, col, col], axis=1)

        return None

    except Exception:
        return None


def apply_rsp_to_images(images, rsp_curve):
    """
    Aplica a curva de resposta a cada imagem (uint8) antes do merge.
    Lineariza os valores: pixel_linear = rsp_curve[DN] * DN / 255.
    """
    if rsp_curve is None:
        return images

    corrected = []
    for img in images:
        img_f = img.astype(np.float32) / 255.0
        out = np.zeros_like(img_f)
        for c in range(3):
            # Índice DN de 0–255, interpola a curva
            dn = (img[:, :, c]).astype(np.int32)
            factor = rsp_curve[dn, c]  # (H, W)
            out[:, :, c] = np.clip(img_f[:, :, c] * factor, 0.0, 1.0)
        corrected.append((out * 255).astype(np.uint8))
    return corrected


# ─────────────────────────────────────────────
#  LEITURA DE EXIF
# ─────────────────────────────────────────────

def get_exposure_time_exif(file_bytes):
    if not PIL_AVAILABLE:
        return None
    try:
        import io
        img_pil = Image.open(io.BytesIO(file_bytes))
        exif_data = img_pil._getexif()
        if exif_data is None:
            return None
        for tag_id, value in exif_data.items():
            if TAGS.get(tag_id) == "ExposureTime":
                if hasattr(value, 'numerator'):
                    return float(value.numerator) / float(value.denominator)
                elif isinstance(value, tuple) and len(value) == 2:
                    return float(value[0]) / float(value[1])
                return float(value)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
#  MERGE HDR
# ─────────────────────────────────────────────

def resize_to_ref(imgs):
    if not imgs:
        return []
    h, w = imgs[0].shape[:2]
    return [cv2.resize(i, (w, h)) if i.shape[:2] != (h, w) else i for i in imgs]


def sanitize_hdr(hdr):
    hdr = np.nan_to_num(hdr)
    return np.clip(hdr, 0.0, None).astype(np.float32)


def merge_debevec(imgs, times):
    cal = cv2.createCalibrateDebevec()
    resp = cal.process(imgs, times=np.array(times, np.float32))
    hdr = cv2.createMergeDebevec().process(imgs, times=np.array(times, np.float32), response=resp)
    return sanitize_hdr(hdr)


def merge_robertson(imgs, times):
    cal = cv2.createCalibrateRobertson()
    resp = cal.process(imgs, times=np.array(times, np.float32))
    hdr = cv2.createMergeRobertson().process(imgs, times=np.array(times, np.float32), response=resp)
    return sanitize_hdr(hdr)


def merge_mertens(imgs):
    hdr = cv2.createMergeMertens().process(imgs)
    return np.clip(hdr, 0.0, None).astype(np.float32)


# ─────────────────────────────────────────────
#  LUMINÂNCIA
# ─────────────────────────────────────────────

def hdr_to_luminance(hdr, calib=CALIBRATION_FACTOR):
    hdr = np.clip(hdr, 0.0, None)
    if hdr.ndim == 3 and hdr.shape[2] == 3:
        B, G, R = hdr[:, :, 0], hdr[:, :, 1], hdr[:, :, 2]
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    else:
        Y = hdr.squeeze()
    lum = np.clip(Y * calib, 1e-6, None)
    return lum.astype(np.float32)


def auto_scale(lum_map):
    flat = lum_map.flatten()
    flat = flat[flat > 1e-6]
    if len(flat) == 0:
        return DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX
    p01  = float(np.percentile(flat, 1))
    p999 = float(np.percentile(flat, 99.9))
    lo = max(10 ** np.floor(np.log10(max(p01,  1e-6))), 1e-6)
    hi = 10 ** np.ceil (np.log10(max(p999, lo * 10)))
    return float(lo), float(hi)


# ─────────────────────────────────────────────
#  LUT — CORRIGIDA (menos laranja, cores mais fiéis ao Radiance/falsecolor)
#
#  Distribuição anterior era muito laranja porque o segmento laranja
#  ocupava t=0.65→0.80 (15 pontos percentuais) e comprimia o vermelho.
#  Nova distribuição:
#    • Preto/Azul escuro:  0.00 → 0.08  (muito escuro)
#    • Azul puro:          0.08 → 0.22
#    • Azul → Ciano:       0.22 → 0.36
#    • Ciano → Verde:      0.36 → 0.50
#    • Verde → Amarelo:    0.50 → 0.65  (zona conforto visual)
#    • Amarelo → Laranja:  0.65 → 0.75  (ENCURTADO — era 15pts, agora 10)
#    • Laranja → Vermelho: 0.75 → 0.90  (ALARGADO — laranja vira só transição)
#    • Vermelho → Branco:  0.90 → 1.00
# ─────────────────────────────────────────────

def _build_lut(n=2048):
    lut = np.zeros((n, 3), np.float32)  # float BGR
    stops = [
        # t_start, t_end, (R0,G0,B0), (R1,G1,B1)
        (0.000, 0.020, (0.00, 0.00, 0.00), (0.00, 0.00, 0.35)),  # preto → azul muito escuro
        (0.020, 0.080, (0.00, 0.00, 0.35), (0.00, 0.00, 0.65)),  # azul escuro intermediário
        (0.080, 0.220, (0.00, 0.00, 0.65), (0.00, 0.00, 1.00)),  # azul escuro → azul puro
        (0.220, 0.360, (0.00, 0.00, 1.00), (0.00, 1.00, 1.00)),  # azul → ciano
        (0.360, 0.500, (0.00, 1.00, 1.00), (0.00, 1.00, 0.00)),  # ciano → verde
        (0.500, 0.650, (0.00, 1.00, 0.00), (1.00, 1.00, 0.00)),  # verde → amarelo
        # Transição amarelo→laranja ENCURTADA (10pts em vez de 15)
        (0.650, 0.750, (1.00, 1.00, 0.00), (1.00, 0.45, 0.00)),  # amarelo → laranja
        # Laranja→vermelho ALARGADO: o laranja é só intermediário, vermelho domina
        (0.750, 0.880, (1.00, 0.45, 0.00), (1.00, 0.00, 0.00)),  # laranja → vermelho
        # Vermelho quente → branco: zona de saturação
        (0.880, 1.000, (1.00, 0.00, 0.00), (1.00, 1.00, 1.00)),  # vermelho → branco
    ]
    for t0, t1, (r0, g0, b0), (r1, g1, b1) in stops:
        i0 = int(t0 * (n - 1))
        i1 = int(t1 * (n - 1))
        for i in range(i0, i1 + 1):
            if i >= n:
                break
            f = (i - i0) / max(i1 - i0, 1)
            r = r0 + f * (r1 - r0)
            g = g0 + f * (g1 - g0)
            b = b0 + f * (b1 - b0)
            lut[i] = [b * 255, g * 255, r * 255]  # BGR
    return np.clip(lut, 0, 255).astype(np.uint8)


LUT = _build_lut()


def apply_false_color(lum_map, lmin, lmax):
    eps    = 1e-9
    log_lo = np.log10(max(lmin, eps))
    log_hi = np.log10(max(lmax, lmin + eps))
    log_L  = np.log10(np.clip(lum_map, lmin, lmax))
    t      = np.clip((log_L - log_lo) / (log_hi - log_lo), 0.0, 1.0)
    idx    = np.clip((t * (len(LUT) - 1)).astype(np.int32), 0, len(LUT) - 1)
    return LUT[idx].astype(np.uint8)


def enhance_false_color(fc_img, saturation=1.35, contrast=1.12, sharpness=0.6):
    """
    Deixa a imagem de falsa cor mais "viva":
      - Boost de saturação no espaço HSV
      - Leve aumento de contraste (curva S suave via CLAHE no canal V)
      - Unsharp mask sutil para realçar bordas de gradiente
    Todos os parâmetros conservadores para não distorcer a leitura fotométrica.
    """
    # ── 1. Saturação ──────────────────────────────────────────────
    hsv = cv2.cvtColor(fc_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    fc = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 2. Contraste — CLAHE no canal V ───────────────────────────
    hsv2 = cv2.cvtColor(fc, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
    hsv2[:, :, 2] = clahe.apply(hsv2[:, :, 2])
    fc = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    # ── 3. Unsharp mask suave ─────────────────────────────────────
    if sharpness > 0:
        blurred = cv2.GaussianBlur(fc, (0, 0), sigmaX=2.0)
        fc = cv2.addWeighted(fc, 1.0 + sharpness, blurred, -sharpness, 0)
        fc = np.clip(fc, 0, 255).astype(np.uint8)

    return fc


# ─────────────────────────────────────────────
#  AUTO-LEVELS DE ISOLINHA
# ─────────────────────────────────────────────

def auto_isoline_levels(lum_map, lmin, lmax, density='normal'):
    flat = lum_map.flatten()
    flat = flat[flat > 1e-6]
    if len(flat) == 0:
        return [v for v in ISOLINE_LEVELS_FIXED if lmin <= v <= lmax]

    p2  = max(float(np.percentile(flat, 2)),  lmin)
    p98 = min(float(np.percentile(flat, 98)), lmax)
    if p98 <= p2:
        return [v for v in ISOLINE_LEVELS_FIXED if lmin <= v <= lmax]

    n = ISOLINE_DENSITY_MAP.get(density, 8)
    raw = np.logspace(np.log10(p2), np.log10(p98), n + 2)[1:-1]

    nice = []
    for v in raw:
        mag  = 10 ** np.floor(np.log10(max(v, 1e-9)))
        best = min([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10],
                   key=lambda m: abs(m - v / mag))
        nice.append(round(best * mag, 6))

    seen, out = set(), []
    for v in sorted(set(nice)):
        if p2 <= v <= p98 and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ─────────────────────────────────────────────
#  ISOLINHAS — VERSÃO HALO MORFOLÓGICO
#
#  Técnica:
#  1. Binariza o mapa suavizado em cada nível
#  2. Detecta a borda via subtração de erosão/dilatação (morfológica)
#     → produz uma faixa de espessura controlada, muito mais limpa que findContours
#  3. Sobre essa máscara: pinta halo preto espesso + linha colorida fina
#  4. Décadas ganham pontilhado ciano adicional
#  5. Labels com fundo opaco blendado + borda da cor do nível na LUT
# ─────────────────────────────────────────────

def _level_color_bgr(level, lmin, lmax):
    eps = 1e-9
    log_lo = np.log10(max(lmin, eps))
    log_hi = np.log10(max(lmax, lmin + eps))
    t = np.clip((np.log10(max(level, eps)) - log_lo) / (log_hi - log_lo), 0.0, 1.0)
    idx = int(np.clip(t * (len(LUT) - 1), 0, len(LUT) - 1))
    return (int(LUT[idx][0]), int(LUT[idx][1]), int(LUT[idx][2]))


def _label_pos_on_contour(cnt, H, W, used_pts, min_dist=110):
    margin = 36
    step   = max(1, len(cnt) // 32)
    cands  = []
    for i in range(0, len(cnt), step):
        x, y = int(cnt[i][0][0]), int(cnt[i][0][1])
        if x < margin or x > W - margin - 80 or y < margin or y > H - margin - 20:
            continue
        if all(abs(x - ux) > min_dist or abs(y - uy) > min_dist for ux, uy in used_pts):
            cands.append((x, y))
    if not cands:
        return None
    cx, cy = W // 2, H // 2
    return min(cands, key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))


def draw_isolines(fc_img, lum_map, lmin, lmax, density='normal', custom_levels=None):
    result = fc_img.copy()
    H, W   = lum_map.shape

    levels = [v for v in (custom_levels or []) if lmin <= v <= lmax] \
             if custom_levels else auto_isoline_levels(lum_map, lmin, lmax, density)

    if not levels:
        return result, 0

    # Suavização agressiva para contornos limpos
    blur_k = max(7, min(25, (H // 60) * 2 + 1))
    lum_sm = cv2.GaussianBlur(lum_map, (blur_k, blur_k), 0)

    # Kernels morfológicos — adaptados à resolução
    r_halo = max(3, W // 400)        # raio do halo preto
    r_line = max(1, r_halo // 2)     # raio da linha colorida interna
    k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_halo*2+1, r_halo*2+1))
    k_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_line*2+1, r_line*2+1))
    k_dot  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(2,r_halo-1)*2+1,)*2)

    # Identifica décadas exatas
    decades = set()
    for v in levels:
        mag = 10 ** round(np.log10(max(v, 1e-9)))
        if abs(v - mag) / mag < 0.05:
            decades.add(v)

    used_label_pts = []
    drawn = 0

    for level in sorted(levels):
        is_decade = level in decades

        # Binariza em nível
        binary = np.zeros((H, W), np.uint8)
        binary[lum_sm >= level] = 255

        # ── Borda morfológica ───────────────────────────────────────
        # dilate − erode sobre o binário dá uma faixa ao redor da isoline
        dilated = cv2.dilate(binary, k_halo)
        eroded  = cv2.erode(binary,  k_halo)
        halo_mask = cv2.subtract(dilated, eroded)   # banda de ~2×r_halo px

        dilated2 = cv2.dilate(binary, k_line)
        eroded2  = cv2.erode(binary,  k_line)
        line_mask = cv2.subtract(dilated2, eroded2)  # banda fina interna

        # Verifica se há pixels suficientes
        if halo_mask.sum() == 0:
            continue

        # ── Camada 1: Halo preto ────────────────────────────────────
        result[halo_mask > 0] = (0, 0, 0)

        # ── Camada 2: Linha colorida (contraste adaptativo) ─────────
        lev_bgr = _level_color_bgr(level, lmin, lmax)
        brightness = (lev_bgr[0] + lev_bgr[1] + lev_bgr[2]) / 3
        if brightness < 70:
            line_color = (200, 200, 200)
        elif brightness > 190:
            line_color = (15, 15, 15)
        else:
            line_color = (255, 255, 255)

        result[line_mask > 0] = line_color

        # ── Camada 3: Pontilhado ciano para décadas ─────────────────
        if is_decade:
            # Cria uma máscara pontilhada dilatando apenas pontos espaçados
            dot_sparse = np.zeros((H, W), np.uint8)
            # Preenche o contorno a cada N pixels
            cnts_info = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            cnts_raw  = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
            min_p = max(40, min(W, H) * 0.03)
            for cnt in cnts_raw:
                if cv2.arcLength(cnt, True) < min_p:
                    continue
                step = max(4, len(cnt) // 50)
                for j in range(0, len(cnt), step):
                    px, py = int(cnt[j][0][0]), int(cnt[j][0][1])
                    dot_sparse[py, px] = 255
            dot_dilated = cv2.dilate(dot_sparse, k_dot)
            result[dot_dilated > 0] = (0, 255, 255)   # ciano puro

        # ── Labels ─────────────────────────────────────────────────
        if level >= 1000:
            lbl = f"{level / 1000:.1g}k"
        elif level >= 1:
            lbl = f"{int(round(level))}"
        else:
            lbl = f"{level:.2f}"

        font   = cv2.FONT_HERSHEY_SIMPLEX
        # Labels bem maiores — mínimo 0.55, máximo 0.90 px/px
        fscale = max(0.55, min(0.90, W / 900)) * (1.25 if is_decade else 1.0)
        fthick = 3 if is_decade else 2
        (tw, th), baseline = cv2.getTextSize(lbl, font, fscale, fthick)
        pad = 7  # padding generoso ao redor do texto

        # Pega contornos para posicionar label
        cnts_info = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        cnts_raw  = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
        min_p = max(40, min(W, H) * 0.03)
        valid = [c for c in cnts_raw if cv2.arcLength(c, True) >= min_p]

        for cnt in sorted(valid, key=lambda c: cv2.arcLength(c, True), reverse=True)[:3]:
            pt = _label_pos_on_contour(cnt, H, W, used_label_pts)
            if pt is None:
                continue
            lx, ly = pt
            lx = int(np.clip(lx, pad, W - tw - pad * 2))
            ly = int(np.clip(ly, th + pad, H - baseline - pad))

            bx0, by0 = lx - pad,      ly - th - pad
            bx1, by1 = lx + tw + pad, ly + baseline + pad

            # Fundo: blend escuro (30% da cor original) com borda colorida
            roi = result[by0:by1, bx0:bx1]
            if roi.size > 0:
                result[by0:by1, bx0:bx1] = (roi.astype(np.float32) * 0.22).astype(np.uint8)

            # Borda externa preta (nítida)
            cv2.rectangle(result, (bx0-1, by0-1), (bx1+1, by1+1), (0, 0, 0), 1)
            # Borda colorida do nível
            cv2.rectangle(result, (bx0, by0), (bx1, by1), lev_bgr, 1)
            # Texto branco
            cv2.putText(result, lbl, (lx, ly), font, fscale,
                        (255, 255, 255), fthick, cv2.LINE_AA)

            used_label_pts.append((lx, ly))
            break

        drawn += 1

    return result, drawn


# ─────────────────────────────────────────────
#  COLORBAR
# ─────────────────────────────────────────────

def generate_colorbar(height=400, lmin=DEFAULT_SCALE_MIN, lmax=DEFAULT_SCALE_MAX):
    W_bar = 40
    W_txt = 80
    full  = np.full((height, W_bar + W_txt, 3), 22, np.uint8)

    rows  = np.arange(height)
    t     = 1.0 - rows / (height - 1)
    idx   = np.clip((t * (len(LUT) - 1)).astype(np.int32), 0, len(LUT) - 1)
    full[:, :W_bar, :] = LUT[idx][:, np.newaxis, :]

    log_lo  = np.log10(max(lmin, 1e-9))
    log_hi  = np.log10(max(lmax, lmin + 1))
    log_rng = log_hi - log_lo

    decade = int(np.floor(log_lo))
    while decade <= int(np.ceil(log_hi)):
        for mult in [1, 2, 5]:
            v = mult * (10 ** decade)
            if lmin <= v <= lmax:
                t_v = (np.log10(v) - log_lo) / log_rng
                row = int(np.clip((1.0 - t_v) * (height - 1), 0, height - 1))
                full[row, W_bar - 6: W_bar + 2, :] = 255
                lbl = f"{int(v)}" if v >= 1 else f"{v:.2f}"
                cv2.putText(full, lbl, (W_bar + 4, row + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
        decade += 1

    return full


def generate_scale_bar_b64(width=512, height=28, lmin=DEFAULT_SCALE_MIN, lmax=DEFAULT_SCALE_MAX):
    bar  = np.zeros((height, width, 3), np.uint8)
    cols = np.arange(width)
    t    = cols / (width - 1)
    idx  = np.clip((t * (len(LUT) - 1)).astype(np.int32), 0, len(LUT) - 1)
    bar[:, :, :] = LUT[idx][np.newaxis, :, :]
    return _b64(bar)


# ─────────────────────────────────────────────
#  TONE MAPPING
# ─────────────────────────────────────────────

def tonemap(hdr_float, method='mertens'):
    if method == 'mertens':
        gamma   = 1.0 / 2.2
        preview = np.clip(hdr_float ** gamma, 0.0, 1.0)
        return (preview * 255).astype(np.uint8)
    else:
        tm  = cv2.createTonemapReinhard(gamma=1.8, intensity=0.0,
                                         light_adapt=0.8, color_adapt=0.0)
        ldr = np.clip(tm.process(hdr_float) * 255, 0, 255)
        return ldr.astype(np.uint8)


# ─────────────────────────────────────────────
#  ESTATÍSTICAS
# ─────────────────────────────────────────────

def compute_stats(lum_map):
    flat = lum_map.flatten()
    flat = flat[flat > 1e-6]
    if len(flat) == 0:
        return {}

    log_flat = np.log(flat)
    s = {
        "min":     float(np.min(flat)),
        "max":     float(np.max(flat)),
        "mean":    float(np.mean(flat)),
        "median":  float(np.median(flat)),
        "p10":     float(np.percentile(flat, 10)),
        "p90":     float(np.percentile(flat, 90)),
        "log_mean":float(np.exp(np.mean(log_flat))),
        "dynamic_range_db": float(10 * np.log10(
                                np.max(flat) / max(np.min(flat), 1e-9))),
        "pixel_count": int(len(flat)),
    }
    m = s["mean"]
    if m < 50:     s["ambiente"] = "Baixa (< 50 cd/m²) — Corredor / Armazenamento"
    elif m < 200:  s["ambiente"] = "Moderada (50–200 cd/m²) — Escritório geral"
    elif m < 500:  s["ambiente"] = "Adequada (200–500 cd/m²) — Trabalho preciso"
    elif m < 2000: s["ambiente"] = "Alta (500–2000 cd/m²) — Iluminação especial"
    else:          s["ambiente"] = "Muito alta (> 2000 cd/m²) — Exterior / Sobreexposição"
    return s


def compute_histogram(lum_map, lmin, lmax, bins=32):
    flat = lum_map.flatten()
    flat = flat[(flat >= lmin) & (flat <= lmax)]
    if len(flat) == 0:
        return []
    edges  = np.logspace(np.log10(lmin), np.log10(lmax), bins + 1)
    counts, edges = np.histogram(flat, bins=edges)
    total  = max(counts.sum(), 1)
    out    = []
    for i, c in enumerate(counts):
        center = np.sqrt(edges[i] * edges[i + 1])
        out.append({
            "x": round(float(center), 2),
            "y": round(float(c) / total * 100, 2),
            "label": f"{center:.0f}" if center >= 10 else f"{center:.1f}",
        })
    return out


# ─────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────

def _b64(img):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return base64.b64encode(buf).decode()


def ev_to_time(ev, base=0.5):
    return base * (2.0 ** float(ev))


# ─────────────────────────────────────────────
#  ROTAS
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    try:
        files         = request.files.getlist("images")
        opt_align     = request.form.get('auto_align')    == 'true'
        opt_isolines  = request.form.get('show_isolines') == 'true'
        iso_density   = request.form.get('isoline_density', 'normal')
        ev_json       = request.form.get('ev_values', '[]')
        scale_max     = float(request.form.get('scale_max', DEFAULT_SCALE_MAX))
        scale_min     = float(request.form.get('scale_min', DEFAULT_SCALE_MIN))
        merge_method  = request.form.get('merge_method', 'auto')

        if len(files) < 2:
            return jsonify({"error": "Envie pelo menos 2 imagens."}), 400

        # ── Response curve (.rsp) ─────────────────────────────────
        rsp_curve = None
        rsp_file  = request.files.get('rsp_file')
        rsp_used  = False
        if rsp_file and rsp_file.filename:
            rsp_bytes = rsp_file.read()
            rsp_curve = parse_rsp(rsp_bytes)
            rsp_used  = rsp_curve is not None

        # ── Carregar imagens ──────────────────────────────────────
        images, bytes_list = [], []
        for f in files:
            fb = f.read()
            bytes_list.append(fb)
            arr = np.frombuffer(fb, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

        if len(images) < 2:
            return jsonify({"error": "Não foi possível decodificar as imagens."}), 400

        images = resize_to_ref(images)

        # ── Aplica curva de resposta (se fornecida) ───────────────
        if rsp_curve is not None:
            images = apply_rsp_to_images(images, rsp_curve)

        # ── Tempos de exposição ───────────────────────────────────
        exp_times  = None
        exp_source = "none"

        try:
            ev_list = json.loads(ev_json)
            if ev_list and len(ev_list) == len(images):
                exp_times  = [ev_to_time(e) for e in ev_list]
                exp_source = "ev_manual"
        except Exception:
            pass

        if exp_times is None:
            exif = [get_exposure_time_exif(fb) for fb in bytes_list]
            if all(t is not None and t > 0 for t in exif):
                exp_times  = exif
                exp_source = "exif"

        # ── Alinhamento ───────────────────────────────────────────
        if opt_align:
            aligner = cv2.createAlignMTB()
            aligner.process(images, images)

        # ── Merge ─────────────────────────────────────────────────
        hdr       = None
        used_meth = "mertens"

        if exp_times is not None and merge_method != 'mertens':
            try:
                if merge_method == 'robertson':
                    hdr       = merge_robertson(images, exp_times)
                    used_meth = "robertson"
                else:
                    hdr       = merge_debevec(images, exp_times)
                    used_meth = "debevec"
            except Exception as e:
                print(f"[WARN] Merge {merge_method} falhou: {e} → fallback Mertens")
                hdr = None

        if hdr is None:
            hdr       = merge_mertens(images)
            used_meth = "mertens"
            if exp_source == "none":
                exp_source = "relative"

        # ── Luminância ────────────────────────────────────────────
        lum = hdr_to_luminance(hdr)

        auto_scaled = False
        if used_meth == 'mertens':
            detected_min, detected_max = auto_scale(lum)
            if scale_min == DEFAULT_SCALE_MIN and scale_max == DEFAULT_SCALE_MAX:
                scale_min, scale_max = detected_min, detected_max
                auto_scaled = True

        if np.log10(max(scale_max, 1e-9)) - np.log10(max(scale_min, 1e-9)) < 1.5:
            scale_max = scale_min * 100

        # ── Preview HDR ───────────────────────────────────────────
        preview = tonemap(hdr, method=used_meth)

        # ── Falsa cor limpa → enhance ─────────────────────────────
        fc_raw   = apply_false_color(lum, scale_min, scale_max)
        fc_clean = enhance_false_color(fc_raw, saturation=1.35, contrast=1.12, sharpness=0.6)

        # ── Falsa cor com isolinhas ───────────────────────────────
        drawn_count = 0
        fc_iso = fc_clean.copy()
        if opt_isolines:
            fc_iso, drawn_count = draw_isolines(
                fc_clean, lum, scale_min, scale_max, density=iso_density
            )

        # ── Colorbar e barra de escala ────────────────────────────
        colorbar  = generate_colorbar(height=fc_clean.shape[0],
                                       lmin=scale_min, lmax=scale_max)
        scale_bar = generate_scale_bar_b64(lmin=scale_min, lmax=scale_max)

        # ── Estatísticas e histograma ─────────────────────────────
        stats     = compute_stats(lum)
        histogram = compute_histogram(lum, scale_min, scale_max, bins=32)

        log_parts = [
            f"Merge: {used_meth}",
            f"Exposição: {exp_source}",
            f"Escala: {scale_min:.1f}–{scale_max:.0f} cd/m²{'[auto]' if auto_scaled else ''}",
            f"RSP: {'aplicada' if rsp_used else 'não'}",
            f"Isolinhas: {drawn_count}",
        ]

        return jsonify({
            "hdr_preview":       _b64(preview),
            "false_color_clean": _b64(fc_clean),
            "false_color":       _b64(fc_iso),
            "colorbar":          _b64(colorbar),
            "lut_preview":       scale_bar,
            "stats":             stats,
            "histogram":         histogram,
            "merge_method":      used_meth,
            "exposure_source":   exp_source,
            "scale_min":         scale_min,
            "scale_max":         scale_max,
            "auto_scaled":       auto_scaled,
            "isoline_count":     drawn_count,
            "rsp_applied":       rsp_used,
            "log":               " | ".join(log_parts),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro interno no servidor Python: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)