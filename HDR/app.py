import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64

app = Flask(__name__)

# --- CONFIGURAÇÕES ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- FUNÇÕES AUXILIARES ---

def resize_to_reference(img_list):
    if not img_list: return []
    ref_h, ref_w = img_list[0].shape[:2]
    resized_list = []
    for img in img_list:
        h, w = img.shape[:2]
        if h != ref_h or w != ref_w:
            img = cv2.resize(img, (ref_w, ref_h))
        resized_list.append(img)
    return resized_list

def bio_inspired_hdr(images):
    # Mertens 
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)
    hdr_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    return hdr_8bit

def generate_custom_blue_red_lut():
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    
    for i in range(256):
        # (Mantém a regra de só Azul e Vermelho misturados, Verde = 0)
        lut[i, 0, 1] = 0
        
        if i < 128:
            # Baixa luz (Azul -> Magenta)
            lut[i, 0, 0] = 255                # Azul fixo
            lut[i, 0, 2] = i * 2              # Vermelho sobe
        else:
            # Alta luz (Magenta -> Vermelho)
            lut[i, 0, 0] = max(0, 255 - (i - 128) * 2)  # Azul desce
            lut[i, 0, 2] = 255                          # Vermelho fixo
            
    return lut

def generate_false_color(image_8bit):
    # 1. Converter para escala de cinza
    gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
    
    # 2. Normalizar
    norm_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 3. Converter para 3 canais (Hack para cv2.LUT funcionar)
    img_3ch = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    
    # 4. Gerar e Aplicar a LUT
    custom_lut = generate_custom_blue_red_lut()
    false_color_img = cv2.LUT(img_3ch, custom_lut)
    
    return false_color_img

def array_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# --- ROTAS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    uploaded_files = request.files.getlist("images")
    opt_align = request.form.get('auto_align') == 'true'
    
    if len(uploaded_files) < 2:
        return jsonify({"error": "Envie pelo menos 2 imagens."}), 400

    images_cv = []

    try:
        # Carregar imagens
        for file in uploaded_files:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images_cv.append(img)

        images_cv = resize_to_reference(images_cv)

        if opt_align:
            print("Alinhando imagens...")
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(images_cv, images_cv)

        print("Processando HDR (Mertens)...")
        result_hdr = bio_inspired_hdr(images_cv)

        # Gerar Falsa cor
        false_color = generate_false_color(result_hdr)

        return jsonify({
            "hdr_preview": array_to_base64(result_hdr),
            "false_color": array_to_base64(false_color),
            "log": "Processado com sucesso"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)