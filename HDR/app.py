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
    
    # Mertens fusion: pondera pixels por Contraste, Saturação e Bem-estar (Exposure)
    merge_mertens = cv2.createMergeMertens()
    
    # O resultado é uma imagem float32 entre 0 e 1
    hdr_image = merge_mertens.process(images)
    
    # Converter para 8-bit (0-255) para visualização
    hdr_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    return hdr_8bit

def generate_false_color(image_8bit):
    gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
    norm_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

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
    
    # Checkboxes opcionais
    opt_align = request.form.get('auto_align') == 'true'
    
    if len(uploaded_files) < 2:
        return jsonify({"error": "Envie pelo menos 2 imagens."}), 400

    images_cv = []

    try:
        # 1. Carregar imagens
        for file in uploaded_files:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images_cv.append(img)

        # 2. Normalizar tamanhos
        images_cv = resize_to_reference(images_cv)

        # 3. Alinhamento
        if opt_align:
            print("Alinhando imagens (Spatial Alignment)...")
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(images_cv, images_cv)

        # 4. Processamento "Bio-Inspirado" (Mertens)
        print("Processando HDR via Fusão Espacial (Método Mertens)...")
        result_hdr = bio_inspired_hdr(images_cv)

     
        false_color = generate_false_color(result_hdr)

        return jsonify({
            "hdr_preview": array_to_base64(result_hdr),
            "false_color": array_to_base64(false_color),
            "log": "Processado com sucesso"
        })

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)