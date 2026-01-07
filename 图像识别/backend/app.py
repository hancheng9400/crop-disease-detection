# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindcv.models import create_model
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import time

# === 新增：前端文件配置（关键！）===
FRONTEND_DIR = u"/home/developer/lzh/frontend"  # 前端文件夹绝对路径
# 验证前端目录存在性
if not os.path.exists(FRONTEND_DIR):
    print(f"[WARNING] Frontend directory not found: {FRONTEND_DIR}")

# === 1. 全局配置中心 (核心部分) ===
CROPS_CONFIG = {
    "corn": {
        "ckpt_path": "ckpt/corn_best_epoch.ckpt",
        "num_classes": 4,
        "classes": [
            'Corn___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn___Common_rust', 
            'Corn___Northern_Leaf_Blight', 
            'Corn___healthy'
        ],
        "translation": {
            'Corn___Cercospora_leaf_spot Gray_leaf_spot': '玉米_灰斑病',
            'Corn___Common_rust': '玉米_普通锈病',
            'Corn___Northern_Leaf_Blight': '玉米_大斑病',
            'Corn___healthy': '健康_玉米'
        }
    },
    "potato": {
        "ckpt_path": "ckpt/potato_best_epoch.ckpt",
        "num_classes": 3,
        "classes": [
            'Potato___Early_blight', 
            'Potato___Late_blight', 
            'Potato___healthy'
        ],
        "translation": {
            'Potato___Early_blight': '马铃薯_早疫病', 
            'Potato___Late_blight': '马铃薯_晚疫病',
            'Potato___healthy': '健康_马铃薯'
        }
    },
    "tomato": {
        "ckpt_path": "ckpt/tomato_best_epoch.ckpt",
        "num_classes": 10,
        "classes": [
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
            'Tomato___healthy'
        ],
        "translation": {
            'Tomato___Bacterial_spot': '番茄_细菌性斑点病',
            'Tomato___Early_blight': '番茄_早疫病',
            'Tomato___Late_blight': '番茄_晚疫病',
            'Tomato___Leaf_Mold': '番茄_叶霉病',
            'Tomato___Septoria_leaf_spot': '番茄_斑枯病',
            'Tomato___Spider_mites Two-spotted_spider_mite': '番茄_二斑叶螨(红蜘蛛)',
            'Tomato___Target_Spot': '番茄_靶斑病',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': '番茄_黄化曲叶病毒病',
            'Tomato___Tomato_mosaic_virus': '番茄_花叶病毒病',
            'Tomato___healthy': '健康_番茄'
        }
    },
    "apple": {
        "ckpt_path": "ckpt/apple_best_epoch.ckpt",
        "num_classes": 4,
        "classes": [
            'Apple___Apple_scab', 'Apple___Black_rot', 
            'Apple___Cedar_apple_rust', 'Apple___healthy'
        ],
        "translation": {
            'Apple___Apple_scab': '苹果_黑星病',
            'Apple___Black_rot': '苹果_黑腐病',
            'Apple___Cedar_apple_rust': '苹果_锈病',
            'Apple___healthy': '健康_苹果'
        }
    },
    "grape": {
        "ckpt_path": "ckpt/grape_best_epoch.ckpt",
        "num_classes": 4,
        "classes": [
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'
        ],
        "translation": {
            'Grape___Black_rot': '葡萄_黑腐病',
            'Grape___Esca_(Black_Measles)': '葡萄_黑麻疹病',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': '葡萄_叶枯病(褐斑病)',
            'Grape___healthy': '健康_葡萄'
        }
    }
}

# === 初始化Flask（添加前端静态文件托管）===
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,  # 前端静态文件目录
    static_url_path='/'  # 根路径映射到前端目录
)
CORS(app)

# 全局变量：缓存已加载的模型
LOADED_MODELS = {}

def get_or_load_model(crop_type):
    """按需加载模型"""
    global LOADED_MODELS
    
    if crop_type not in CROPS_CONFIG:
        return None, "未知的作物类型"
    
    if crop_type in LOADED_MODELS:
        return LOADED_MODELS[crop_type], None

    config = CROPS_CONFIG[crop_type]
    ckpt_path = config['ckpt_path']
    
    if not os.path.exists(ckpt_path):
        return None, f"模型文件不存在: {ckpt_path}"

    try:
        print(f"[INFO] 正在加载 {crop_type} 模型...")
        model_arch = 'resnet50' if crop_type != 'corn' else 'mobilenet_v3_small_100'
        
        network = create_model(model_name=model_arch, 
                               num_classes=config['num_classes'], 
                               pretrained=False)
        
        param_dict = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(network, param_dict)
        network.set_train(False)
        
        LOADED_MODELS[crop_type] = network
        return network, None
    except Exception as e:
        return None, str(e)

def preprocess_image_for_ms(img_np):
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[:2]
    start_h = (h - 224) // 2
    start_w = (w - 224) // 2
    img = img[start_h:start_h+224, start_w:start_w+224]
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return Tensor(img[np.newaxis, :], ms.float32)

def analyze_spectral_health(img_bgr):
    img_float = img_bgr.astype(np.float32)
    B, G, R = cv2.split(img_float)
    denominator = G + R - B + 0.001
    vari = (G - R) / denominator
    health_score = np.mean(vari) * 100 + 50
    health_score = np.clip(health_score, 0, 100)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    _, buffer = cv2.imencode('.png', heatmap)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    return health_score, heatmap_base64

# === 核心接口：/predict（保持不变）===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 获取参数
        crop_type = request.form.get('crop_type')
        if not crop_type:
            return jsonify({"error": "请选择作物类型"}), 400
            
        if 'image' not in request.files:
            return jsonify({"error": "未上传图片"}), 400

        image_file = request.files['image']
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 2. 加载对应的模型
        model, err = get_or_load_model(crop_type)
        if model is None:
            return jsonify({"error": f"模型加载失败: {err}"}), 500

        # 3. 设置上下文
        if not context.get_context("device_target"):
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

        # 4. 推理
        input_tensor = preprocess_image_for_ms(original_img)
        logits = model(input_tensor)
        probabilities = ms.ops.Softmax()(logits)
        probs_np = probabilities.asnumpy().flatten()
        
        predicted_idx = int(probs_np.argmax())
        confidence = float(probs_np.max())
        
        # 5. 获取结果并翻译
        config = CROPS_CONFIG[crop_type]
        english_name = config['classes'][predicted_idx]
        chinese_name = config['translation'].get(english_name, english_name)

        # 6. 光谱分析
        health_score, heatmap_base64 = analyze_spectral_health(original_img)

        return jsonify({
            "success": True,
            "crop_type": crop_type,
            "disease_name_cn": chinese_name,
            "disease_name_en": english_name,
            "confidence": f"{confidence*100:.2f}%",
            "health_score": f"{health_score:.1f}",
            "heatmap_base64": heatmap_base64
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# === 新增：根路由返回前端index.html（关键！）===
@app.route('/')
def index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return send_file(index_file)
    else:
        return f"Frontend file not found: {index_file}", 404

if __name__ == '__main__':
    # 开发环境启动，生产环境用 Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)