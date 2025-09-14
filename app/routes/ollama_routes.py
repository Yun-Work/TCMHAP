# ollama_routes.py
from flask import Blueprint, request, jsonify
import requests
import os
import traceback
import opencc
converter = opencc.OpenCC('s2tw')  # 簡體(s) 轉 繁體(t)

ollama_bp = Blueprint("ollama", __name__)

# ===== 可調整參數 =====
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://163.13.202.117:11434")  # 你的 Ollama 伺服器
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "taozhiyuai/openbiollm-llama-3:8b_q8_0")  # 模型名
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # 秒

SYSTEM_PROMPT = (
    "你是一個中醫小助手，請嚴格遵守以下規則：\n"
    "1. 嚴格使用繁體中文回答。\n"
    "2. 僅能根據中醫理論（臟腑、經絡、五行、面診、養生）作答，不得加入與中醫無關的內容。\n"
    "3. 必須緊扣使用者的問題作答，不可偏題或自行發揮。\n"
    "4. 請以條列式或分段方式呈現，最後加入【統整】簡短總結。\n"
    "5. 禁止虛構人物或與問題無關的情境。\n"
)

# 健檢：確認 Flask -> Ollama 有通
@ollama_bp.route("/ollama/health", methods=["GET"])
def health():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        r.raise_for_status()
        tags = r.json().get("models", [])
        # 回傳目前是否找到指定模型
        has_model = any(m.get("name") == OLLAMA_MODEL for m in tags)
        return jsonify(ok=True, base=OLLAMA_BASE, model=OLLAMA_MODEL, has_model=has_model, tags_count=len(tags))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

# 簡單 echo：排除 App 傳輸問題
@ollama_bp.route("/ollama/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify(ok=True, got=data)

# 主要：轉呼叫 Ollama /api/chat
@ollama_bp.route("/ollama/ask", methods=["POST"])
def ask_ollama():
    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify(success=False, message="缺少 question"), 400

        body = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            "stream": False,
        }

        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=body, timeout=TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        # 期望格式：{"message":{"role":"assistant","content":"..."}}
        content = (data.get("message") or {}).get("content", "").strip()

        # 如果是 /api/generate 走法，會是 "response"
        if not content:
            content = data.get("response", "").strip()

        if not content:
            return jsonify(success=False, message="Ollama 回傳內容為空", raw=data), 502

        # ✅ 強制轉繁體
        content = converter.convert(content)

        return jsonify(success=True, answer=content)

    except requests.exceptions.RequestException as re:
        return jsonify(success=False, message=f"Ollama 連線錯誤：{str(re)}"), 502
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify(success=False, message=str(e), traceback=tb), 500