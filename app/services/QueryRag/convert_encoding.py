# convert_encoding.py
import chardet

path = "data/face_map.md"
raw = open(path, "rb").read()

det = chardet.detect(raw)
print("Detected encoding:", det)

# 嘗試用偵測到的編碼轉成 UTF-8
with open(path, "r", encoding=det["encoding"], errors="ignore") as f:
    content = f.read()

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("已轉存為 UTF-8！")
