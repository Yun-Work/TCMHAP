import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

#標楷體
rcParams['font.sans-serif'] = ['DFKai-SB']
rcParams['axes.unicode_minus'] = False

# === 會從資料庫輸入過來的資料（總和應為100） ===
# 每個項目：[發白%, 發黑%, 發紅%]
data_list = [
    [30, 50, 20],  # 先假設對應小腸
    [10, 20, 70],  # 先假設對應心臟
    [40, 30, 30]   # 先假設對應肝臟
]

labels = ["小腸", "心臟", "肝臟"]

# 取出各色資料
white = [d[0] for d in data_list]
black = [d[1] for d in data_list]
red = [d[2] for d in data_list]

x = np.arange(len(labels))

# === 統計：有發現過的次數 ===
white_count = sum(1 for w in white if w > 0)
black_count = sum(1 for b in black if b > 0)
red_count = sum(1 for r in red if r > 0)

# === 繪圖 ===
plt.figure(figsize=(9, 7))
plt.bar(x, white, label='發白', color='#FFFFFF', edgecolor='gray')
plt.bar(x, black, bottom=white, label='發黑', color='#333333')
plt.bar(x, red, bottom=np.array(white) + np.array(black), label='發紅', color='#AE0000')

# 標題與座標
plt.xlabel("器官")
plt.ylabel("百分比 (%)")
plt.title("各器官區域的發白 / 發黑 / 發紅 分佈（總和為100%）")
plt.xticks(x, labels)
plt.ylim(0, 100)
plt.legend()

# 顯示症狀出現次數
summary_text = f"偵測結果：發白：{white_count} 次，發黑：{black_count} 次，發紅：{red_count} 次"
plt.figtext(0.5, 0.01, summary_text, wrap=True, horizontalalignment='center', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])


# 儲存圖檔
plt.savefig("output_chart.png", dpi=300, bbox_inches='tight')  # 儲存為高解析度圖片
plt.show()

