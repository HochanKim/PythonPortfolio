import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

names = {f.name for f in fm.fontManager.ttflist}
for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
    else:
        print("[WARN] 한글 폰트를 찾지 못했습니다.")
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
