from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 이 파일(fonts.py) 기준으로 fonts/ 폴더 안의 폰트 파일 경로를 찾음
FONT_PATH = Path(__file__).parent / "fonts" / "NanumGothic-Regular.ttf"

if FONT_PATH.exists():
    # 1) 폰트 파일을 matplotlib의 폰트 매니저에 직접 등록
    fm.fontManager.addfont(str(FONT_PATH))
    # 2) 등록한 폰트의 실제 이름을 가져와서 사용 폰트로 지정
    font_name = fm.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["font.family"] = font_name
else:
    # 폰트 파일이 없을 때를 대비한 기존 방식 (로컬 개발 시 폴백용)
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    else:
        print(
            "[WARN] 한글 폰트를 찾지 못했습니다. fonts/NanumGothic.ttf를 추가해주세요."
        )

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
