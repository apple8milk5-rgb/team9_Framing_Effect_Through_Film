import platform
import re
import time
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi
from transformers import pipeline


# -------------------------------------------------------------
# 1. 환경 설정 및 모델/도구 로드
# -------------------------------------------------------------
def set_korean_font():
    """운영체제별 한글 폰트 설정으로 시각화 시 한글 깨짐 방지"""
    system_name = platform.system()
    if system_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif system_name == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource
def load_analysis_tools():
    """형태소 분석기(Kiwi) 및 네이버 영화 리뷰 특화 감정분석 모델 로드"""
    kiwi = Kiwi()
    # NSMC(네이버 영화 리뷰 데이터셋) 기반 fine-tuned 모델 사용
    sentiment_model = pipeline(
        "text-classification",
        model="daekeun-choi/koelectra-small-v3-nsmc",
        top_k=None,
    )
    return kiwi, sentiment_model


# -------------------------------------------------------------
# 2. 데이터 수집 및 텍스트 정제
# -------------------------------------------------------------
def fetch_text_from_url(url: str) -> str:
    """URL에 접근하여 HTML 태그를 제거하고 순수 본문 텍스트만 추출"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def extract_keywords_kiwi(kiwi: Kiwi, text: str) -> list:
    """kiwipiepy를 사용하여 본문에서 감정 표현에 유효한 품사(명사, 형용사, 동사) 단어 추출"""
    tokens = kiwi.tokenize(text)
    # N(일반명사 등), V(동사/형용사 등), XR(어근) 위주 필터링 (2글자 이상)
    valid_tags = {"NNG", "NNP", "VV", "VA", "XR"}
    extracted_words = [
        token.form
        for token in tokens
        if token.tag in valid_tags and len(token.form) > 1
    ]
    return extracted_words


# -------------------------------------------------------------
# 3. 감정 분석 로직
# -------------------------------------------------------------
def analyze_sentiment_with_ai(
    sentiment_model, extracted_words: list
) -> tuple[float, float, str]:
    """수집된 단어들을 묶어 AI 모델로 전달하고 긍정/부정 확률 계산"""
    if not extracted_words:
        return 0.0, 0.0, "판정불가"

    # 상위 150개 단어를 조합하여 입력 문장 생성 (모델 최대 토큰 길이 고려)
    sampled_text = " ".join(extracted_words[:150])
    raw_result = sentiment_model(sampled_text)[0]

    pos_score = 0.0
    neg_score = 0.0

    for item in raw_result:
        label = item["label"].lower()
        if "pos" in label or label == "1":
            pos_score = item["score"]
        else:
            neg_score = item["score"]

    total = pos_score + neg_score
    pos_percent = round((pos_score / total) * 100, 2) if total > 0 else 0.0
    neg_percent = round((neg_score / total) * 100, 2) if total > 0 else 0.0

    verdict = (
        "긍정"
        if pos_percent > 55
        else ("부정" if neg_percent > 55 else "중립")
    )
    return pos_percent, neg_percent, verdict


# -------------------------------------------------------------
# 4. 파이프라인 및 시각화
# -------------------------------------------------------------
def run_collection_pipeline(
    targets: list, kiwi: Kiwi, sentiment_model
) -> list:
    """URL별 크롤링, 형태소 추출, 감정 분석을 실행하고 상태를 업데이트"""
    results = []
    total = len(targets)
    accumulated_words = 0

    progress_bar = st.progress(0)

    with st.status("플랫폼별 데이터 수집 및 분석 진행 중...", expanded=True) as status:
        for idx, target in enumerate(targets):
            platform_name = target["platform"]
            url = target["url"]
            st.write(f"현재 처리 중인 플랫폼: {platform_name}")

            try:
                raw_text = fetch_text_from_url(url)
                words = extract_keywords_kiwi(kiwi, raw_text)
                word_count = len(words)
                accumulated_words += word_count

                pos_score, neg_score, verdict = analyze_sentiment_with_ai(
                    sentiment_model, words
                )

                results.append(
                    {
                        "플랫폼": platform_name,
                        "URL": url,
                        "수집상태": "성공",
                        "수집단어수": word_count,
                        "추출단어목록": words[:30],  # 화면 표시용 상위 30개
                        "긍정비율(%)": pos_score,
                        "부정비율(%)": neg_score,
                        "반응판정": verdict,
                    }
                )
                st.write(
                    f"- {platform_name} 완료 | 누적 단어: {accumulated_words}개 수집"
                )
            except Exception as e:
                results.append(
                    {
                        "플랫폼": platform_name,
                        "URL": url,
                        "수집상태": f"실패 ({str(e)})",
                        "수집단어수": 0,
                        "추출단어목록": [],
                        "긍정비율(%)": 0.0,
                        "부정비율(%)": 0.0,
                        "반응판정": "판정불가",
                    }
                )
                st.write(f"- {platform_name} 실패: {str(e)}")

            progress_bar.progress((idx + 1) / total)
            time.sleep(0.3)

        status.update(label="전체 작업 완료", state="complete", expanded=False)

    return results


def render_sentiment_pie_charts(data_list: list):
    """사이트별 긍정/부정 비율을 개별 원형 차트로 시각화"""
    set_korean_font()

    valid_items = [
        item for item in data_list if item["수집상태"] == "성공"
    ]
    if not valid_items:
        st.write("시각화할 성공 데이터가 없습니다.")
        return

    cols = st.columns(len(valid_items))
    colors = ["#4A90E2", "#FF6B6B"]

    for idx, item in enumerate(valid_items):
        with cols[idx]:
            labels = ["긍정", "부정"]
            sizes = [item["긍정비율(%)"], item["부정비율(%)"]]

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            )
            ax.set_title(
                f"{item['플랫폼']} 반응 분포\n(판정: {item['반응판정']})",
                fontsize=12,
            )
            st.pyplot(fig)
            plt.close(fig)


# -------------------------------------------------------------
# 5. 메인 UI 화면
# -------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="호프 반응 분석 시스템", layout="wide"
    )
    st.title("호프 반응의 플랫폼별 반응 분석")

    kiwi, sentiment_model = load_analysis_tools()

    # 사이드바: 입력 제어부
    st.sidebar.subheader("분석 대상 URL 입력")
    yt_url = st.sidebar.text_input(
        "유튜브 URL", placeholder="https://www.youtube.com/..."
    )
    naver_url = st.sidebar.text_input(
        "네이버 영화 리뷰 URL", placeholder="https://movie.naver.com/..."
    )
    coupang_url = st.sidebar.text_input(
        "쿠팡플레이 URL", placeholder="https://www.coupangplay.com/..."
    )
    start_button = st.sidebar.button("크롤링 시작")

    if start_button:
        targets = []
        if yt_url.strip():
            targets.append({"platform": "유튜브", "url": yt_url.strip()})
        if naver_url.strip():
            targets.append(
                {"platform": "네이버 영화 리뷰", "url": naver_url.strip()}
            )
        if coupang_url.strip():
            targets.append({"platform": "쿠팡플레이", "url": coupang_url.strip()})

        if not targets:
            st.warning("분석할 URL을 하나 이상 입력하세요.")
            return

        # 크롤링 및 AI 분석 실행
        analysis_data = run_collection_pipeline(targets, kiwi, sentiment_model)
        df_results = pd.DataFrame(analysis_data)

        # 탭 구성
        tab1, tab2 = st.tabs(["감정 분석 결과", "전체 데이터 요약"])

        # 탭 1 배치: 감성 단어 영역 -> 감성 점수 계산 -> 시각화
        with tab1:
            # 1. 감성 단어 영역
            st.subheader("1. 본문 수집 감성 주요 단어")
            for item in analysis_data:
                if item["수집상태"] == "성공":
                    words_preview = ", ".join(item["추출단어목록"])
                    st.markdown(
                        f"**[{item['플랫폼']}]** (총 {item['수집단어수']}개 단어 추출됨)"
                    )
                    st.info(
                        words_preview
                        if words_preview
                        else "추출된 단어가 없습니다."
                    )
                else:
                    st.error(f"[{item['플랫폼']}] 수집 실패로 단어 표시 불가")

            st.divider()

            # 2. 감성 점수 계산 영역
            st.subheader("2. AI 감성 점수 판정 결과")
            score_columns = [
                "플랫폼",
                "수집단어수",
                "긍정비율(%)",
                "부정비율(%)",
                "반응판정",
                "수집상태",
            ]
            st.dataframe(
                df_results[score_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # 3. 시각화 영역 (원형 그래프 개별 생성)
            st.subheader("3. 플랫폼별 감성 분포 시각화")
            render_sentiment_pie_charts(analysis_data)

        with tab2:
            st.subheader("수집 및 분석 원본 데이터")
            display_df = df_results.copy()
            display_df["추출단어목록"] = display_df["추출단어목록"].apply(
                lambda x: ", ".join(x)
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()