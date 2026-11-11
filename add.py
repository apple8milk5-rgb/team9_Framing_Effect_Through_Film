import platform
import re
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from kiwipiepy import Kiwi
from transformers import pipeline


# ----------------------------------------------------
# 1. 폰트 설정 및 모델 초기화
# ----------------------------------------------------
def set_korean_font():
    """OS 환경에 맞춰 Matplotlib 한글 폰트 및 마이너스 부호 설정"""
    os_name = platform.system()
    if os_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif os_name == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource
def load_sentiment_model():
    """공개 한국어 감정분석 사전학습 모델 로드"""
    model_name = "matthewburke/korean_sentiment"
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=128,
    )
    return sentiment_classifier


@st.cache_resource
def load_kiwi():
    """단어 추출을 위한 Kiwipiepy 형태소 분석기 로드"""
    return Kiwi()


# ----------------------------------------------------
# 2. 데이터 수집 함수 (50개 수집 및 작성자 익명화)
# ----------------------------------------------------
def extract_video_id(url: str) -> str:
    """유튜브 URL에서 비디오 ID 추출"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def get_video_comments(
    api_key: str, video_id: str, max_results: int = 50
) -> list:
    """YouTube Data API v3를 호출하여 댓글 수집 및 작성자 user_001 형태로 익명화"""
    youtube = build("youtube", "v3", developerKey=api_key)
    response = (
        youtube.commentThreads()
        .list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance",
        )
        .execute()
    )
    comments = []
    # user_001, user_002 형식으로 순차적 익명화
    for idx, item in enumerate(response.get("items", []), start=1):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append(
            {
                "작성자": f"user_{idx:03d}",
                "댓글 내용": snippet.get("textDisplay", ""),
                "좋아요 수": snippet.get("likeCount", 0),
                "작성일": snippet.get("publishedAt", ""),
            }
        )
    return comments


# ----------------------------------------------------
# 3. AI 모델 감정 분석 및 단어 추출 함수
# ----------------------------------------------------
def extract_keywords_from_comment(text: str, kiwi: Kiwi) -> list:
    """댓글에서 주요 의미를 가진 명사, 형용사, 동사 어간 추출"""
    tokens = kiwi.tokenize(text)
    # NNG(일반명사), NNP(고유명사), VA(형용사), VV(동사), XR(어근)
    valid_tags = {"NNG", "NNP", "VA", "VV", "XR"}
    words = [
        token.form
        for token in tokens
        if token.tag in valid_tags and len(token.form) > 1
    ]
    return words


def analyze_dataframe_sentiment(
    df: pd.DataFrame, classifier, kiwi: Kiwi
) -> pd.DataFrame:
    """영화 리뷰 특화 AI 모델을 통해 댓글 감정 판별 및 점수 계산"""
    sentiments = []
    scores = []
    extracted_words_list = []

    for comment in df["댓글 내용"]:
        # AI 모델 추론
        result = classifier(comment)[0]
        label_raw = result["label"].upper()
        confidence = result["score"]

        # matthewburke/korean_sentiment 기준 (LABEL_1: 긍정, LABEL_0: 부정)
        if label_raw in ["LABEL_1", "POSITIVE", "1"]:
            label = "긍정"
            sentiment_score = round(confidence * 100, 2)
        else:
            label = "부정"
            sentiment_score = round(-confidence * 100, 2)

        # 주요 형태소 단어 추출
        keywords = extract_keywords_from_comment(comment, kiwi)

        sentiments.append(label)
        scores.append(sentiment_score)
        extracted_words_list.append(", ".join(keywords) if keywords else "-")

    df["감정"] = sentiments
    df["감성 점수"] = scores
    df["추출 단어"] = extracted_words_list
    return df


# ----------------------------------------------------
# 4. 시각화 함수
# ----------------------------------------------------
def draw_sentiment_pie(df: pd.DataFrame, title: str):
    """영상별 감정 비율 원형 그래프 시각화"""
    counts = df["감정"].value_counts()
    pos_count = counts.get("긍정", 0)
    neg_count = counts.get("부정", 0)

    if pos_count == 0 and neg_count == 0:
        st.write("표시할 감정 데이터가 없습니다.")
        return

    labels = ["긍정", "부정"]
    sizes = [pos_count, neg_count]
    colors = ["#4CAF50", "#F44336"]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------
# 5. 메인 UI
# ----------------------------------------------------
def main():
    set_korean_font()
    st.title("영화 호프로 알아보는 프레이밍 효과")

    classifier = load_sentiment_model()
    kiwi = load_kiwi()

    api_key = st.text_input(
        "YouTube Data API v3 키를 입력하세요", type="password"
    )

    st.subheader("분석할 영상 URL 입력")
    url_1 = st.text_input(
        "영상 1 URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_1",
    )
    url_2 = st.text_input(
        "영상 2 URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_2",
    )
    url_3 = st.text_input(
        "영상 3 URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_3",
    )
    urls = [url_1, url_2, url_3]

    if st.button("분석 시작"):
        if not api_key:
            st.error("API 키를 입력해야 합니다.")
            return

        valid_targets = []
        for idx, url in enumerate(urls, start=1):
            if url.strip():
                vid = extract_video_id(url.strip())
                if vid:
                    valid_targets.append(
                        {"index": idx, "url": url.strip(), "video_id": vid}
                    )
                else:
                    st.warning(f"영상 {idx}의 비디오 ID를 확인할 수 없습니다.")

        if not valid_targets:
            st.error("유효한 유튜브 URL이 없습니다.")
            return

        progress_bar = st.progress(0.0)
        total_targets = len(valid_targets)
        collected_data = {}
        total_accumulated = 0

        with st.status("데이터 처리 진행 중...", expanded=True) as status:
            for i, target in enumerate(valid_targets):
                idx = target["index"]
                vid = target["video_id"]
                status.write(
                    f"현재 처리 중인 주제: 영상 {idx} (ID: {vid}) 수집 및 분석 시작"
                )

                try:
                    # 각 영상당 최대 50개 댓글 수집
                    comments = get_video_comments(
                        api_key, vid, max_results=50
                    )
                    count = len(comments)
                    total_accumulated += count

                    if count > 0:
                        raw_df = pd.DataFrame(comments)
                        analyzed_df = analyze_dataframe_sentiment(
                            raw_df, classifier, kiwi
                        )
                        collected_data[f"영상 {idx}"] = analyzed_df

                    status.write(
                        f"영상 {idx} 완료: 수집 건수 {count}개 | 누적 수집 개수: {total_accumulated}개"
                    )
                except HttpError as e:
                    status.write(
                        f"영상 {idx} 오류 (API 에러: {e.resp.status}) | 누적 개수: {total_accumulated}개"
                    )
                except Exception as e:
                    status.write(
                        f"영상 {idx} 오류 (원인: {str(e)}) | 누적 개수: {total_accumulated}개"
                    )

                progress_bar.progress((i + 1) / total_targets)

            status.update(
                label=f"작업 완료 (총 {total_accumulated}개 수집 및 분석 완료)",
                state="complete",
                expanded=False,
            )

        if collected_data:
            tab1, tab2 = st.tabs(["감정 분석 결과", "수집 데이터 원본"])

            with tab1:
                # 1. 감성 단어 영역 (각 영상별 상위 20개만 미리보기 출력)
                st.subheader("1. 본문 수집 단어 및 감정 판별 (상위 20개 미리보기)")
                for label, df in collected_data.items():
                    st.write(f"[{label}] (수집된 {len(df)}개 중 20개 표시)")
                    st.dataframe(
                        df[["댓글 내용", "추출 단어", "감정", "감성 점수"]].head(20),
                        use_container_width=True,
                    )

                # 2. 감성 점수 계산 영역 (수집된 50개 전체 기준 통계)
                st.subheader("2. 감성 점수 계산 요약 (전체 데이터 기준)")
                summary_list = []
                for label, df in collected_data.items():
                    summary_list.append(
                        {
                            "영상": label,
                            "수집 댓글 수": len(df),
                            "평균 감성 점수": round(df["감성 점수"].mean(), 2),
                            "긍정 댓글 수": (df["감정"] == "긍정").sum(),
                            "부정 댓글 수": (df["감정"] == "부정").sum(),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(summary_list), use_container_width=True
                )

                # 3. 시각화 영역 (전체 댓글 기준 원형 그래프)
                st.subheader("3. 영상별 감정 분석 원형 그래프")
                cols = st.columns(len(collected_data))
                for col, (label, df) in zip(cols, collected_data.items()):
                    with col:
                        draw_sentiment_pie(df, f"{label} 감정 비율")

            with tab2:
                # 원본 탭에서도 20개 미리보기 표기 (작성자 컬럼은 user_001 형식으로 표시)
                for label, df in collected_data.items():
                    st.write(f"[{label}] 원본 데이터 (상위 20개 미리보기)")
                    st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()