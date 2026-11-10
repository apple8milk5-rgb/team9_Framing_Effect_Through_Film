import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

# 기본 긍정/부정 감정 사전 (프로젝트 기준에 맞춰 확장 가능)
POSITIVE_WORDS = ["재미", "최고", "감동", "추천", "꿀잼", "명작", "좋다", "대박", "흥미", "완벽"]
NEGATIVE_WORDS = ["지루", "최악", "실망", "노잼", "아쉽", "별로", "비추", "시간낭비", "어색", "망작"]


def extract_sentiment_words(text: str) -> dict:
    """텍스트에서 정의된 긍정/부정 감정 단어를 추출하고 개수를 계산하는 함수"""
    pos_found = [word for word in POSITIVE_WORDS if word in text]
    neg_found = [word for word in NEGATIVE_WORDS if word in text]

    sentiment = "중립"
    if len(pos_found) > len(neg_found):
        sentiment = "긍정"
    elif len(neg_found) > len(pos_found):
        sentiment = "부정"

    return {
        "긍정단어": ", ".join(set(pos_found)) if pos_found else "-",
        "부정단어": ", ".join(set(neg_found)) if neg_found else "-",
        "전체단어수": len(pos_found) + len(neg_found),
        "판정": sentiment,
    }


def crawl_platform(platform_name: str, url: str) -> dict:
    """플랫폼 URL에 접근하여 텍스트를 수집하고 감정 단어를 분석하는 함수"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        # 스크립트 및 스타일 태그 제거 후 본문 텍스트 추출
        for tag in soup(["script", "style", "header", "footer"]):
            tag.decompose()
        clean_text = soup.get_text(separator=" ", strip=True)

        # 감정 단어 분석 수행
        analysis = extract_sentiment_words(clean_text)

        return {
            "플랫폼": platform_name,
            "URL": url,
            "수집상태": "성공",
            "수집단어수": analysis["전체단어수"],
            "긍정단어": analysis["긍정단어"],
            "부정단어": analysis["부정단어"],
            "반응판정": analysis["판정"],
        }
    except Exception as e:
        return {
            "플랫폼": platform_name,
            "URL": url,
            "수집상태": f"실패 ({str(e)})",
            "수집단어수": 0,
            "긍정단어": "-",
            "부정단어": "-",
            "반응판정": "판정불가",
        }


def run_analysis_pipeline(targets: list) -> pd.DataFrame:
    """진행 상황(st.status, st.progress)을 갱신하며 플랫폼별 크롤링을 수행하는 함수"""
    results = []
    total_targets = len(targets)
    accumulated_count = 0

    progress_bar = st.progress(0)

    with st.status("분석 진행 중...", expanded=True) as status:
        for idx, target in enumerate(targets):
            platform = target["platform"]
            url = target["url"]

            # 현재 처리 중인 플랫폼 알림
            st.write(f"현재 처리 중: {platform}")

            # 크롤링 및 분석 실행
            res = crawl_platform(platform, url)
            results.append(res)

            accumulated_count += res["수집단어수"]

            # 성공/실패 여부 및 누적 개수 출력
            if "성공" in res["수집상태"]:
                st.write(
                    f"- {platform} 수집 성공 | 누적 수집 단어 수: {accumulated_count}개"
                )
            else:
                st.write(f"- {platform} 수집 실패 | 이유: {res['수집상태']}")

            # 진행률 업데이트
            progress_bar.progress((idx + 1) / total_targets)
            time.sleep(0.5)

        status.update(label="분석 완료", state="complete", expanded=False)

    return pd.DataFrame(results)


def main():
    st.title("호프 반응의 플랫폼별 반응 분석")

    # 1. URL 입력 인터페이스
    st.subheader("분석 대상 URL 입력")
    yt_url = st.text_input("유튜브 URL", placeholder="https://www.youtube.com/...")
    naver_url = st.text_input(
        "네이버 영화 리뷰 URL", placeholder="https://movie.naver.com/..."
    )
    coupang_url = st.text_input(
        "쿠팡플레이 URL", placeholder="https://www.coupangplay.com/..."
    )

    # 2. 크롤링 시작 버튼
    if st.button("크롤링 시작"):
        # 입력 데이터 목록화
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
            st.warning("최소 1개 이상의 URL을 입력해주세요.")
            return

        # 3 ~ 4. 크롤링 및 실시간 진행 상태 출력
        df_result = run_analysis_pipeline(targets)

        # 5. DataFrame 결과 테이블 출력
        st.subheader("플랫폼별 반응 분석 결과")
        st.dataframe(df_result, use_container_width=True)


if __name__ == "__main__":
    main()