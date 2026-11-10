import platform
import re
import time
import urllib.parse
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi
from playwright.sync_api import sync_playwright
from transformers import pipeline

#
# 1. 한글 폰트 설정 및 모델 로드
#
def set_korean_font():
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
    kiwi = Kiwi()
    # 단어 및 문장 감정 평가용 허깅페이스 모델 로드
    sentiment_model = pipeline(
        "text-classification",
        model="jaehyeong/koelectra-base-v3-generalized-sentiment-analysis",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    return kiwi, sentiment_model

#
# 2. 브라우저 스크롤 수집 함수
#
def fetch_text_with_scroll(url: str, platform_type: str, scroll_count: int = 5) -> str:
    extracted_texts = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage"
            ]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
            locale="ko-KR"
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        try:
            # 1) 유튜브 수집
            if platform_type == "youtube":
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                time.sleep(2)
                page.evaluate("window.scrollTo(0, 600)")
                time.sleep(1.5)
                for _ in range(scroll_count):
                    page.evaluate("window.scrollBy(0, 1200)")
                    time.sleep(1.2)

                comments = page.query_selector_all("#content-text, yt-formatted-string#content-text")
                for c in comments:
                    t = c.inner_text().strip()
                    if t:
                        extracted_texts.append(t)

            # 2) 네이버 영화/검색 수집
            elif platform_type == "naver":
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                time.sleep(2)

                for _ in range(scroll_count):
                    page.evaluate("window.scrollBy(0, 1000)")
                    time.sleep(0.8)

                soup = BeautifulSoup(page.content(), "html.parser")
                for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    tag.decompose()

                review_elements = soup.select(".area_text_box, .desc_box, .sp_nreview, .review_text, .text._text, .api_txt_lines")
                if review_elements:
                    for el in review_elements:
                        t = el.get_text(strip=True)
                        if len(t) >= 6:
                            extracted_texts.append(t)
                else:
                    main_pack = soup.select_one("#main_pack") or soup.select_one("body")
                    if main_pack:
                        extracted_texts.append(main_pack.get_text(separator=" ", strip=True))

            # 3) 엑스(트위터) 수집
            elif platform_type == "x":
                parsed = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed.query)
                query = qs.get("q", [""])[0]

                if not query:
                    query = url.split("/")[-1].replace("?f=live", "").replace("?f=top", "")

                query_cleaned = urllib.parse.unquote(query).strip()

                # Nitter 공개 인스턴스 시도
                nitter_url = f"https://nitter.poast.org/search?f=tweets&q={urllib.parse.quote(query_cleaned)}"
                try:
                    page.goto(nitter_url, timeout=15000, wait_until="domcontentloaded")
                    time.sleep(2)
                    tweet_nodes = page.query_selector_all(".tweet-content, .tweet-body")
                    for node in tweet_nodes:
                        txt = node.inner_text().strip()
                        if txt:
                            extracted_texts.append(txt)
                except Exception:
                    pass

                # DuckDuckGo 색인 폴백
                if not extracted_texts:
                    ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote('site:x.com ' + query_cleaned)}"
                    try:
                        page.goto(ddg_url, timeout=15000, wait_until="domcontentloaded")
                        time.sleep(1.5)
                        soup_ddg = BeautifulSoup(page.content(), "html.parser")
                        snippets = soup_ddg.select(".result__snippet, .result__title")
                        for snip in snippets:
                            clean_snip = snip.get_text(strip=True)
                            if len(clean_snip) >= 8:
                                extracted_texts.append(clean_snip)
                    except Exception:
                        pass

                # 원래 x.com 본문 시도
                if not extracted_texts:
                    try:
                        page.goto(url, timeout=15000, wait_until="domcontentloaded")
                        time.sleep(2)
                        tweets = page.query_selector_all("[data-testid='tweetText'], article div[lang]")
                        for tw in tweets:
                            t = tw.inner_text().strip()
                            if t:
                                extracted_texts.append(t)
                    except Exception:
                        pass

        except Exception as e:
            try:
                txt = page.inner_text("body")
                if txt:
                    extracted_texts.append(txt)
            except Exception:
                pass
        finally:
            browser.close()

    return " ".join(extracted_texts)

#
# 3. AI 기반 감정 단어 판별 및 추출
#
def extract_sentiment_words_with_ai(kiwi: Kiwi, sentiment_model, text: str, confidence_threshold: float = 0.70) -> list:
    """
    1차: Kiwi로 어휘 후보군(명사, 형용사, 부사, 어근)을 추출
    2차: 사전 필터 대신 AI 모델에 각 단어를 통과시켜 강한 긍정/부정 감성을 띠는 단어만 직접 판별
    """
    if not text or not text.strip():
        return []

    cleaned_text = re.sub(r"[^\w\s가-힣]", " ", text)
    tokens = kiwi.tokenize(cleaned_text)

    # 단순 시스템 UI/기능 단어만 제외
    system_stop_words = {
        "유튜브", "영상", "네이버", "검색", "게시물", "댓글", "트위터", "더보기", "링크",
        "공유", "접기", "사진", "본문", "계정", "가입", "로그인", "스마트블록", "고객센터"
    }

    # 후보 어휘군 수집 (중복 제거 및 빈도 보존)
    candidate_words = []
    seen = set()
    for token in tokens:
        form = token.form.strip()
        # 2글자 이상, 유효 품사 대상
        if len(form) >= 2 and form not in system_stop_words and token.tag in {"NNG", "VA", "MAG", "XR", "IC"}:
            if form not in seen:
                seen.add(form)
                candidate_words.append(form)

    if not candidate_words:
        return []

    # AI 모델 연산 속도를 위해 상위 80개 후보 단어 일괄 배치 평가
    eval_candidates = candidate_words[:80]
    ai_filtered_words = []

    try:
        model_results = sentiment_model(eval_candidates)
        for word, res in zip(eval_candidates, model_results):
            # 가장 높은 확신도를 가진 라벨의 스코어 추출
            top_pred = max(res, key=lambda x: x["score"])
            # AI가 70% 이상의 확신으로 긍정/부정 감정이라고 판정한 어휘만 최종 통과
            if top_pred["score"] >= confidence_threshold:
                ai_filtered_words.append(word)
    except Exception:
        # 배치 추론 에러 시 개별 추론 폴백
        for word in eval_candidates:
            try:
                res = sentiment_model(word)[0]
                top_pred = max(res, key=lambda x: x["score"])
                if top_pred["score"] >= confidence_threshold:
                    ai_filtered_words.append(word)
            except Exception:
                continue

    return ai_filtered_words

#
# 4. 감정 분석 로직 (최종 수치 판정)
#
def analyze_sentiment_with_ai(sentiment_model, extracted_words: list) -> tuple[float, float, str]:
    if not extracted_words:
        return 0.0, 0.0, "판정불가"

    sampled_text = " ".join(extracted_words[:200])
    raw_result = sentiment_model(sampled_text)[0]

    pos_score = 0.0
    neg_score = 0.0
    for item in raw_result:
        label = str(item["label"]).lower()
        if "pos" in label or label == "1" or label == "label_1":
            pos_score = float(item["score"])
        else:
            neg_score = float(item["score"])

    total = pos_score + neg_score
    pos_percent = round((pos_score / total) * 100, 2) if total > 0 else 0.0
    neg_percent = round((neg_score / total) * 100, 2) if total > 0 else 0.0

    verdict = "긍정" if pos_percent > 55 else ("부정" if neg_percent > 55 else "중립")
    return pos_percent, neg_percent, verdict

#
# 5. 수집 파이프라인
#
def run_collection_pipeline(targets: list, kiwi: Kiwi, sentiment_model) -> list:
    results = []
    total = len(targets)
    progress_bar = st.progress(0)

    with st.status("플랫폼별 데이터 수집 및 AI 감정 분석 진행 중...", expanded=True) as status:
        for idx, target in enumerate(targets):
            platform_name = target["platform"]
            platform_type = target["type"]
            url = target["url"]

            st.write(f"현재 수집 중 ({idx+1}/{total}): **{platform_name}**")

            try:
                raw_text = fetch_text_with_scroll(url, platform_type, scroll_count=5)
                # AI 모델이 감정 단어를 직접 심사하여 추출
                words = extract_sentiment_words_with_ai(kiwi, sentiment_model, raw_text, confidence_threshold=0.70)
                word_count = len(words)

                pos_score, neg_score, verdict = analyze_sentiment_with_ai(sentiment_model, words)

                status_label = "성공" if word_count >= 3 else "수집량 부족"
                results.append({
                    "플랫폼": platform_name,
                    "URL": url,
                    "수집상태": status_label,
                    "수집단어수": word_count,
                    "추출단어목록": words[:30],
                    "긍정비율(%)": pos_score,
                    "부정비율(%)": neg_score,
                    "반응판정": verdict if word_count >= 3 else "판정불가",
                })
                st.write(f"✔ **{platform_name}** 완료 | AI 선별 감정 단어: {word_count}개")
            except Exception as e:
                results.append({
                    "플랫폼": platform_name,
                    "URL": url,
                    "수집상태": f"실패 ({str(e)})",
                    "수집단어수": 0,
                    "추출단어목록": [],
                    "긍정비율(%)": 0.0,
                    "부정비율(%)": 0.0,
                    "반응판정": "판정불가",
                })
                st.write(f"✖ **{platform_name}** 에러 발생: {str(e)}")

            progress_bar.progress((idx + 1) / total)

        status.update(label="모든 플랫폼 수집 및 분석 완료!", state="complete", expanded=False)

    return results

#
# 6. 시각화 함수
#
def render_sentiment_pie_charts(data_list: list):
    set_korean_font()
    valid_items = [item for item in data_list if item["수집단어수"] >= 3]
    if not valid_items:
        st.info("유효한 감정 단어가 충분하지 않아 원형 차트를 생성하지 못했습니다.")
        return

    cols = st.columns(len(valid_items))
    colors = ["#4A90E2", "#FF6B6B"]

    for idx, item in enumerate(valid_items):
        with cols[idx]:
            sizes = [item["긍정비율(%)"], item["부정비율(%)"]]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(sizes, labels=["긍정", "부정"], autopct="%1.1f%%", startangle=90, colors=colors,
                   wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            ax.set_title(f"{item['플랫폼']} 반응 분포\n(판정: {item['반응판정']})", fontsize=11)
            st.pyplot(fig)
            plt.close(fig)

def render_sentiment_bar_chart(data_list: list):
    set_korean_font()
    valid_items = [item for item in data_list if item["수집단어수"] >= 3]
    if not valid_items:
        return

    platforms = [item["플랫폼"] for item in valid_items]
    pos_rates = [item["긍정비율(%)"] for item in valid_items]
    neg_rates = [item["부정비율(%)"] for item in valid_items]

    x = np.arange(len(platforms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - width / 2, pos_rates, width, label="긍정", color="#4A90E2")
    rects2 = ax.bar(x + width / 2, neg_rates, width, label="부정", color="#FF6B6B")

    ax.set_ylabel("비율 (%)", fontsize=11)
    ax.set_title("플랫폼별 긍정 vs 부정 반응 비교", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    st.pyplot(fig)
    plt.close(fig)

#
# 7. 메인 UI
#
def main():
    st.set_page_config(page_title="호프 반응 분석 시스템", layout="wide")
    st.title("호프 반응의 플랫폼별 반응 분석")

    kiwi, sentiment_model = load_analysis_tools()

    st.sidebar.subheader("분석 대상 URL 입력")
    yt_url = st.sidebar.text_input("유튜브 URL", placeholder="https://www.youtube.com/watch?v=...")
    naver_url = st.sidebar.text_input("네이버 영화 리뷰 URL", placeholder="https://search.naver.com/search.naver?where=nexearch&query=...")
    x_url = st.sidebar.text_input("엑스 (트위터) URL", placeholder="https://x.com/search?q=...")
    start_button = st.sidebar.button("크롤링 시작")

    if start_button:
        targets = []
        if yt_url and yt_url.strip():
            targets.append({"platform": "유튜브", "type": "youtube", "url": yt_url.strip()})
        if naver_url and naver_url.strip():
            targets.append({"platform": "네이버 영화 리뷰", "type": "naver", "url": naver_url.strip()})
        if x_url and x_url.strip():
            targets.append({"platform": "엑스 (트위터)", "type": "x", "url": x_url.strip()})

        if not targets:
            st.warning("분석할 URL을 하나 이상 입력해 주세요.")
            return

        analysis_data = run_collection_pipeline(targets, kiwi, sentiment_model)
        df_results = pd.DataFrame(analysis_data)

        tab1, tab2 = st.tabs(["감정 분석 결과", "전체 데이터 요약"])

        with tab1:
            st.subheader("1. AI 선별 주요 감정 단어")
            for item in analysis_data:
                if item["수집상태"] == "성공":
                    words_preview = ", ".join(item["추출단어목록"])
                    st.markdown(f"**[{item['플랫폼']}]** (AI 판정 감정 단어 {item['수집단어수']}개)")
                    st.info(words_preview if words_preview else "추출된 감정 단어가 없습니다.")
                else:
                    st.warning(f"**[{item['플랫폼']}]** {item['수집상태']} (수집 단어: {item['수집단어수']}개)")
            st.divider()

            st.subheader("2. AI 감성 점수 판정 결과")
            score_columns = ["플랫폼", "수집단어수", "긍정비율(%)", "부정비율(%)", "반응판정", "수집상태"]
            st.dataframe(df_results[score_columns], use_container_width=True, hide_index=True)
            st.divider()

            st.subheader("3. 플랫폼별 감성 분포 시각화")
            st.markdown("#### 플랫폼별 반응 비교 막대그래프")
            render_sentiment_bar_chart(analysis_data)

            st.markdown("#### 플랫폼별 세부 원형 차트")
            render_sentiment_pie_charts(analysis_data)

        with tab2:
            st.subheader("수집 및 분석 원본 데이터")
            display_df = df_results.copy()
            display_df["추출단어목록"] = display_df["추출단어목록"].apply(lambda x: ", ".join(x))
            st.dataframe(display_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()