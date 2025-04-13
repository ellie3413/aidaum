# survey.py

import streamlit as st

#========== 상태 초기화 ==========
def init_state():
    if "page" not in st.session_state:
        st.session_state.page = 0
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "survey_complete" not in st.session_state:
        st.session_state.survey_complete = False

#========== 페이지 전환 함수 ==========
def next_page():
    st.session_state.page += 1
    st.rerun()

def reset_survey():
    st.session_state.page = 0
    st.session_state.responses = {}
    st.session_state.survey_complete = False
    st.rerun()

#========== 질문 목록 ==========
questions = [
    {
        "question": "현재 AI에 대해 얼마나 알고 계신가요?",
        "options": ["전혀 모른다", "이름만 들어봤다", "기본 개념은 알고 있다", "실제로 활용해본 경험이 있다", "AI 모델이나 알고리즘을 직접 다뤄본 적 있다"],
        "key": "ai_knowledge",
        "help": "이 정보는 적절한 난이도의 도구를 추천하는 데 활용됩니다."
    },
    {
        "question": "귀하의 직업 또는 현재 활동 분야는 무엇인가요?",
        "options": ["학생", "개발자/IT 종사자", "교육자/연구원", "디자이너/창작자", "마케터/홍보", "사무직", "경영/관리자", "창업가/프리랜서", "의료/건강 종사자", "법률/금융 전문가", "기타"],
        "key": "job",
        "help": "직업 분야에 맞는 특화된 AI 도구를 추천해 드립니다."
    },
    {
        "question": "어떤 종류의 AI 도구에 관심이 있으신가요? (여러 개 선택 가능)",
        "options": ["텍스트 생성", "이미지 생성", "영상/음성 합성", "데이터 분석 및 시각화", "업무 자동화", "검색 및 지식 관리", "코드 생성 및 개발 지원", "번역 및 언어 학습", "기타"],
        "key": "tool_interest",
        "multi": True,
        "help": "관심 있는 도구 유형을 알려주시면 해당 카테고리의 도구를 우선적으로 추천해 드립니다."
    },
    {
        "question": "구체적으로 어떤 작업에 AI 도구를 활용하고 싶으신가요? (여러 개 선택 가능)",
        "options": ["문서 작성 및 편집", "이미지/영상 제작", "데이터 분석", "프로그래밍 및 개발", "마케팅 및 홍보", "교육 및 학습", "업무 자동화", "고객 서비스", "연구 및 논문 작성", "기타"],
        "key": "specific_purpose",
        "multi": True,
        "help": "구체적인 용도를 알려주시면 더 정확한 도구를 추천해 드립니다."
    },
    {
        "question": "선호하는 AI 도구의 난이도는 어느 정도인가요?",
        "options": ["쉬움 (초보자도 바로 사용 가능한 도구)", "중간 (기본적인 지식이 필요한 도구)", "어려움 (전문적인 지식이 필요한 고급 도구)", "난이도보다는 기능 중심으로 선택하고 싶음"],
        "key": "preferred_difficulty",
        "help": "선호하는 난이도를 알려주시면 해당 수준에 맞는 도구를 우선적으로 추천해 드립니다."
    }
]

#========== 설문 실행 함수 ==========
def run_survey():
    init_state()
    total_q = len(questions)
    curr_page = st.session_state.page

    st.markdown("### 🤖 AI 리터러시 진단 설문")
    st.progress(curr_page / total_q)

    if curr_page < total_q:
        q = questions[curr_page]
        st.markdown(f"#### Q{curr_page + 1}. {q['question']}")
        st.markdown("---")

        if q.get("multi"):
            response = st.multiselect("✅ 선택하세요", q["options"], key=f"resp_{q['key']}")
        else:
            response = st.radio("✅ 선택하세요", q["options"], key=f"resp_{q['key']}")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("👉 다음", use_container_width=True):
                st.session_state.responses[q["key"]] = response
                next_page()
        
        # 설문 진행 상태 표시
        st.caption(f"{curr_page + 1} / {total_q} 질문")

    else:
        st.success("🎉 설문이 완료되었습니다! 감사합니다.")
        st.session_state.survey_complete = True
        
        # 응답 결과 요약 시각화
        st.markdown("#### 📊 당신의 응답 결과 요약")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"💡 **AI 지식 수준**: {st.session_state.responses.get('ai_knowledge', '-')}")
            st.info(f"💡 **주 사용 목적**: {st.session_state.responses.get('purpose', '-')}")
            
        with col2:
            st.info(f"💡 **직업/활동**: {st.session_state.responses.get('job', '-')}")
            st.info(f"💡 **AI 도구 사용 빈도**: {st.session_state.responses.get('ai_tool_usage', '-')}")
        
        # 선택적으로 전체 상세 결과 보기
        with st.expander("전체 응답 상세 결과", expanded=False):
            st.json(st.session_state.responses)
        
        st.markdown("---")
        st.button("🔄 처음부터 다시 하기", on_click=reset_survey)
