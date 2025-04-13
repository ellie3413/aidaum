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

#========== 질문 목록 - 핵심 질문과 직업 포함 ==========
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

    # 진행 상태 표시
    if curr_page < total_q:
        st.progress((curr_page) / total_q)
        st.markdown(f"### 질문 {curr_page + 1}/{total_q}")

    if curr_page < total_q:
        q = questions[curr_page]
        st.markdown(f"#### {q['question']}")
        
        # 도움말 표시
        if "help" in q:
            st.caption(q["help"])
        
        st.markdown("---")

        if q.get("multi"):
            # 기본값 설정 (이전에 응답했던 값이 있으면 유지)
            default_val = st.session_state.responses.get(q["key"], []) if q["key"] in st.session_state.responses else []
            response = st.multiselect("✅ 선택하세요", q["options"], default=default_val, key=f"resp_{q['key']}")
        else:
            # 기본값 설정
            default_idx = 0
            if q["key"] in st.session_state.responses:
                if st.session_state.responses[q["key"]] in q["options"]:
                    default_idx = q["options"].index(st.session_state.responses[q["key"]])
            
            response = st.radio("✅ 선택하세요", q["options"], index=default_idx, key=f"resp_{q['key']}")

        col1, col2, col3 = st.columns([1, 1, 3])

        # 이전 버튼 (첫 번째 질문이 아닌 경우에만)
        if curr_page > 0:
            with col1:
                if st.button("👈 이전", use_container_width=True):
                    st.session_state.page -= 1
                    st.rerun()

        with col2:
            if st.button("👉 다음", use_container_width=True):
                st.session_state.responses[q["key"]] = response
                next_page()


    else:
        st.success("🎉 설문이 완료되었습니다! 당신에게 맞는 AI 도구를 추천해 드립니다.")
        st.session_state.survey_complete = True
        