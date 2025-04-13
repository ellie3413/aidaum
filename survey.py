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
        "key": "ai_knowledge"
    },
    {
        "question": "주로 어떤 목적에서 AI 기술을 배우고 싶으신가요?",
        "options": ["업무에 활용하고 싶다", "학업/연구에 활용하고 싶다", "개인 프로젝트나 창작에 사용하고 싶다", "기본 개념부터 알고 싶다", "그냥 흥미/호기심 때문에"],
        "key": "purpose"
    },
    {
        "question": "구체적으로 어떤 업무/활동에 AI 도구를 활용하고 싶으신가요?",
        "options": ["문서 작성 및 편집", "이미지/영상 제작", "데이터 분석", "프로그래밍 및 개발", "마케팅 및 홍보", "교육 및 학습", "업무 자동화", "고객 서비스", "연구 및 논문 작성", "기타"],
        "key": "specific_purpose",
        "multi": True
    },
    {
        "question": "귀하의 직업 또는 현재 활동은 무엇인가요?",
        "options": ["학생", "취업 준비 중", "사무직", "개발자/IT 종사자", "창업가/프리랜서", "교육자", "연구원", "디자이너/창작자", "마케터", "기타"],
        "key": "job"
    },
    {
        "question": "현재 AI 도구를 얼마나 사용하고 계신가요?",
        "options": ["전혀 사용해본 적 없다", "가끔 사용한다 (주 1회 미만)", "자주 사용한다 (주 1회 이상)", "일상적으로 사용한다 (거의 매일)"],
        "key": "ai_tool_usage"
    },
    {
        "question": "어떤 종류의 AI 도구에 관심이 있으신가요?",
        "options": ["텍스트 생성", "이미지 생성", "영상/음성 합성", "데이터 분석 및 시각화", "업무 자동화", "검색 및 지식 관리", "코드 생성 및 개발 지원", "번역 및 언어 학습", "기타"],
        "key": "tool_interest",
        "multi": True
    },
    {
        "question": "선호하는 AI 도구의 난이도는 어느 정도인가요?",
        "options": ["쉬움 (초보자도 바로 사용 가능한 도구)", "중간 (기본적인 지식이 필요한 도구)", "어려움 (전문적인 지식이 필요한 고급 도구)", "난이도보다는 기능 중심으로 선택하고 싶음"],
        "key": "preferred_difficulty"
    },
    {
        "question": "AI 학습에서 가장 필요한 것이 무엇이라고 생각하시나요?",
        "options": ["개념 정리 및 기초 이론", "실습 예제 및 실전 활용법", "다양한 도구 소개 및 사용법", "나에게 맞는 추천 학습 경로", "함께 배우는 커뮤니티 공간"],
        "key": "learning_need"
    },
    {
        "question": "주로 어떤 플랫폼/기기에서 AI 도구를 사용하시나요?",
        "options": ["PC/노트북", "모바일 기기", "웹 브라우저", "특정 소프트웨어 내", "모든 플랫폼"],
        "key": "platform",
        "multi": True
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
