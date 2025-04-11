import streamlit as st


#=========설문조사===========
# 페이지 번호 초기화
if "page" not in st.session_state:
    st.session_state.page = 0

# 사용자 응답 저장용
if "responses" not in st.session_state:
    st.session_state.responses = {}

# 페이지 전환 함수
def next_page():
    st.session_state.page += 1

# 질문 목록
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
        "question": "귀하의 직업 또는 현재 활동은 무엇인가요?",
        "options": ["학생", "취업 준비 중", "사무직", "개발자/IT 종사자", "창업가/프리랜서", "기타"],
        "key": "job"
    },
    {
        "question": "현재 AI 도구를 얼마나 사용하고 계신가요?",
        "options": ["전혀 사용해본 적 없다", "가끔 사용한다 (주 1회 미만)", "자주 사용한다 (주 1회 이상)", "일상적으로 사용한다 (거의 매일)"],
        "key": "ai_tool_usage"
    },
    {
        "question": "어떤 종류의 AI 도구에 관심이 있으신가요?",
        "options": ["텍스트 생성", "이미지 생성", "영상/음성 합성", "데이터 분석 및 시각화", "업무 자동화", "기타"],
        "key": "tool_interest",
        "multi": True
    },
    {
        "question": "AI 학습에서 가장 필요한 것이 무엇이라고 생각하시나요?",
        "options": ["개념 정리 및 기초 이론", "실습 예제 및 실전 활용법", "다양한 도구 소개 및 사용법", "나에게 맞는 추천 학습 경로", "함께 배우는 커뮤니티 공간"],
        "key": "learning_need"
    }
]

# 설문 문항 출력
if st.session_state.page < len(questions):
    q = questions[st.session_state.page]
    st.subheader(f"Q{st.session_state.page + 1}. {q['question']}")
    
    if q.get("multi"):
        response = st.multiselect("선택하세요", q["options"], key=q["key"])
    else:
        response = st.radio("선택하세요", q["options"], key=q["key"])

    if st.button("다음"):
        st.session_state.responses[q["key"]] = response
        next_page()

else:
    st.success("설문이 완료되었습니다! 🎉")
    st.write("📝 당신의 응답:")
    st.json(st.session_state.responses)
