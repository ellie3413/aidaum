import streamlit as st
# Streamlit 설정을 가장 먼저 호출해야 함
st.set_page_config(page_title="AI 도구 추천", page_icon="🌟", layout="wide")

import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import re
from dotenv import load_dotenv
from survey import questions, reset_survey, run_survey
from langchain.chains import RetrievalQA
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader  
from langchain.chains.question_answering import load_qa_chain
from user_type import determine_user_type, get_user_type_description


#========== 환경 변수 로딩 ==========
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API 키가 누락되었습니다. .env 파일을 확인해주세요.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# OpenAI API 키가 유효한지 간단히 테스트
try:
    # OpenAI 객체 생성 테스트
    test_llm = OpenAI(temperature=0.1)
except Exception as e:
    st.error(f"❌ OpenAI API 키 검증 중 오류가 발생했습니다: {str(e)}")
    st.stop()

#========== 함수 정의 ==========
def load_json_data():
    """JSON 파일에서 AI 도구 데이터 로드"""
    try:
        with open("tools.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ JSON 파일 로드 오류: {e}")
        return []

def filter_tools_by_difficulty(tools, difficulty_level):
    """난이도 기준으로 AI 도구 필터링"""
    if difficulty_level == "모든 난이도":
        return tools
    
    difficulty_map = {
        "쉬움": "low", 
        "중간": "medium", 
        "어려움": "hard"
    }
    
    target_difficulty = difficulty_map.get(difficulty_level)
    
    filtered = []
    for tool in tools:
        # None 난이도는 중간 난이도로 간주
        if tool.get("difficulty") is None and target_difficulty == "medium":
            filtered.append(tool)
        elif tool.get("difficulty") == target_difficulty:
            filtered.append(tool)
    return filtered

def filter_tools_by_category(tools, category):
    """카테고리 기준으로 AI 도구 필터링"""
    if category == "모든 카테고리":
        return tools
    
    return [tool for tool in tools if tool.get("category") == category]

def filter_tools_by_search(tools, search_term):
    """검색어 기준으로 AI 도구 필터링"""
    if not search_term:
        return tools
    
    return [tool for tool in tools if search_term.lower() in tool.get("name", "").lower() or 
            (tool.get("description") and search_term.lower() in tool.get("description", "").lower())]

def get_tool_details(tool_name, tools_data):
    """도구 이름으로 세부 정보 검색"""
    for tool in tools_data:
        if tool["name"].lower() == tool_name.lower():
            return tool
    return None

def find_best_matching_tool(tool_name, tools_data):
    """가장 유사한 도구 이름 찾기"""
    # 정확히 일치하는 도구 먼저 확인
    for tool in tools_data:
        if tool["name"].lower() == tool_name.lower():
            return tool
    
    # 부분 일치하는 도구 확인
    for tool in tools_data:
        if tool_name.lower() in tool["name"].lower() or tool["name"].lower() in tool_name.lower():
            return tool
    
    return None

def save_user_feedback(tool_name, rating, feedback_text):
    """사용자 피드백 저장"""
    feedback_data = {
        "tool": tool_name,
        "rating": rating,
        "feedback": feedback_text,
        "responses": st.session_state.responses
    }
    
    # 피드백 파일 저장
    try:
        if os.path.exists("user_feedback.json"):
            with open("user_feedback.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_data.append(feedback_data)
            with open("user_feedback.json", "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        else:
            with open("user_feedback.json", "w", encoding="utf-8") as f:
                json.dump([feedback_data], f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")
        return False

def visualize_category_distribution(tools_data):
    """카테고리별 AI 도구 분포 시각화"""
    categories = {}
    for tool in tools_data:
        category = tool.get("category", "기타")
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1
    
    # 상위 10개 카테고리만 표시
    sorted_categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_categories.keys(), sorted_categories.values(), color='skyblue')
    
    # 값 레이블 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.title('AI 도구 카테고리 분포 (상위 10개)')
    plt.tight_layout()
    return fig

def recommend_tools_by_criteria(tools_data, user_responses, max_recommendations=3):
    """사용자 응답 기반으로 AI 도구 알고리즘적 추천"""
    if not tools_data:
        return []
    
    # 점수 초기화
    for tool in tools_data:
        tool['score'] = 0
        
    # 1. 난이도 기준 점수화 (사용자 선호 난이도에 가까울수록 높은 점수)
    difficulty_map = {
        "쉬움 (초보자도 바로 사용 가능한 도구)": "low",
        "중간 (기본적인 지식이 필요한 도구)": "medium", 
        "어려움 (전문적인 지식이 필요한 고급 도구)": "hard",
        "난이도보다는 기능 중심으로 선택하고 싶음": None
    }
    
    preferred_difficulty = difficulty_map.get(
        user_responses.get('preferred_difficulty', "난이도보다는 기능 중심으로 선택하고 싶음")
    )
    
    # 난이도 점수 계산
    if preferred_difficulty:
        for tool in tools_data:
            if tool.get('difficulty') == preferred_difficulty:
                tool['score'] += 5
            # None 난이도는 medium으로 간주
            elif tool.get('difficulty') is None and preferred_difficulty == "medium":
                tool['score'] += 4
    
    # 2. AI
    # 2. AI 지식 수준에 따른 난이도 조정
    knowledge_level = user_responses.get('ai_knowledge', '')
    if knowledge_level in ['전혀 모른다', '이름만 들어봤다']:
        # 초보자는 쉬운 도구 선호
        for tool in tools_data:
            if tool.get('difficulty') == 'low':
                tool['score'] += 3
    elif knowledge_level in ['AI 모델이나 알고리즘을 직접 다뤄본 적 있다']:
        # 전문가는 어려운 도구 선호
        for tool in tools_data:
            if tool.get('difficulty') == 'hard':
                tool['score'] += 3
    
    # 3. 관심 분야 기반 점수화
    interests = user_responses.get('tool_interest', [])
    interest_category_map = {
        "텍스트 생성": ["AI Assistants (Chatbots)", "Writing", "Grammar and Writing Improvement"],
        "이미지 생성": ["Image Generation", "Graphic Design"],
        "영상/음성 합성": ["Video Generation and Editing", "Voice Generation", "Music Generation"],
        "데이터 분석 및 시각화": ["Research"],
        "업무 자동화": ["Project Management", "Scheduling", "Email"],
        "검색 및 지식 관리": ["Search Engines", "Knowledge Management"],
        "코드 생성 및 개발 지원": ["App Builders & Coding"],
        "번역 및 언어 학습": ["Grammar and Writing Improvement"],
        "기타": []
    }
    
    for interest in interests:
        matching_categories = interest_category_map.get(interest, [])
        for tool in tools_data:
            if tool.get('category') in matching_categories:
                tool['score'] += 4
    
    # 4. 특정 목적 기반 점수화
    purposes = user_responses.get('specific_purpose', [])
    purpose_category_map = {
        "문서 작성 및 편집": ["Writing", "Grammar and Writing Improvement"],
        "이미지/영상 제작": ["Image Generation", "Video Generation and Editing"],
        "데이터 분석": ["Research"],
        "프로그래밍 및 개발": ["App Builders & Coding"],
        "마케팅 및 홍보": ["Marketing", "Social Media Management"],
        "교육 및 학습": ["Knowledge Management"],
        "업무 자동화": ["Project Management", "Scheduling", "Email"],
        "고객 서비스": ["Customer Service"],
        "연구 및 논문 작성": ["Research", "Writing"],
        "기타": []
    }
    
    for purpose in purposes:
        matching_categories = purpose_category_map.get(purpose, [])
        for tool in tools_data:
            if tool.get('category') in matching_categories:
                tool['score'] += 4
    
    # 5. 직업 기반 점수화 (추가됨)
    job = user_responses.get('job', '')
    job_category_map = {
        "학생": ["Writing", "Research", "Grammar and Writing Improvement", "Knowledge Management"],
        "개발자/IT 종사자": ["App Builders & Coding", "AI Assistants (Chatbots)"],
        "교육자/연구원": ["Research", "Knowledge Management", "Presentations", "Writing"],
        "디자이너/창작자": ["Image Generation", "Video Generation and Editing", "Graphic Design", "Music Generation"],
        "마케터/홍보": ["Social Media Management", "Marketing", "Writing", "Image Generation"],
        "사무직": ["Email", "Project Management", "Scheduling", "Writing"],
        "경영/관리자": ["Project Management", "Knowledge Management", "Presentations"],
        "창업가/프리랜서": ["Marketing", "Social Media Management", "Email", "Customer Service"],
        "의료/건강 종사자": ["Research", "Knowledge Management"],
        "법률/금융 전문가": ["Research", "Grammar and Writing Improvement", "Writing"],
        "기타": []
    }
    
    if job:
        matching_categories = job_category_map.get(job, [])
        for tool in tools_data:
            if tool.get('category') in matching_categories:
                tool['score'] += 5  # 직업 관련성이 높은 도구에 더 높은 가중치 부여
    
    # 도구 설명이 있는 도구에 가중치 부여
    for tool in tools_data:
        if tool.get('description') and len(str(tool.get('description'))) > 10:
            tool['score'] += 1
    
    # 최종 점수 기준 정렬 및 상위 추천
    sorted_tools = sorted(tools_data, key=lambda x: x.get('score', 0), reverse=True)
    
    # 최소한 하나의 쉬운 도구가 포함되도록 보장 (초보자를 위한 배려)
    recommended = []
    has_easy_tool = False
    
    # 상위 도구들 중에서 선택
    for tool in sorted_tools:
        if len(recommended) < max_recommendations:
            recommended.append(tool)
            if tool.get('difficulty') == 'low':
                has_easy_tool = True
        elif not has_easy_tool and tool.get('difficulty') == 'low':
            # 쉬운 도구가 없으면 마지막 도구를 쉬운 도구로 교체
            recommended[-1] = tool
            has_easy_tool = True
            break
    
    return recommended

def translate_difficulty(difficulty):
    """난이도 영어 표현을 한국어로 변환"""
    if difficulty == "low":
        return "쉬움"
    elif difficulty == "medium":
        return "중간"
    elif difficulty == "hard":
        return "어려움"
    return "중간"  # 기본값

def add_korean_description(tools):
    """영어 설명이 있는 도구에 한국어 설명 추가"""
    korean_descriptions = {
        "ChatGPT": "다양한 텍스트 생성과 대화가 가능한 OpenAI의 대표적인 AI 챗봇으로, 코딩, 글쓰기, 질문 응답 등 다양한 작업에 활용할 수 있습니다.",
        "Claude": "Anthropic에서 개발한 AI 어시스턴트로, 친절하고 정확한 응답과 특히 코딩에 강점을 가지고 있습니다.",
        "Gemini": "Google에서 개발한 AI 어시스턴트로 구글 생태계와 높은 통합성을 가지고 있으며 검색과 정보 요약에 강점이 있습니다.",
        "Midjourney": "텍스트 프롬프트를 기반으로 고품질 이미지를 생성하는 AI 도구로, 예술적 표현과 창의적인 시각화에 탁월합니다.",
        "Perplexity": "다양한 정보 소스를 활용해 깊이 있는 검색과 답변을 제공하는 AI 검색 엔진입니다.",
        "Grammarly": "텍스트 작성 시 문법, 맞춤법, 문체를 자동으로 교정해주는 AI 글쓰기 도우미입니다.",
        "Canva Magic Studio": "손쉬운 디자인 제작을 위한 AI 기능이 강화된 그래픽 디자인 플랫폼입니다.",
        "Cursor": "AI 기반 코드 작성과 편집을 도와주는 개발자 도구로, 코딩 생산성을 크게 향상시킵니다."
    }
    
    for tool in tools:
        if tool.get("name") in korean_descriptions and (tool.get("description") is None or "Korean" not in tool.get("lang", [])):
            tool["korean_description"] = korean_descriptions[tool.get("name")]
    
    return tools

#========== Streamlit UI ==========
st.title("에이아이다움")
st.write("설문조사를 완료하시면, 당신에게 맞는 AI 도구를 추천해드립니다.")

#========== 설문 화면 ==========
run_survey()

#========== 설문 완료 여부 확인 ==========
if not st.session_state.get("survey_complete", False):
    st.stop()

responses = st.session_state.responses

#========== tools.txt 및 JSON 데이터 로드 ==========

loader = TextLoader("tools.txt", encoding="utf-8") 
pages = loader.load_and_split()

# JSON 데이터 로드
tools_data = load_json_data()

#========== 사용자 선호도에 맞는 검색 매개변수 결정 ==========
search_kwargs = {"k": 5}  # 기본값

# AI 지식 수준에 따라 검색 깊이 조정
if responses.get('ai_knowledge') in ['전혀 모른다', '이름만 들어봤다']:
    search_kwargs["k"] = 3  # 초보자는 더 기본적인 내용만 검색
elif responses.get('ai_knowledge') in ['AI 모델이나 알고리즘을 직접 다뤄본 적 있다']:
    search_kwargs["k"] = 7  # 전문가는 더 깊은 검색

#========== AI 유형 추천 ==========

st.markdown("### 🧩 당신의 AI 유형은?")

# 사용자 유형 결정
user_type = determine_user_type(responses)
user_type_info = get_user_type_description(user_type)

# 유형 정보 표시
st.markdown(f"## {user_type_info['title']}")
st.markdown(user_type_info['description'])

# 유형 세부 정보
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 💪 강점")
    st.markdown(user_type_info['strengths'])
    
with col2:
    st.markdown("#### 🚀 추천 접근법")
    st.markdown(user_type_info['recommended_approach'])

st.markdown("---")


#========== 알고리즘 기반 도구 추천 ==========
st.markdown("### 🔎 당신을 위한 AI 도구 추천")

# 한국어 설명 추가
tools_data = add_korean_description(tools_data)

# 알고리즘 기반 추천
with st.spinner("추천 생성 중입니다..."):
    recommended_tools = recommend_tools_by_criteria(tools_data, responses, max_recommendations=3)

# 추천 결과 표시
if recommended_tools:
    st.success(f"✅ 설문 응답에 기반한 맞춤형 AI 도구 추천이 완료되었습니다.")
    st.markdown(f"**{user_type}** 유형인 당신을 위한 맞춤형 AI 도구입니다. 이 도구들은 당신의 관심사, 목적, 직업을 고려하여 특별히 선정되었습니다.")


    
    # 3개의 열로 추천 도구 표시
    cols = st.columns(len(recommended_tools))
    
    for i, tool in enumerate(recommended_tools):
        with cols[i]:
            st.markdown(f"### {i+1}. {tool.get('name')}")
            st.markdown(f"**카테고리**: {tool.get('category', '정보 없음')}")
            st.markdown(f"**난이도**: {translate_difficulty(tool.get('difficulty', 'medium'))}")
            
            # 한국어 설명 우선, 없으면 기존 설명 사용
            if tool.get("korean_description"):
                st.markdown(f"**설명**: {tool.get('korean_description')}")
            elif tool.get("description"):
                st.markdown(f"**설명**: {tool.get('description')}")
            else:
                st.markdown("**설명**: 상세 설명 정보가 없습니다.")
            
            # 도구별 상세 설명 버튼
            if st.button(f"{tool.get('name')} 자세히 보기", key=f"detail_{i}"):
                # 세션 상태에 선택된 도구 저장
                st.session_state.selected_tool = tool.get('name')
                st.rerun()
else:
    st.warning("⚠️ 추천 도구를 찾을 수 없습니다. 다른 설문 응답을 시도해 보세요.")

# 선택된 도구 상세 정보 표시
if hasattr(st.session_state, 'selected_tool') and st.session_state.selected_tool:
    tool_name = st.session_state.selected_tool
    st.markdown(f"## {tool_name} 상세 정보")
    
    # 툴 정보 찾기
    tool_info = get_tool_details(tool_name, tools_data)
    
    if tool_info:
        st.markdown(f"**카테고리**: {tool_info.get('category', '정보 없음')}")
        st.markdown(f"**난이도**: {translate_difficulty(tool_info.get('difficulty', 'medium'))}")
        
        # 한국어 설명 우선, 없으면 기존 설명 사용
        if tool_info.get("korean_description"):
            st.markdown(f"**설명**: {tool_info.get('korean_description')}")
        elif tool_info.get("description"):
            st.markdown(f"**설명**: {tool_info.get('description')}")
        else:
            st.markdown("**설명**: 상세 설명 정보가 없습니다.")
        
        # tools.txt에서 해당 도구 검색
        embeddings = OpenAIEmbeddings()
        try:
            vectorstore = FAISS.from_texts(texts=[page.page_content for page in pages], embedding=embeddings)
            query = f"{tool_name}에 대한 상세 정보와 사용 방법"
            docs = vectorstore.similarity_search(query, k=2)
            
            if docs:
                st.markdown("### 📚 추가 정보")
                for doc in docs:
                    st.markdown(doc.page_content)
        except Exception as e:
            st.error(f"도구 상세 정보 검색 중 오류 발생: {e}")
    else:
        st.error(f"{tool_name}에 대한 정보를 찾을 수 없습니다.")
    
    # 상세보기 닫기
    if st.button("상세 정보 닫기"):
        del st.session_state.selected_tool
        st.rerun()

st.markdown("---")

#========== 점수표 ==========
if recommended_tools:
    with st.expander("🤖 맞춤형 도구 선정 근거(점수표)", expanded=False):
        st.markdown("#### 추천 점수 분포")
        score_fig, ax = plt.subplots(figsize=(8, 4))
        tool_names = [tool.get('name') for tool in recommended_tools]
        scores = [tool.get('score', 0) for tool in recommended_tools]
        
        bars = ax.barh(tool_names, scores, color=['#2E86C1', '#3498DB', '#85C1E9'])

        # 값 표시
        for i, (score, bar) in enumerate(zip(scores, bars)):
            ax.text(score + 0.5, i, f"{score}", ha='left', va='center')
        
        plt.xlabel('your score')
        plt.tight_layout()
        st.pyplot(score_fig)


#========== 난이도 필터 및 세부 정보 ==========
st.markdown("---")
st.markdown("### 🔍 AI 도구 데이터베이스 탐색")

if tools_data:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 난이도별 필터링
        difficulty_options = ["모든 난이도", "쉬움", "중간", "어려움"]
        selected_difficulty = st.selectbox("난이도별 필터링", difficulty_options)
    
    with col2:
        # 카테고리별 필터링
        categories = ["모든 카테고리"] + sorted(list(set([tool.get("category", "기타") for tool in tools_data if tool.get("category")])))
        selected_category = st.selectbox("카테고리별 필터링", categories)
    
    with col3:
        # 검색어 필터링
        search_term = st.text_input("🔍 도구 이름 또는 설명 검색")
    
    # 필터링 적용
    filtered_tools = filter_tools_by_difficulty(tools_data, selected_difficulty)
    filtered_tools = filter_tools_by_category(filtered_tools, selected_category)
    filtered_tools = filter_tools_by_search(filtered_tools, search_term)
    
    # 카테고리별 도구 분포 시각화
    st.markdown("### 📊 AI 도구 카테고리 분포")
    fig = visualize_category_distribution(tools_data)
    st.pyplot(fig)
    
    # 필터링된 도구 리스트
    st.markdown("### 📋 필터링된 도구 목록")
    if filtered_tools:
        tool_df = pd.DataFrame([{
            "이름": tool.get("name", ""),
            "카테고리": tool.get("category", ""),
            "난이도": translate_difficulty(tool.get("difficulty", "medium"))
        } for tool in filtered_tools])
        
        st.dataframe(tool_df, use_container_width=True)
        
        # 도구 상세 정보 확인
        selected_tool_name = st.selectbox("상세 정보를 볼 도구 선택", ["선택하세요"] + tool_df["이름"].tolist())
        
        if selected_tool_name != "선택하세요":
            tool_info = get_tool_details(selected_tool_name, tools_data)
            if tool_info:
                st.markdown(f"### {selected_tool_name} 상세 정보")
                st.markdown(f"**카테고리**: {tool_info.get('category', '정보 없음')}")
                st.markdown(f"**난이도**: {translate_difficulty(tool_info.get('difficulty', 'medium'))}")
                
                # 한국어 설명 우선, 없으면 기존 설명 사용
                if tool_info.get("korean_description"):
                    st.markdown(f"**설명**: {tool_info.get('korean_description')}")
                elif tool_info.get("description"):
                    st.markdown(f"**설명**: {tool_info.get('description')}")
                else:
                    st.markdown("**설명**: 상세 설명 정보가 없습니다.")
    else:
        st.info("선택한 조건에 맞는 도구가 없습니다.")

#========== 사용자 피드백 ==========
st.markdown("---")
st.markdown("### 📝 추천 피드백")
feedback_tool = st.text_input("피드백을 남길 도구 이름")

if feedback_tool:
    rating = st.slider("만족도 평가", 1, 5, 3)
    feedback_text = st.text_area("상세 피드백 (선택사항)")
    
    if st.button("피드백 제출"):
        if save_user_feedback(feedback_tool, rating, feedback_text):
            st.success("피드백이 성공적으로 저장되었습니다. 감사합니다!")
        else:
            st.error("피드백 저장 중 오류가 발생했습니다.")

#========== 학습 리소스 추천 ==========
st.markdown("---")
st.markdown("### 📚 추천 학습 리소스")

learning_resources = {
    "초보자": [
        "AI 기초 개념 이해하기 - 쉬운 시작 가이드",
        "처음 만나는 ChatGPT - 기본 사용법",
        "AI 도구 입문자를 위한 단계별 학습 경로"
    ],
    "중급자": [
        "실무에 바로 적용하는 AI 도구 활용법",
        "데이터 분석을 위한 AI 모델 활용하기",
        "업무 자동화를 위한 AI 워크플로우 구축"
    ],
    "전문가": [
        "고급 프롬프트 엔지니어링 기법",
        "AI 모델 미세 조정 및 커스터마이징",
        "RAG(Retrieval Augmented Generation) 고급 활용법"
    ]
}

# 사용자 수준에 맞는 리소스 표시
user_level = "초보자"
if responses.get('ai_knowledge') in ['기본 개념은 알고 있다', '실제로 활용해본 경험이 있다']:
    user_level = "중급자"
elif responses.get('ai_knowledge') in ['AI 모델이나 알고리즘을 직접 다뤄본 적 있다']:
    user_level = "전문가"

st.info(f"📚 당신의 수준({user_level})에 맞는 학습 리소스를 추천합니다:")
for resource in learning_resources[user_level]:
    st.markdown(f"- {resource}")

st.button("🔄 설문 다시 하기", on_click=reset_survey)