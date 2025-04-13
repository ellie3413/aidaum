# user_type.py

def determine_user_type(responses):
    """사용자 응답에 기반한 AI 사용자 유형 결정"""
    # 포인트 초기화
    user_type_points = {
        "AI 탐험가": 0,
        "디지털 아티스트": 0,
        "효율성 추구자": 0,
        "지식 수집가": 0,
        "코드 마법사": 0,
        "콘텐츠 크리에이터": 0,
        "비즈니스 전략가": 0,
        "AI 초보 탐험가": 0
    }
    
    # AI 지식 수준 기반 점수 부여
    knowledge_level = responses.get('ai_knowledge', '')
    if knowledge_level in ['전혀 모른다', '이름만 들어봤다']:
        user_type_points["AI 초보 탐험가"] += 10
    elif knowledge_level in ['기본 개념은 알고 있다']:
        user_type_points["AI 탐험가"] += 5
        user_type_points["지식 수집가"] += 3
    elif knowledge_level in ['실제로 활용해본 경험이 있다']:
        user_type_points["효율성 추구자"] += 5
        user_type_points["콘텐츠 크리에이터"] += 3
        user_type_points["비즈니스 전략가"] += 3
    elif knowledge_level in ['AI 모델이나 알고리즘을 직접 다뤄본 적 있다']:
        user_type_points["코드 마법사"] += 8
        user_type_points["AI 탐험가"] += 5
    
    # 직업 기반 점수 부여
    job = responses.get('job', '')
    if job == "학생":
        user_type_points["AI 탐험가"] += 3
        user_type_points["지식 수집가"] += 3
    elif job == "개발자/IT 종사자":
        user_type_points["코드 마법사"] += 8
        user_type_points["효율성 추구자"] += 3
    elif job == "교육자/연구원":
        user_type_points["지식 수집가"] += 7
        user_type_points["콘텐츠 크리에이터"] += 3
    elif job == "디자이너/창작자":
        user_type_points["디지털 아티스트"] += 10
        user_type_points["콘텐츠 크리에이터"] += 5
    elif job == "마케터/홍보":
        user_type_points["비즈니스 전략가"] += 7
        user_type_points["콘텐츠 크리에이터"] += 5
    elif job == "사무직":
        user_type_points["효율성 추구자"] += 8
        user_type_points["지식 수집가"] += 3
    elif job == "경영/관리자":
        user_type_points["비즈니스 전략가"] += 9
        user_type_points["효율성 추구자"] += 6
    elif job == "창업가/프리랜서":
        user_type_points["AI 탐험가"] += 5
        user_type_points["비즈니스 전략가"] += 5
        user_type_points["효율성 추구자"] += 4
    
    # 관심 분야 기반 점수 부여
    interests = responses.get('tool_interest', [])
    for interest in interests:
        if interest == "텍스트 생성":
            user_type_points["콘텐츠 크리에이터"] += 4
        elif interest == "이미지 생성":
            user_type_points["디지털 아티스트"] += 5
        elif interest == "영상/음성 합성":
            user_type_points["디지털 아티스트"] += 4
            user_type_points["콘텐츠 크리에이터"] += 3
        elif interest == "데이터 분석 및 시각화":
            user_type_points["비즈니스 전략가"] += 4
            user_type_points["지식 수집가"] += 3
        elif interest == "업무 자동화":
            user_type_points["효율성 추구자"] += 6
        elif interest == "검색 및 지식 관리":
            user_type_points["지식 수집가"] += 6
        elif interest == "코드 생성 및 개발 지원":
            user_type_points["코드 마법사"] += 7
        elif interest == "번역 및 언어 학습":
            user_type_points["지식 수집가"] += 3
            user_type_points["콘텐츠 크리에이터"] += 2
    
    # 활용 목적 기반 점수 부여
    purposes = responses.get('specific_purpose', [])
    for purpose in purposes:
        if purpose == "문서 작성 및 편집":
            user_type_points["콘텐츠 크리에이터"] += 4
            user_type_points["효율성 추구자"] += 2
        elif purpose == "이미지/영상 제작":
            user_type_points["디지털 아티스트"] += 6
        elif purpose == "데이터 분석":
            user_type_points["비즈니스 전략가"] += 4
            user_type_points["지식 수집가"] += 3
        elif purpose == "프로그래밍 및 개발":
            user_type_points["코드 마법사"] += 6
        elif purpose == "마케팅 및 홍보":
            user_type_points["비즈니스 전략가"] += 5
            user_type_points["콘텐츠 크리에이터"] += 3
        elif purpose == "교육 및 학습":
            user_type_points["지식 수집가"] += 5
        elif purpose == "업무 자동화":
            user_type_points["효율성 추구자"] += 6
        elif purpose == "고객 서비스":
            user_type_points["비즈니스 전략가"] += 3
        elif purpose == "연구 및 논문 작성":
            user_type_points["지식 수집가"] += 6
            user_type_points["콘텐츠 크리에이터"] += 2
    
    # 초보 레벨일 경우 초보 탐험가 가중치 추가
    if knowledge_level in ['전혀 모른다', '이름만 들어봤다']:
        user_type_points["AI 초보 탐험가"] = max(user_type_points.values()) + 5
    
    # 점수가 가장 높은 유형 선택
    user_type = max(user_type_points.items(), key=lambda x: x[1])[0]
    
    return user_type

def get_user_type_description(user_type):
    """사용자 유형에 대한 설명 반환"""
    descriptions = {
        "AI 탐험가": {
            "title": "AI 탐험가 🧭",
            "description": "당신은 새로운 AI 기술과 도구에 호기심이 많은 '**AI 탐험가**'입니다! 다양한 AI 도구를 시도하고 탐색하는 것을 즐기며, 항상 최신 기술 트렌드를 따라가는 얼리어답터의 기질을 가지고 있습니다. 당신은 AI의 가능성을 넓게 보고 다양한 분야에서 활용 방법을 찾아냅니다.",
            "strengths": "호기심, 적응력, 다양한 도구 활용 능력",
            "recommended_approach": "다양한 AI 도구를 시도해보고, 각 도구의 장단점을 비교해보세요. 새로운 사용 사례를 발견하는 데 집중하세요."
        },
        "디지털 아티스트": {
            "title": "디지털 아티스트 🎨",
            "description": "당신은 AI를 통해 창의적인 작품을 만들어내는 '**디지털 아티스트**'입니다! 이미지, 영상, 음악 생성 등 AI의 창작 능력을 활용하여 독창적인 콘텐츠를 제작하는 재능이 있습니다. 기술과 예술의 경계를 탐험하며 새로운 표현 방식을 개척하고 있습니다.",
            "strengths": "창의력, 시각적 감각, 실험 정신",
            "recommended_approach": "다양한 프롬프트 실험과 스타일 조합을 통해 자신만의 창작 방식을 개발하세요. 생성형 AI의 가능성을 최대한 활용하세요."
        },
        "효율성 추구자": {
            "title": "효율성 추구자 ⚡",
            "description": "당신은 일상과 업무의 효율을 극대화하는 '**효율성 추구자**'입니다! 반복적인 작업을 자동화하고 프로세스를 최적화하는 데 AI를 활용하는 데 탁월합니다. 시간을 절약하고 생산성을 높이는 방법을 끊임없이 모색하며, 복잡한 문제를 단순화하는 능력이 뛰어납니다.",
            "strengths": "체계적 사고, 최적화 능력, 자동화 역량",
            "recommended_approach": "워크플로우를 분석하고 자동화할 수 있는 부분을 찾아내세요. AI 도구를 통합하여 시스템을 구축하는 데 집중하세요."
        },
        "지식 수집가": {
            "title": "지식 수집가 📚",
            "description": "당신은 정보를 체계화하고 지식을 축적하는 '**지식 수집가**'입니다! AI를 활용하여 방대한 양의 정보를 수집, 정리, 분석하는 데 능숙합니다. 복잡한 주제를 이해하고 통찰력 있는 결론을 도출하는 능력이 뛰어나며, 지식 관리 시스템을 구축하는 데 관심이 많습니다.",
            "strengths": "정보 분석력, 체계화 능력, 지적 호기심",
            "recommended_approach": "개인 지식 베이스를 구축하고 AI로 정보를 효과적으로 검색, 요약, 분석하는 방법을 익히세요."
        },
        "코드 마법사": {
            "title": "코드 마법사 💻",
            "description": "당신은 AI를 활용하여 코드를 작성하고 개발 과정을 향상시키는 '**코드 마법사**'입니다! 프로그래밍과 AI를 결합하여 더 효율적이고 혁신적인 솔루션을 만들어내는 능력이 있습니다. 복잡한 기술적 문제를 해결하고, AI의 가능성을 기술적으로 구현하는 데 관심이 많습니다.",
            "strengths": "기술적 사고력, 문제 해결 능력, 코딩 스킬",
            "recommended_approach": "AI 코딩 도구를 개발 워크플로우에 통합하고, 더 복잡한 프로젝트에 도전하세요. AI와 함께 프로그래밍 스킬을 향상시키는 데 집중하세요."
        },
        "콘텐츠 크리에이터": {
            "title": "콘텐츠 크리에이터 ✍️",
            "description": "당신은 글쓰기와 콘텐츠 제작에 AI를 활용하는 '**콘텐츠 크리에이터**'입니다! 블로그, 소셜 미디어, 마케팅 자료 등 다양한 형태의 콘텐츠를 제작하는 데 AI의 도움을 받아 효율적으로 작업합니다. 아이디어 발굴부터 편집까지 콘텐츠 제작의 전 과정에서 AI를 전략적으로 활용합니다.",
            "strengths": "표현력, 창의적 사고, 콘텐츠 기획 능력",
            "recommended_approach": "AI를 협업 도구로 활용하여 아이디어 발굴, 초안 작성, 편집 과정을 효율화하세요. 자신만의 콘텐츠 스타일을 개발하는 데 AI를 보조 도구로 활용하세요."
        },
        "비즈니스 전략가": {
            "title": "비즈니스 전략가 📊",
            "description": "당신은 비즈니스 의사결정과 전략 수립에 AI를 활용하는 '**비즈니스 전략가**'입니다! 데이터 분석, 시장 조사, 트렌드 예측 등에 AI 도구를 활용하여 더 나은 비즈니스 인사이트를 얻습니다. 경쟁 우위를 확보하고 성장 기회를 발견하는 데 AI의 분석력을 전략적으로 활용합니다.",
            "strengths": "분석적 사고, 전략적 안목, 비즈니스 감각",
            "recommended_approach": "AI를 활용한 데이터 분석과 예측 모델링에 집중하세요. 비즈니스 결정을 지원하는 AI 기반 대시보드와 보고서를 개발하는 데 투자하세요."
        },
        "AI 초보 탐험가": {
            "title": "AI 초보 탐험가 🌱",
            "description": "당신은 AI의 세계에 첫 발을 내딛은 '**AI 초보 탐험가**'입니다! 새로운 기술에 대한 호기심과 배움의 의지를 가지고 있으며, AI가 제공하는 가능성을 알아가는 여정을 시작했습니다. 기초부터 차근차근 배우며 AI를 일상에 적용하는 방법을 탐색하고 있습니다.",
            "strengths": "열린 마음, 학습 의지, 새로운 시각",
            "recommended_approach": "사용하기 쉬운 AI 도구부터 시작하여 점진적으로 경험을 쌓아가세요. 기초 개념을 이해하는 데 시간을 투자하고, 작은 프로젝트로 실전 경험을 쌓아보세요."
        }
    }
    
    return descriptions.get(user_type, {
        "title": "AI 탐험가",
        "description": "다양한 AI 도구에 관심이 많으신 분입니다!",
        "strengths": "호기심, 적응력",
        "recommended_approach": "다양한 AI 도구를 시도해보세요."
    })