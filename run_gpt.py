from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv
from global_method import *
from datetime import datetime
import re

load_dotenv()

user  = json.load(open('test.json'))

def first_day_persona(name , user):
    prompt = f"""
            # Input
            당신은 캐릭터를 만들어주는 소설가입니다.
            인사이드 아웃처럼 한 사람의 성격을 매우 극단적으로 만들려고 계획하고 있습니다
            캐릭터의 이름은 { name } 이고, 이번 캐릭터의 MBTI 는 { user['profile']['mbti'] } 입니다.
            다음 Exam 처럼 Output 을 작성하고 json 형태로 값을 출력해줘.
            이번 캐릭터의 레퍼런스는 { name } 입니다

            # Exam
            name : { name }
            age : 나이를 적어주세요.
            personality : 기본적인 성격을 작성해주세요. 예를들어 매우 활달하고 사교성이 높으며 항상 행복한 상태입니다.
            speech : 캐릭터의 말투를 작성해주세요.
            lifestyle : 일반적인 생활 패턴을 적어주세요. 예를들어 6시에 일어나서 11시에 취침합니다. 주말과 주중의 lifestyle 을 구별에서 작성해주세요.
            gender : 성별을 작성 해주세요.
            character : 캐릭터의 기본적인 생활 태도를 작성해주세요.
            """
    
    llm = ChatOpenAI(
        temperature= 0,
        model_name = "gpt-4o-mini",
    )

    response = llm.invoke(prompt)

    print(response.content)

    response_data = make_scratch_memory(name, user , response.content)

# test 용
# first_day_persona("Joy", user)

# 새로운 날의 스케쥴 생성
def new_day_plan( persona , user ):

    today = datetime.today()

    formatted_date = today.strftime("%Y-%m-%d")

    def __func_clean_up(gpt_response):
        schedule_array = []
        
        for line in gpt_response.split('\n'):
            # 줄이 비어 있거나 ')'가 포함되어 있지 않은 경우 건너뜀
            if line.strip() and ')' in line:
                try:
                    # 줄을 분리하고 두 번째 요소를 추가
                    schedule_array.append(line.split(') ')[1].strip())
                except IndexError:
                    # 예외가 발생하면 해당 줄을 건너뜀
                    continue

                format = { formatted_date : schedule_array }
        return format
    
    def set_curr_date(persona, formatted_date):
        with open(f"memory_storage/{user['uid']}/{persona.name}/scratch.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        data['curr_date'] = formatted_date

        persona.scratch.curr_date = formatted_date

        with open(f"memory_storage/{user['uid']}/{persona.name}/scratch.json", 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


    
    set_curr_date(persona , formatted_date)

    response = wake_up_time(persona , today)

    filepath = f"memory_storage/{user['uid']}/{persona.name}/scratch.json"

    persona_scratch = load_persona_data(filepath)

    map_data = json.load(open('memory_storage\DwgZh7Ud7STbVBnkyvK5kmxUIzw1\Joy\spatial.json'))

    prompt = f"""

    {persona.scratch.get_str_iss()}

    Answer in Korean

    In general, {persona.scratch.get_str_lifestyle()}

    너는 지금 {map_data} 이러한 세계에 살고 있습니다.

    x 축 또는 y축의 값이 1 증가해서 이동할 경우 1분 걸리는걸로 합니다.

    모든 계획은 어디서 시작하고 어디서 끝날지 명확하게 좌표로 작성해주세요.

    공간의 이동의 경우 계획에 시작과 끝을 작성하고 계획을 이동 그리고 걸리는 시간을 작성해주세요.

    Today is {formatted_date}. Here is {persona.name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm): 

    1) wake up and complete the morning routine at {response}, 
    2)  # 다음 항목을 여기에 추가하세요

"""

    llm = ChatOpenAI(
        temperature= 0,
        model_name = "gpt-4o-mini",
    )

    response = llm.invoke(prompt)

    print(response.content)

    daily_req = __func_clean_up(response.content)

    update_daily_req(user['uid'], persona.name , daily_req)

    persona.scratch.daily_req = daily_req[formatted_date]


def daily_plan_hourly(persona , user):

    now = datetime.now()

    formatted_time = now.strftime("%H:%M:%S")

    prompt = f"""
    {persona.scratch.get_str_iss()}

    당신은 '{persona.name}'입니다.
    
    오늘의 주요 일정은 "{persona.scratch.daily_req}"입니다.
    
    오늘의 일정을 고려하여, **모든 시간을 반드시 '활동, X분' 형식으로 작성**해 주세요. 일상적인 활동 외에 특별한 아이디어나 새로운 활동을 추가해도 좋습니다.

    **작성 시 지켜야 할 형식**:
    - 각 활동은 반드시 한 줄로 작성하며, "활동, X분" 형식을 유지해 주세요.
    - 예시:
      - "일어나서 칫솔질, 5분"
      - "아침 운동, 30분"
      - "창밖을 보며 커피 한 잔의 여유, 10분"

    **지침**:
    - 모든 활동은 간단하고 명확하게 작성해 주세요.
    - 시간을 '분' 단위로만 작성해 주세요.
    - 창의적인 활동을 추가하여 오늘을 특별하게 만들어보세요!

    이제 당신의 계획을 작성해 주세요.
""" 

    llm = ChatOpenAI(
        temperature=0.8,
        model = "gpt-4o-mini",
    )

    response = llm.invoke(prompt)

    print(response.content)

    # 빈 리스트 생성
    events = []

    # 각 줄을 분리하여 반복 처리
    for line in response.content.split("\n"):
    # 각 줄을 ", "를 기준으로 분리
        parts = line.split(", ")
        if len(parts) == 2:
            task = parts[0].strip('- "')
            duration_text = parts[1].strip()
            # "시간"을 분 단위로 변환
            match = re.search(r'\d+', duration_text)
            if match:
                duration = int(match.group())
                events.append([task, f"{duration}"])

    # 결과 출력
    print(events)


    update_daily_req_hourly(user['uid'], persona.name , events)






def wake_up_time(persona , day):
    prompt = f"""
        # Input
        오늘은 {day} 입니다.
        당신은 {persona.scratch.get_str_iss()} 이러한 사람입니다.
        그렇다면 당신은 오늘 몇시에 일어날까요? Exam 의 형식으로 작성해주세요

        #Exam
        6am
        """
    
    llm = ChatOpenAI(
        temperature=0.8,
        model = "gpt-4o-mini",
    )

    response = llm.invoke(prompt)

    print(response.content)

    return response.content.replace('Exam','')
