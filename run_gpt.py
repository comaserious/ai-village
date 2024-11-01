from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv
from global_method import *
from datetime import datetime
import re
from prompt import *
import numpy as np
from spatial_memory.spatial import PathFinder, Position
from spatial_memory.maze import map_matrix, zone_labels

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

    

    prompt = f"""

    {persona.scratch.get_str_iss()}

    Answer in Korean

    In general, {persona.scratch.get_str_lifestyle()}

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
        temperature=0.2,
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





# spatial_data => data spatial.json 값
def plan_daily_route(daily_activity, spatial_data,persona):
    current_position = spatial_data["zones"][f"{persona.name}_home"]["positions"][0] #시작점
    route_plan = []



    for activity_obj in daily_activity: # <- 이부분 개선이 필요할듯
        activity = activity_obj["activity"]
        duration = activity_obj["duration"]

        
        prompt = create_location_prompt(persona, activity, duration, current_position, spatial_data)

        llm = ChatOpenAI(
            temperature=0.8,
            model = "gpt-4o-mini",
        )

        location_info = llm.invoke(prompt).content

        location_info = location_info.replace("```json","").replace("```","")

        location_info = json.loads(location_info)

        print("location_info 활동 정보 값",location_info)

        route_plan.append({
            "activity" : activity,
            "location" : location_info["position"],
            "duration" : duration,
            "zone" : location_info["zone"]
        })

        current_position = location_info["position"]

    return route_plan


# ==========================================================================================

def create_full_schedule(route_plan, spatial_data,persona):
    """
    일정과 이동 경로를 포함한 전체 스케줄을 생성합니다.
    """
    complete_schedule = []
    
    for i in range(len(route_plan)):
        current = route_plan[i]
        
        # 현재 활동 추가
        complete_schedule.append({
            "type": "activity",
            "activity": current["activity"],
            "location": current["location"],
            "duration": int(current["duration"]),
            "zone": current["zone"]
        })
        
        # 다음 활동이 있는 경우 이동 경로 추가
        if i < len(route_plan) - 1:
            next_activity = route_plan[i + 1]
            
            # 현재 위치와 다음 위치가 다른 경우에만 경로 계산
            if current["location"] != next_activity["location"]:
                path = get_path_between_points(
                    current["location"],
                    next_activity["location"],
                    spatial_data
                )
                
                # 경로가 존재하고 두 점 이상의 경로가 있는 경우에만 이동 추가
                if path and len(path) > 1:
                    complete_schedule.append({
                        "type": "movement",
                        "path": path,
                        "start_zone": current["zone"],
                        "end_zone": next_activity["zone"],
                        "duration": len(path) - 1  # 각 스텝당 1분으로 가정
                    })
    
    return complete_schedule


def get_path_between_points(start, end, spatial_data):
    """
    spatial.py의 PathFinder를 사용하여 두 점 사이의 경로를 찾습니다.
    
    Args:
        start: [x, y] 시작 좌표
        end: [x, y] 도착 좌표
        spatial_data: 공간 데이터
    
    Returns:
        path: 경로 좌표 리스트
    """
    # numpy array로 변환
    maze_matrix = np.array(map_matrix)
    pathfinder = PathFinder(maze_matrix)
    
    # Position 객체로 변환
    start_pos = Position(start[0], start[1])
    end_pos = Position(end[0], end[1])
    
    # 경로 찾기
    path_result = pathfinder.find_shortest_path(start_pos, end_pos)
    
    if path_result:
        path, cost = path_result
        # Position 객체 리스트를 좌표 리스트로 변환
        return [[p.x, p.y] for p in path]
    else:
        print(f"경로를 찾을 수 없습니다: {start} -> {end}")
        return [start, end]







