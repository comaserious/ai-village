from persona import Persona
from run_gpt import daily_plan_hourly
import json
from spatial_memory.spatial import *
from spatial_memory.maze import *
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

user = json.load(open('test.json')) 

# 페르소나 생성
joy_persona = Persona("Joy" , user)

# 페르소나 스케쥴 생성
joy_persona.plan("Joy",True, user)



print(joy_persona.scratch.daily_req)

daily_plan_hourly(joy_persona , user)





# spatial_memory = SpatialMemory(map_matrix, zone_labels)

# path_result = spatial_memory.get_path(2, 11)

# if path_result:
#         path, cost = path_result
#         print(f"\nFound path from {spatial_memory.zone_labels[2]} to {spatial_memory.zone_labels[11]}")
#         print(f"Path cost: {cost}")
        
#         # 경로 설명 출력
#         print("\nRoute description:")
#         for step in spatial_memory.get_route_description(path):
#             print(f"- {step}")
        
#         # 경로 시각화
#         spatial_memory.visualize_path(path, {2, 11, 8})



# # 구조파악
# data = json.load(open('memory_storage\DwgZh7Ud7STbVBnkyvK5kmxUIzw1\Joy\spatial.json'))

# user = json.load(open('test.json'))

# joy = Persona("Joy" , user)

# # prompt = f"{data} 이 데이터는 무엇을 의미하는가?"

# prompt = f"""
# {joy.scratch.get_str_iss()}

# 당신은 {joy.name} 입니다.
# 당신은 지금 {data} 이러한 공간에 살고 있습니다.

# 아침에 일어나서 무엇을 할 것 인지 대략적인 스케쥴을 만들고

# 각각의 스케쥴에 따른 위치를 명확하게 확정 지어서 그 위치를 작성해주세요.
# """

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# response = llm.invoke(prompt)

# print(response.content)





