from fastapi import FastAPI, Request

from dotenv import load_dotenv
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials
import json
from test_convo_v8 import *

from global_method import *
from run_gpt import *

from spatial_memory.spatial import *
from spatial_memory.maze import *

load_dotenv()

cred = credentials.Certificate("mirrorgram-20713-firebase-adminsdk-u9pdx-c3e12134b4.json")
firebase_admin.initialize_app(cred,{
    'storageBucket' : 'mirrorgram-20713.appspot.com'
})

app = FastAPI()

db = firestore.client()

@app.post("/")
def read_root():
    return {"message": "Hello, World!"}


# @app.post("/start")
# async def start(request: Request):
#     print("게임 시작")
#     data = await request.json()
    
#     uid = data['uid']
#     daily_activity = []

#     personas = [
#         Persona("Joy", data),
#         Persona("Anger", data),
#         Persona("Sadness", data)
#     ]

#     for persona in personas:
#         activities = []
        
#         persona.plan(persona.name, True, data)
#         daily_plan_hourly(persona, data)
        
#         json_file = json.load(open(f"memory_storage/{uid}/{persona.name}/scratch.json"))
#         daily_schedule_hourly = json_file["daily_req_hourly"]
        
#         for activity, duration in daily_schedule_hourly:
#             activities.append({
#                 "activity": activity,
#                 "duration": duration
#             })
        
#         spatial_memory = SpatialMemory(map_matrix, zone_labels)
#         filepath = f"memory_storage/{uid}/{persona.name}/spatial.json"
#         spatial_memory.export_spatial_memory(filepath)
        
#         spatial_data = json.load(open(filepath))
#         route_plan = plan_daily_route(activities, spatial_data, persona)
#         complete_schedule = create_full_schedule(route_plan, spatial_data, persona)
        
#         daily_activity.append({
#             "name": persona.name,
#             "wake_up_time": persona.scratch.wake_up_time,
#             "daily_schedule": complete_schedule
#         })

#     # village/schedule/{uid} 경로에 저장
#     try:
#         # village/schedule 컬렉션에 문서 추가
#         schedule_ref = db.collection('village').document('schedule')
#         schedule_ref.set({
#             uid: {
#                 'timestamp': datetime.now(),
#                 'schedule': daily_activity
#             }
#         }, merge=True)  # merge=True로 설정하여 기존 문서를 덮어쓰지 않고 업데이트
        
#         print(f"Successfully saved schedule for user {uid} to Firestore")
#     except Exception as e:
#         print(f"Error saving to Firestore: {str(e)}")
#         # 에러 발생시에도 클라이언트에게는 스케줄 데이터 반환

#     return {"schedule": daily_activity}

@app.post("/start")
async def start(request: Request):
    print("게임 시작")
    data = await request.json()
    uid = data['uid']
    daily_activity = []

    # 현재 날짜 구하기 (시간은 제외)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    personas = [
        Persona("Joy", data),
        Persona("Anger", data),
        Persona("Sadness", data)
    ]

    for persona in personas:
        activities = []
        
        persona.plan(persona.name, True, data)
        daily_plan_hourly(persona, data)
        
        json_file = json.load(open(f"memory_storage/{uid}/{persona.name}/scratch.json"))
        daily_schedule_hourly = json_file["daily_req_hourly"]
        
        for activity, duration in daily_schedule_hourly:
            activities.append({
                "activity": activity,
                "duration": duration
            })
        
        spatial_memory = SpatialMemory(map_matrix, zone_labels)
        filepath = f"memory_storage/{uid}/{persona.name}/spatial.json"
        spatial_memory.export_spatial_memory(filepath)
        
        spatial_data = json.load(open(filepath))
        route_plan = plan_daily_route(activities, spatial_data, persona)
        complete_schedule = create_full_schedule(route_plan, spatial_data, persona)
        
        daily_activity.append({
            "name": persona.name,
            "wake_up_time": persona.scratch.wake_up_time,
            "daily_schedule": complete_schedule
        })

    try:
        # daily_activity를 JSON 문자열로 변환
        daily_activity_str = json.dumps(daily_activity)
        
        schedule_data = {
            'timestamp': datetime.now(),
            'date': today,
            'schedule': daily_activity_str,  # JSON 문자열로 저장
            'uid' : uid
        }
        
        doc_id = f"{uid}_{today.strftime('%Y%m%d')}"
        db.collection('village').document('schedule').collection('schedules').document(doc_id).set(schedule_data)
        print(f"Successfully saved schedule for user {uid} to Firestore")
        
    except Exception as e:
        print(f"Error saving to Firestore: {str(e)}")
        
    return {"schedule": daily_activity}


@app.post("/end")
def end():
    return {"message" : "게임 종료"}



@app.post("/move")
def move():
    return {"message" : "움직임 값"}



@app.post("/chat/persona")
def chat_persona():
    return {"message" : "페르소나 채팅 값"}




@app.post("/chat/user")
async def chat_user(request : Request):
    print("유저와의 채팅")
    param = await request.json()

    data = json.loads(param['param'])

    uid = data['uid']
    persona_name = data['persona']
    persona = Persona(persona_name , data)

    response = run_conversation(uid , persona , message = data['message'])



    return {"message" : response}





# uvicorn main:app --host 0.0.0.0 --port 1919 --reload
if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버 실행")
    uvicorn.run(app, host="0.0.0.0", port=1919)

    
