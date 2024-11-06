from fastapi import FastAPI

from dotenv import load_dotenv
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials


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


@app.post("/start")
def start():
    db.collection('users').document('test').set({'name' : 'test'})

    
    return {"message" : "게임 스타트"}


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
def chat_user():
    return {"message" : "유저 채팅 값"}





# uvicorn main:app --host 0.0.0.0 --port 1919 --reload
if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버 실행")
    uvicorn.run(app, host="0.0.0.0", port=1919)

    