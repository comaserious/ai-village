from persona import Persona
import json

user = json.load(open('test.json')) 

# 페르소나 생성
joy_persona = Persona("Joy" , user)

# 페르소나 스케쥴 생성
joy_persona.plan("Joy",True, user)



print(joy_persona.scratch.daily_req)





