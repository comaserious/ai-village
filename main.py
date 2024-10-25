from persona import Persona
import json

user = json.load(open('test.json')) 

joy_persona = Persona("Joy" , user)

joy_persona.plan("Joy",True)



