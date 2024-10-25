from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv
from global_method import *
from datetime import datetime

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

def new_day_plan( persona , user ):

    today = datetime.today()

    formatted_date = today.strftime("%Y-%m-%d")

    response = wake_up_time(persona , today)

    filepath = f"memory_storage/{user['uid']/{persona.name}}"

    persona_scratch = load_persona_data(filepath)

    persona_scratch.get('wake_up_time')= wake_up_time

    prompt = f"""
    # Input
    Today is {formatted_date}. You plan to wake up {response}. You have the following personal details: {persona.scratch.get_str_iss()}.
    Please analyze {persona.scratch.get_str_iss()} and create your schedule today by hourly interval in the format of an Example. Speak in Korean

    # Example
    1) am ~ am: 
    """

    llm = ChatOpenAI(
        temperature= 0,
        model_name = "gpt-4o-mini",
    )

    response = llm.invoke(prompt)

    print(response.content)

    filepath = f''



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