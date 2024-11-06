from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing import List, Dict
import time
import random

from persona import Persona

class ConversationAgent:
    def __init__(self, persona1: Persona, persona2: Persona, model: str = "gpt-4"):
        self.persona = persona1
        self.name = persona1.name
        
        # persona의 기본 특성 가져오기
        self.iss = persona1.scratch.get_str_iss()
        self.personality = persona1.scratch.get_str_personality()
        self.speech = persona1.scratch.get_str_speech()
        self.lifestyle = persona1.scratch.get_str_lifestyle()
        
        # persona2와의 관계 정보 가져오기
        self.relationship_info = persona1.get_relationship_info(persona2.name)
        self.relationship_type = persona1.get_relationship_type_with(persona2.name)
        self.closeness = persona1.get_my_closeness_to(persona2.name)
        self.interaction_style = persona1.get_interaction_style_with(persona2.name)
        self.preferred_activities = persona1.get_activities_i_prefer_with(persona2.name)
        self.potential_conflicts = persona1.get_conflicts_i_feel_with(persona2.name)
        
        self.chat_history: List[dict] = []
        
        # 에이전트의 프롬프트 템플릿 설정
        system_message = f"""당신은 {self.name}입니다.

당신의 기본 특성:
- 성격과 특징: {self.iss}
- 성격: {self.personality}
- 말투: {self.speech}
- 생활 방식: {self.lifestyle}

{persona2.name}와의 관계:
- 관계 유형: {self.relationship_type}
- 친밀도: {self.closeness}/10
- 선호하는 상호작용 방식: {self.interaction_style}
- 함께하고 싶은 활동: {', '.join(self.preferred_activities) if self.preferred_activities else '없음'}
- 잠재적 갈등 요소: {', '.join(self.potential_conflicts) if self.potential_conflicts else '없음'}

대화 지침:
1. 당신의 성격과 말투를 일관되게 유지하세요
2. {persona2.name}와의 관계와 친밀도를 고려하여 대화하세요
3. 상호작용 스타일을 반영한 대화를 하세요
4. 잠재적 갈등 요소를 인지하고 적절히 대응하세요
5. 자연스럽고 감정이 담긴 대화를 해주세요
6. 응답할 때 자신의 역할명({persona1.name}:)을 앞에 붙이지 마세요


대화 시 주의사항:
- 친밀도가 낮다면 (5 미만) 더 조심스럽고 공손하게 대화하세요
- 친밀도가 높다면 (7 이상) 더 편하고 친근하게 대화하세요
- 잠재적 갈등 요소와 관련된 주제는 신중하게 다루세요

대화를 끝내야 하는 상황
1. 상대방이 작별 인사를 하면, 당신도 작별 인사로 응답하세요
2. 대화가 자연스럽게 마무리되는 시점

시뮬레이션이 종료되면
<END>
태그를 추가하세요.

"""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        self.llm = ChatOpenAI(model=model, temperature=0.7)

    def receive_message(self, message: str, sender: str) -> str:
        # 대화 기록에 메시지 추가
        self.chat_history.append({"sender": sender, "message": message})
        
        # 대화 기록을 프롬프트에 맞는 형식으로 변환
        formatted_history = []
        for msg in self.chat_history[:-1]:
            if msg["sender"] != self.name:
                formatted_history.append(HumanMessage(content=f"{msg['message']}"))
            else:
                formatted_history.append(AIMessage(content=msg['message']))
        
        # 응답 생성
        chain = self.prompt | self.llm
        response = chain.invoke({
            "chat_history": formatted_history,
            "input": message
        })
        
        # 응답을 대화 기록에 추가
        self.chat_history.append({"sender": self.name, "message": response.content})
        
        return response.content

class ConversationSimulation:
    def __init__(self):
        self.agents: Dict[str, ConversationAgent] = {}
    
    def add_agent(self, agent: ConversationAgent):
        self.agents[agent.name] = agent
    
    def generate_initial_message(self, speaker_agent: ConversationAgent) -> str:
        """선택된 페르소나의 성격에 맞는 대화 시작 메시지를 생성"""
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        prompt = f"""당신은 {speaker_agent.name}입니다.

당신의 특성:
{speaker_agent.iss}
말투: {speaker_agent.speech}

상황: 당신은 지금 {list(self.agents.keys())[0] if list(self.agents.keys())[0] != speaker_agent.name else list(self.agents.keys())[1]}를 만났습니다.
당신의 성격과 말투를 반영하여, 자연스러운 대화 시작 문장을 만들어주세요.

규칙:
1. 당신의 성격에 맞게 대화를 시작하세요
2. 간단한 인사나 안부를 물어보는 정도로 시작하세요
3. 너무 길지 않게 해주세요

응답 형식: 대화 시작 문장만 작성해주세요."""

        response = llm.invoke(prompt)
        return response.content.strip()
    
    def select_initial_speaker(self) -> ConversationAgent:
        """무작위로 초기 화자를 선택하거나, 성격을 고려하여 선택"""
        agents_list = list(self.agents.values())
        
        # 각 페르소나의 성격을 고려하여 가중치 부여
        weights = []
        for agent in agents_list:
            # 예를 들어, Joy는 대화를 시작하기 더 좋아할 수 있음
            if "Joy" in agent.name:
                weights.append(0.7)
            elif "Anger" in agent.name:
                weights.append(0.2)
            else:
                weights.append(0.1)
        
        # 가중치를 정규화
        total = sum(weights)
        weights = [w/total for w in weights]
        
        # 가중치를 기반으로 초기 화자 선택
        return random.choices(agents_list, weights=weights, k=1)[0]

    def simulate_conversation(self, turns: int = 10):
        """대화 시뮬레이션 시작"""
        # 초기 화자 선택
        initial_speaker = self.select_initial_speaker()
        
        # 초기 메시지 생성
        initial_message = self.generate_initial_message(initial_speaker)
        
        print(f"\n=== 대화 시작 ===")
        print(f"{initial_speaker.name}: {initial_message}")
        
        current_speaker = initial_speaker.name
        current_message = initial_message
        turn_count = 0
        
        while turn_count < turns:
            # 다음 화자 결정
            speakers = list(self.agents.keys())
            current_idx = speakers.index(current_speaker)
            next_speaker = speakers[(current_idx + 1) % len(speakers)]
            
            # 다음 화자의 응답 생성
            response = self.agents[next_speaker].receive_message(current_message, current_speaker)
            
            # 자연스러운 대화 흐름을 위한 짧은 대기
            time.sleep(0.5)
            
            # 응답 출력
            print(f"{next_speaker}: {response}")
            
            # END 태그가 있는 경우 대화 종료
            if "<END>" in response:
                print("\n=== 대화가 자연스럽게 종료되었습니다 ===")
                break
            
            # 대화 계속 진행
            current_speaker = next_speaker
            current_message = response
            turn_count += 1
            
            if turn_count >= turns:
                print(f"\n=== {turns}턴의 대화가 완료되었습니다 ===")


