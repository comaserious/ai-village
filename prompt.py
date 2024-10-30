def create_location_prompt(persona, activity, duration, current_position):

    map_data = persona.get_map_data()
    zone_labels = persona.get_zone_labels()
    prompt = f"""
        {persona.scratch.get_str_iss()}

        {persona.name} 의 하루 일정 중 다음 활동에 대한 최적의 위치를 추천해주세요.

        공간정보 : {map_data}, {zone_labels}

        활동 : {activity}
        소요시간 : {duration} 분
        현재 위치 : {current_position}

        고려사항:
        1. 활동의 성격에 맞는 공간을 선택해주세요
            - 명상/휴식 → 조용한 공간
            - 사회적 활동 → 공용 공간
        2. 현재 위치에서의 접근성을 고려해주세요
        3. 해당 공간의 수용 인원을 확인해주세요

        다음 형식으로 응답해주세요.
        {{
            "zone" : "추천 장소명",
            "positon": [x,y],
            "duration": "체류 시간(분)"
            "reason" : "선택 이유"
        }}
    """

    return prompt

