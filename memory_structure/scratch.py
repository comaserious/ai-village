import json

class Scratch:
    def __init__(self , scratched_saved):

        self.s_mem_data = scratched_saved

        self.curr_time = None;
        self.curr_tile = None;
        self.daily_req = [];
        self.f_daily_req = [];
        self.f_daily_schedule_hourly_org= [];
    
        self.lifestyle = None;
        self.name = None;
    

    
    
    def get_str_iss(self):
        common_state = '';
        common_state += f"Name : {self.s_mem_data['name']} \n"
        common_state += f"Age : {self.s_mem_data['age']} \n"
        common_state += f"Personality : {self.s_mem_data['personality']} \n"
        common_state += f"Speech : {self.s_mem_data['speech']} \n"
        common_state += f"Life Style : {self.s_mem_data['lifestyle']} \n"
        common_state += f"Gender : {self.s_mem_data['gender']} \n"
        common_state += f"Character : {self.s_mem_data['character']} \n"

        return common_state
    
    def get_str_lifestyle(self):
        return self.s_mem_data['lifestyle']
