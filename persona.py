import json
from memory_structure.scratch import *
from global_method import *
from run_gpt import *
from plan import *
from spatial_memory.spatial import *
from spatial_memory.maze import *

# persona 객체 생성

class Persona:
    def __init__(self , name , user) -> None:

        scratch_saved = f"memory_storage/{user['uid']}/{name}/scratch.json"
        
        if (check_file_exists(scratch_saved)):
            scratch_saved = json.load(open(scratch_saved))
        else :
            first_day_persona(name, user)
            scratch_saved = json.load(open(scratch_saved))

        self.name = name
        self.scratch = Scratch(scratch_saved)
        self.spatial_memory = SpatialMemory(map_matrix, zone_labels)
        self.daily_plan_count = 0 

    def plan(self, name, new_day , user):
        return plan(self ,new_day , user)
    
    def get_map_data(self):
        return self.spatial_memory.map_matrix
    
    def get_zone_labels(self):
        return self.spatial_memory.zone_labels
    
    
    






        






    