import os
import json

from brisk import config_dir, root_dir
from brisk.utils.cl import print_error

# Class for the brisk subject
class BriskSubject():

    def __init__(self, name) -> None:
        
        self.name = name
        self.db_path = os.path.join(root_dir, '_archive', name)
        self.trials = []
        self.raw_data = []
        self.segmented_data = []
        self.parameters = []
        with open(os.path.join(config_dir,'imu_std.json'), 'r') as f:
            self.imu_config = json.load(f)
        self.fs = 102.4
        self.samples_per_cycle = 100

    def __str__(self) -> str:
        return self.name.title()
    
    def import_from_archive(self):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the database.')
