import json
import sys
import os.path

class JsonManager:
    
    def __init__(self, filename):
        self.filename = filename
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w') as file:
                json.dump({
                    'results': []
                    }, file)
                
    def read_data(self):
        with open(self.filename, 'r') as file:
            return json.load(file)
        
    def write_data(self, data):
        with open(self.filename, 'w') as file:
            json.dump(data, file, indent=4)
            
    def update_data(self, key, value):
        data = self.read_data()
        results_list = data['results']
        results_list.append(value)
        self.write_data(data)
        
        
