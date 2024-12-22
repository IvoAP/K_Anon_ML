class AdultsConfig:
    qi = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country']
    target = 'income'
    path = 'data/adults.csv'

class  IotMedicalConfig:
    qi = [
    'ip.src', 'ip.dst',         
    'tcp.srcport', 'tcp.dstport',  
    'frame.time_relative', 'frame.len',  
    'mqtt.topic', 'mqtt.topic_len',  
    ]
    
    target = 'label'

