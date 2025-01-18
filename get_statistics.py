import os

root_path='./../CompCars/data/image'

make_dict={}
model_dict={}

tot=0

for dirpath,_,filenames in os.walk(os.path.join(os.getcwd(),root_path)):
    
    for file in filenames:
        
        tot+=1
        
        make=dirpath.split('/')[-3]
        model=dirpath.split('/')[-2]
        
        if make in make_dict:
            
            make_dict[make].append(file)
            
        else:
            
            make_dict[make]=[]
            
        if model in model_dict:
            
            model_dict[model].append(file)
            
        else:
            
            model_dict[model]=[]
            
print('Number of makes: ',len(make_dict.keys()))
print('Number of models: ',len(model_dict.keys()))
print('Number of images: ',tot)
print('###############################################')

root_path='./../CompCars/data/part'

make_dict={}
model_dict={}
part_dict={}

tot=0

for dirpath,_,filenames in os.walk(os.path.join(os.getcwd(),root_path)):
    
    for file in filenames:
        
        tot+=1
        
        make=dirpath.split('/')[-4]
        model=dirpath.split('/')[-3]
        part=dirpath.split('/')[-1]
        
        if make in make_dict:
            
            make_dict[make].append(file)
            
        else:
            
            make_dict[make]=[]
            
        if model in model_dict:
            
            model_dict[model].append(file)
            
        else:
            
            model_dict[model]=[]
            
        if part in part_dict:
            
            part_dict[part].append(file)
            
        else:
            
            part_dict[part]=[]
            
print('Number of makes: ',len(make_dict.keys()))
print('Number of models: ',len(model_dict.keys()))
print('Number of parts: ',len(part_dict.keys()))
print('Number of images: ',tot)
print('###############################################')

root_path='./../CompCars/sv_data/image'

make_dict={}
model_dict={}

tot=0

for dirpath,_,filenames in os.walk(os.path.join(os.getcwd(),root_path)):
    
    for file in filenames:
        
        tot+=1
        
        model=dirpath.split('/')[-1]
        
        if make in make_dict:
            
            make_dict[make].append(file)
            
        else:
            
            make_dict[make]=[]
            
        if model in model_dict:
            
            model_dict[model].append(file)
            
        else:
            
            model_dict[model]=[]
            
print('Number of makes: ',0)
print('Number of models: ',len(model_dict.keys()))
print('Number of images: ',tot)