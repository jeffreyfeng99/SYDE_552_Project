import os
import pandas as pd
import numpy as np

def parse_name(input):
    split = input.split('_minerror_SAM_')
    return split[0] + split[-1]

if __name__=='__main__':
    input_dir = './output_cifar/04242022'
    output_dir = './clean_figures'
    output_name = 'parsed.csv'

    models = {}

    for root, dir, files in os.walk(input_dir):

        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                df = pd.read_csv(file_path)

                model = os.path.split(os.path.dirname(file_path))[-1]
                model = parse_name(model)

                if model not in models.keys():
                    models[model] = {}
                
                if 'localization' in file:
                    errors = np.array(df['errors'].tolist())
                    models[model]['localization'] = np.mean(errors)
                    
                elif 'predictions' in file:
                    preds = np.array(df['predictions'].tolist())
                    labels = np.array(df['labels'].tolist())
                    models[model]['accuracies'] = np.mean(preds==labels)
                    
                else:
                    raise NotImplementedError
    

    df = pd.DataFrame(models)
    df.to_csv(os.path.join(output_dir, output_name))