import pandas as pd

df1 = pd.read_csv('healthcare-dataset-stroke-data.csv',sep=',')
df2 = pd.read_csv('stroke_prediction_dataset.csv',sep=',')


concat = pd.concat([df1,df2],ignore_index=True)

concat.to_csv('stroke.csv',index=False)

print("Birleştirilmiş veri 'stroke.csv' dosyasına kaydedildi.")