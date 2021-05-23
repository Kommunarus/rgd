Главная цель - обучение сети. <br/>
Для этого используем файлы:

dataset.py - препроцессинг облаков  <br/> 
unit.py- даталоадер и сеть  <br/> 
script1.py - обучение <br/> 
eval_dataset.py и run_one_img.py - тест <br/> 

сеть выдает
1. степень открытия двери
2. класс объекта в портале
3. есть ли падение в щель

Но наша сетка не умеет искать несколько объектов, поэтому уходим из сетей и идем в openсv

1. readfiles.py - преобразование облаков в 2д
2. test doors_hak.ipynb - последовательное выделение контуров пола и щели около поезда и всех обектов выше их.
3. task1and2.py - тоже самое, только в виде питоновского скрипта


Запуск run_one_img.py выдает json.

/home/neptun/anaconda3/envs/Хакатон/bin/python /home/neptun/Документы/Хакатон/network/train/run_one_img.py
Predicted 1:      2
Predicted 2:      1
Predicted 3:      1
[(0.44782608695652176, 0.67, 1.0), (0.425, 0.5871428571428572, 1.0), (0.3782608695652174, 0.4092857142857143, 1.0)]
[(0.15217391304347827, 0.014285714285714285, 1), (0.19782608695652174, 0.037142857142857144, 1), (0.6304347826086957, 0.19428571428571428, 1)]

Process finished with exit code 0


{"figures": [{"object": "human", "geometry": {"position": {"x": 0.44782608695652176, "y": 0.67, "z": 1.0}, "rotation": {"x": 0, "y": 0, "z": 0}, "dimensions": {"x": 0.15217391304347827, "y": 0.014285714285714285, "z": 1}}, "door": "CLOSED"}]}


P.S. архив без датасетов (оставил по одной картинке) находится [здесь](https://drive.google.com/file/d/1SZy-ThDf_sLuwfd55Xl0WR9O0LOi54w2/view?usp=sharing)


Тестовое задание было сделано с помощью скрипта testtest.py. Находитмя в папке network/train
