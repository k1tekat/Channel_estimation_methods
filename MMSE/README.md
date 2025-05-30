# Методы оценки канала

*В работе проводится сравнительный анализ двух методов оценки канала связи, включая метод наименьших квадратов и метод, основанный на среднеквадратичном отклонении. Модуль оценки канала играет ключевую роль в обеспечении точности демодуляции принятых сигналов. Поскольку стандарты связи не регламентируют его реализацию, разработка эффективных алгоритмов оценки канала остается задачей производителей телекоммуникационного оборудования.*
##
Для оценки канала можно использовать обучающие символы, которые обычно обеспечивают хорошую производительность. 
 Однако эффективность их передачи снижается из-за дополнительных затрат на обучающие символы, такие как преамбула или пилотные тона, которые передаются в дополнение к символам данных.  Для оценки канала при наличии обучающих символов широко используются методы наименьших квадратов (LS) и минимальной среднеквадратичной ошибки (MMSE). 
Мы предполагаем, что все поднесущие являются ортогональными (т. е. не содержат межсимвольных интерференций). Тогда обучающие символы для
N поднесущих могут быть представлены в виде следующей диагональной матрицы:

```math
\mathbf{X} =  
 \begin{vmatrix}
 \mathbf{X[0]} & 0 & \mathbf{ \cdots }& 0
 \\0 & X[1] &  & \vdots
 \\\vdots & & \ddots & 0
 \\ 0 & \cdots & 0 & X[N-1]
 \end{vmatrix} 
```
