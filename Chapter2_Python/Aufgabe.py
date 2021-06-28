import numpy as np
import matplotlib.pyplot as plt


#Für jeden Wert in der Liste neuen e-Wert errechnen
def e_function(my_list):
    return [np.exp(val) for val in my_list]


#Funktion plotten
def plot_func(x, y, farbe, windowName):
    plt.figure(windowName)
    plt.plot(x, y, color=farbe)
    plt.title("My Image")
    plt.xlabel("x")
    plt.ylabel("e(x)")
    plt.show()


a = 1
b = 5

mylist = np.array(range(a, b + 1), dtype=np.int8)

e_list = e_function(mylist)

plot_func(mylist, e_list, "black", "MyWindowName")
