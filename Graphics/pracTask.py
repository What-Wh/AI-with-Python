import matplotlib.pyplot as plt
import numpy as np

# Task 1
x = np.linspace(-10, 10, 500)
y = np.sin(x) * (x ** 2)

plt.plot(x,y)
plt.title("Графік функції sin(x) * x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Task 2
data = np.random.normal(5, 2, 1000)

plt.hist(data, bins=30)
plt.title("Gigroma")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Task 3
hobbies = ['Football', 'Gym', 'Comp. games', 'Programming', 'Lego']
sizes = [40, 20, 15, 15, 10]

plt.pie(sizes, labels=hobbies, autopct='%1.1f%%', startangle=90)
plt.title("Hobbies")
plt.axis('equal')
plt.show()

# Task 4
f1 = np.random.normal(70, 10, 100)
f2 = np.random.normal(30, 20, 100)
f3 = np.random.normal(50, 5, 100)
f4 = np.random.normal(70, 8, 100)

data = [f1, f2, f3, f4]

plt.boxplot(data, labels=['Apples', 'Oranges', 'Bananas', 'Lemons'])
plt.title("Wieght of Friuts")
plt.xlabel("Friuts")
plt.ylabel("Wieght")
plt.grid(True)
plt.show()

# Task 6
x = np.linspace(-10, 10, 500)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) + np.cos(x)

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.title("Графік функції")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['sin(x)', 'cos(x)', 'sin(x) + cos(x)'])
plt.grid(True)
plt.show()