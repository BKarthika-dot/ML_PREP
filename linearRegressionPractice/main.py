import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('data.csv')

def loss_function(m,b,points):   #finding total error function 
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].studytime
        y=points.iloc[i].marks
        total_error+=(y-(m*x+b))**2

    total_error=total_error/float(len(points))
    return total_error


def gradient_descent(m_now,b_now,points,L):   #L is the learning rate
    m_gradient=0  #partial derivatives
    b_gradient=0

    n=len(points)

    for i in range (n):
        x=points.iloc[i].studytime
        y=points.iloc[i].marks


        m_gradient+= -(2/n)*x*(y-(m_now*x + b_now))
        b_gradient+= -(2/n)*(y-(m_now*x+b_now))

    m=m_now-m_gradient*L
    b=b_now-b_gradient*L

    return m,b

m=0
b=0
L=0.0001
epochs=1000 #no of iterations


for i in range(epochs):
    if i%50==0:
        print(f"Epoch: {i}")
    m,b=gradient_descent(m,b,data,L)

print(m,b)
plt.scatter(data.studytime,data.marks,color="blue")
plt.plot(list(range(0,100)),[m*x + b for x in range(0,100)],color="red")
error=loss_function(m,b,data)
print(f"Total Error: {error} ")
plt.show()











