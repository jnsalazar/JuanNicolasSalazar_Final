import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

data = np.genfromtxt('datos_observacionales.dat')
t_obs = data[:,0]
vec_obs = data[:,1:]
#Condiciones iniciales
vec0_obs = vec_obs[0,:]

#Sistema de ecuaciones diferenciales acopladas
def equations(vec, t, sigma, rho, beta):
    x, y, z = vec
    dqdt = [sigma*(y - x), rho*(x - z) - y, x*y - beta*z]
    return dqdt

#Retorna el valor de x, y, z en el tiempo t
def model(t, param): 
    times = np.linspace(0, t, 100)
    sol = odeint(equations, vec0_obs, times, args = (param[0], param[1], param[2]))
    return sol[-1:][0]

def loglikelihood(t_obs, vec_obs, param):
    r = []
    for i in range(1, len(t_obs)):
        d = 0
        d = vec_obs[i] - model(t_obs[i], param) #sigma de los datos es 1 siempre.
        d = -0.5*np.dot(d,d) #es el caso de normal multivariada y asumo independencia.
        r.append(d)
    r = np.array(r)
    return np.sum(r)

#Todos los parametros deben ser positivos menores que 30.
def logprior(param):
    p = -np.inf
    mayor = np.amax(param)
    minim = np.amin(param)
    if(mayor <= 30, minim >= 0):
        p = 0.0
    return p

def H(t_obs, vec_obs, param, param_p): #param_p es los momentos asociados
    m = 200.0
    K = (-0.5*m)*np.dot(param_p, param_p)
    V = -loglikelihood(t_obs, vec_obs, param)
    return K + V

#Derivada del hamiltoniano con respecto a q, que en este caso es param.
#Como lo unico que depende de q en este hamiltoniano es -loglikelihood
#y depende de varias variables, la derivada en este caso es un gradiente,
#por lo que se deben calcular derivadas parciales y retornar un vector.
def dHdq(t_obs, vec_obs, param):
    n = len(param)
    grad = np.zeros(n)
    delta_t = 1E-5
    for i in range(n):
        delta_param = np.zeros(n)
        delta_param[i] = delta_t
        #Se suma y se resta delta_param para que justamente sea el valor de la derivada parcial en param.
        grad[i] = loglikelihood(t_obs, vec_obs, param + delta_param) 
        grad[i] = grad[i] - loglikelihood(t_obs, vec_obs, param - delta_param)
        grad[i] = grad[i]/(2.0 * delta_t)
    return grad

def dHdp(param_p):
    m = 200.0
    return param_p/m

#Resuelve las ecuaciones de kick y drift para proponer nuevos param y param_p
def leapfrog(t_obs, vec_obs, param, param_p):
    pasos = 4
    delta_t = 1E-2
    param_next = param
    param_p_next = param_p
    for i in range(pasos):
        param_p_next = param_p + dHdq(t_obs, vec_obs, param)*delta_t*0.5
        param_next = param + dHdp(param_p)*delta_t
        param_p_next = param_p_next + dHdq(t_obs, vec_obs, param)*delta_t*0.5
    param_p_next = -param_p_next
    return param_next, param_p_next

def MCH(t_obs, vec_obs, N):
    cadena_param = [30*np.random.random(size = 3)]
    cadena_param_p = [np.random.normal(size = 3)]
    
    for i in range(1, N):
        prop_param, prop_param_p = leapfrog(t_obs, vec_obs, cadena_param[i-1], cadena_param_p[i-1])
        E_next = H(t_obs, vec_obs, prop_param, prop_param_p)
        E_now = H(t_obs, vec_obs, cadena_param[i-1], cadena_param_p[i-1])
        r = min(1, np.exp(-(E_next - E_now)))
        alpha = np.random.random()
        if(alpha < r):
            cadena_param.append(prop_param)
        else:
            cadena_param.append(cadena_param[i-1])
        cadena_param_p.append(np.random.normal(size = 3))
    
    return np.array(cadena_param)

#Con 100 ya se demora como 5 minutos, pero compila.
chain = MCH(t_obs, vec_obs, 100)



modelo = odeint(equations, vec0_obs, np.linspace(0,3,1000), 
                args = (np.mean(chain[:,0]), np.mean(chain[:,1]), np.mean(chain[:,2])))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,1], data[:,2], data[:,3])
ax.plot(modelo[:,0], modelo[:,1], modelo[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.hist(chain[:,0], bins = 40, density = True)
plt.hist(chain[:,1], bins = 40, density = True)
plt.hist(chain[:,2], bins = 40, density = True)