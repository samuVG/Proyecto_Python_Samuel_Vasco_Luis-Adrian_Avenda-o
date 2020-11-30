#!/usr/bin/env python
# coding: utf-8

# In[10]:


class PozoDePotencial:
    
    def __init__(self,z,N,hbar2,h,a,m,v0):
        self.z=z   # se crean los atributos de las coordenadas
        self.N=N
        self.hbar2=hbar2
        self.h=h
        self.a=a
        self.m=m
        self.v0=v0
        
        
    def MatrizHamiltoniana(self,Potencial):
        import numpy as np
        aa = [0]+[-self.hbar2/(2*self.m*self.h**2)]*(self.N-1)   # diagonal superior, primer elem debe ser cero
        b= [self.hbar2/(self.m*self.h**2)]*self.N             # diagonal central 
        c= [-self.hbar2/(2*self.m*self.h**2)]*(self.N-1)+[0]    # diagonal inferior, ultimo elem debe ser cero

        K = np.diag(aa[1:], 1) + np.diag(b, 0) + np.diag(c[:-1], -1) # matriz tridiagonal energía cinética del problema de autovalores.
        V = []
        for i in self.z:
            V.append(Potencial(i,self.v0))
        V=np.diag(V) #matriz diagonal energía potencial
        H=K+V
        return H #matriz de Hamilton 
    
    def Autovalores(self,H):
        import scipy.linalg as LA #se importa la libreria para resolver el problema de valores y vectores propios
        import numpy as np

        
        K=LA.eig(H)   #el metodo de la libreria devuelve un vector de dos elementos, donde el primero son los autovalores       
        i=np.argsort(K[0]) #ordena de menor a mayor, y es un arreglo de las posiciones de esos valores.
        autovalores=[]
        for j in range(4):
            autovalores.append(K[0][i[j]])
        
        return autovalores   # y el segundo son las autofunciones correspondientes a dichos autovalores.
    
    def Autofunciones(self,H):
        import scipy.linalg as LA
        import numpy as np
        
        K=LA.eig(H)
        i=np.argsort(K[0])
        autofunciones=[]
        for j in range(4):
            autofunciones.append(K[1][:,i[j]])
        return autofunciones
    
    def DensidadProbabilidad(self,H):
        import numpy as np
        return abs(np.array(self.Autofunciones(H)))**2  #modulo cuadrado de las autofunciones
    
    def AutofuncionTemporal(self,H,t):
        import numpy as np
        Cn  = np.array ([1/2,1/2,1/2,1/2]) # Condición de la escogencia, la suma al cuadrado de los 4 coeficientes es igual a 1
        autoFnTemp = 0
        for i in range(4):
            autoFnTemp += Cn[i]*np.array(self.Autofunciones(H))[i]*np.exp(-1j*np.array(self.Autovalores(H))[i]*t/np.sqrt(self.hbar2))
        return autoFnTemp

    def GraficadorAutofunciones(self,H,Potencial):
        import matplotlib.pyplot as plt
        import numpy as np
        
        V = []
        for i in self.z:
            V.append(Potencial(i,self.v0))
        
        font = {'weight' : 'bold', 'size'   : 19}
        plt.matplotlib.rc('font', **font)
        plt.figure(figsize=(12,8))
        for i in range(4):
            plt.plot(self.z,np.array(self.Autofunciones(H))[i]+np.array(self.Autovalores(H))[i], label="E%0.0f =%0.2f [EV]"%(i+1,np.array(self.Autovalores(H))[i]))  
        plt.plot(self.z, V,color='black' )
        if self.v0!=0:
            plt.xlim(-.6e-9,.6e-9)
        if self.v0>1e3:
            plt.ylim(-1,30)
        plt.title("GRÁFICA DE LAS AUTOFUNCIONES")
        plt.xlabel("z[m]")
        plt.ylabel("V(z)[EV]")
        plt.grid()
        plt.legend()
        plt.show()

        font = {'weight' : 'bold', 'size'   : 16}
        plt.matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(14,12))
        fig.suptitle('ZOOM en los autovalores')
        fig.subplots_adjust(hspace=.6)

        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(self.z,np.array(self.Autofunciones(H))[i]+np.array(self.Autovalores(H))[i], label="E%0.0f =%0.2f [EV]"%(i+1,np.array(self.Autovalores(H))[i]))
            plt.plot(self.z, V ,color='black')
            plt.ylim(np.array(self.Autovalores(H))[i]-0.1,np.array(self.Autovalores(H))[i]+0.1)
            plt.grid()
            plt.xlabel("z[m]")
            plt.ylabel("$\psi(z)$")
            plt.legend()
        plt.show()
        
    def GraficadorDensidadProbabilidad(self,H,Potencial):
        import matplotlib.pyplot as plt
        import numpy as np
        
        V = []
        for i in self.z:
            V.append(Potencial(i,self.v0))
        
        font = {'weight' : 'bold', 'size'   : 19}
        plt.matplotlib.rc('font', **font)
        plt.figure(figsize=(12,8))
        for i in range(4):
            plt.plot(self.z,np.array(self.DensidadProbabilidad(H))[i]+np.array(self.Autovalores(H))[i], label="E%0.0f =%0.2f [EV]"%(i+1,np.array(self.Autovalores(H))[i]))
        plt.plot(self.z, V ,color='black')
        plt.xlim(-.6e-9,.6e-9)
        if self.v0>1e3:
            plt.ylim(-1,30)
        plt.grid()
        plt.title("GRÁFICA DE LAS DENSIDADES DE PROBABILIDAD")
        plt.xlabel("z[m]")
        plt.ylabel("V(z)[EV]")
        plt.legend()
        plt.show()

        font = {'weight' : 'bold', 'size'   : 16}
        plt.matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(14,12))
        fig.suptitle('ZOOM en los autovalores')
        fig.subplots_adjust(hspace=.6)

        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(self.z,np.array(self.DensidadProbabilidad(H))[i]+np.array(self.Autovalores(H))[i], label="E%0.0f =%0.3f [EV]"%(i+1,np.array(self.Autovalores(H))[i]))
            plt.plot(self.z, V ,color='black')
            plt.ylim(np.array(self.Autovalores(H))[i]-0.002,np.array(self.Autovalores(H))[i]+0.0065)
            plt.xlabel("z[m]")
            plt.ylabel("$|\psi(z)|^2$")
            plt.grid()
            plt.legend()
        plt.show()
    
    
    def GraficadorAutofuncionTemporal(self,H,Potencial):
        import matplotlib.pyplot as plt
        import numpy as np
        
        V = []
        for i in self.z:
            V.append(Potencial(i,self.v0))
            
        font = {'weight' : 'bold', 'size'  :13}
        plt.figure(figsize=(12,8))
        for i in range(4):
            plt.plot(self.z,abs(np.array(self.AutofuncionTemporal(H,i)))**2, label="$\Psi(z,t), t=%0.0f$"%i)
        plt.plot(self.z,V,color='black')
        plt.ylim(-0.002,0.016)
        plt.title("GRÁFICAS DE LA DENSIDAD DE PROBABILIDAD TEMPORAL")
        plt.xlabel("z[m]")
        plt.ylabel("$|\psi(z,t)|^2$")
        plt.legend()
        plt.grid()
        plt.show()


# In[ ]:




