import numpy
import math

def GathGeva(data,c,m=2,e=1e-4):
    N,n = data.shape
    X1 = numpy.ones((N,1))

    # Initialize fuzzy partition matrix
    if c.__class__ == int:  # Only number of clusters given
        mm = numpy.mean(data,0)
        aa = numpy.max(numpy.abs(data - numpy.ones((N,1))*mm),0)
        v = 2*(numpy.ones((c,1))*aa)*(numpy.random.rand(c,n) - 0.5) + numpy.ones((c,1))*mm
        
        d = numpy.empty((N,c))
        for j in range(c):
            xv = data - X1*v[j]
            d[:,j] = numpy.sum(xv*xv,1)
        d = (d + 1e-10) ** (-1/(m-1))
        
        f0 = d / (numpy.array([numpy.sum(d,1)]).T*numpy.ones((1,c)))
    else:
        f0 = c
        c = f0.shape[1]
        fm = f0**m
        sumf = numpy.sum(fm,0)
        #This is wrong, but not my fault
        v = numpy.dot(fm,data.T) / numpy.array([sumf for i in range(n)]).T

    f = numpy.zeros((N,c)) # partition matrix
    i = 0                  # iteration counter
    J = []
    while numpy.max(f0-f) > e:
        i += 1
        f = f0
        fm = f ** m
        sumf = numpy.sum(fm,0)
        v = numpy.dot(fm.T,data) / numpy.dot(numpy.array([sumf]).T,numpy.ones((1,n)))
        d = numpy.empty((N,c))
        for j in range(c):
            xv = data - numpy.dot(X1,numpy.array([v[j]]))

            # Calculate covariance matrix
            A = numpy.dot(numpy.multiply(numpy.ones((n,1)),fm[:,j]) * xv.T,xv)/sumf[j]
            #print A
            Pi = 1.0 / N * numpy.sum(fm[:,j])
            Apinv = numpy.linalg.pinv(A)
            Adet = numpy.linalg.det(Apinv)
            if Adet == 0.0:
                d[:,j] = numpy.array([ float('inf') for i in range(N) ])
            else:
                d[:,j] = 1.0/(numpy.linalg.det(Apinv) ** 0.5) * 1/Pi * numpy.exp(0.5*numpy.sum(numpy.dot(xv,Apinv) * xv,1))
        distout = numpy.sqrt(d)
        if m > 1:
            d = (d + 1e-10) ** (-1/(m-1))
        else:
            d = (d + 1e-10) ** (-1)
        f0 = d / numpy.array([ numpy.sum(d,1) for i in range(c) ]).T
        J.append(numpy.sum(f0*d))
    sumf = numpy.sum(f,0)

    P = numpy.zeros((c,n,n))
    V = numpy.zeros((c,n))
    D = numpy.zeros((c,n))

    for j in range(c):
        xv = data - numpy.array([ v[j,:] for i in range(N) ])
        #Calculate covariance matrix
        A = numpy.dot(numpy.ones((n,1))*f[:,j].T*xv.T,xv)/sumf[j]
        ed,ev = numpy.linalg.eig(A)
        ev = ev[:,min(range(c),key=lambda v: ed[v])]
        P[j] = A
        V[j,:] = ev.T
        D[j,:] = ed

    return (f0,distout,v,P,V,D,i,J)
    
#arr = numpy.array([[ 0.30378781,  0.02906274,  0.49884541,  0.98547817,  0.64672225],
#                   [ 0.55051508,  0.71226131,  0.34930771,  0.14759514,  0.00915009],
#                   [ 0.33132046,  0.83609986,  0.51956773,  0.39819306,  0.00175866],
#                   [ 0.44840285,  0.63916514,  0.36577865,  0.76332418,  0.53283004],
#                   [ 0.88475747,  0.9782427 ,  0.75628771,  0.69149094,  0.7511497 ]],dtype=numpy.dtype(complex))

#f0 = numpy.array([[1.649290092503832e-01,8.350709907496168e-01],
#                      [1.056159064399837e-01,8.943840935600162e-01],
#                      [2.117579407352767e-01,7.882420592647233e-01],
#                      [1.379460262615436e-01,8.620539737384565e-01],
#                      [5.620536542131311e-01,4.379463457868689e-01],
#                      [7.853513112261510e-01,2.146486887738492e-01],
#                      [6.744966914042189e-01,3.255033085957811e-01],
#                      [7.753563192547401e-01,2.246436807452598e-01]])

#arr = numpy.array([[0.6,3.9],
#                   [0.9,3.4],
#                   [1.2,4.1],
#                   [1.4,3.5],
#                   [3.5,0.5],
#                   [4.0,1.3],
#                   [4.1,0.4],
#                   [4.9,0.5]])

#print GathGeva(arr,f0)
