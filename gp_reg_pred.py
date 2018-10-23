# Suppose K is positive definate and the cholesky decomposition of K is K = LL', 
# where L is lower triangle matrix and L' is the transpose of L.
# Here we write the forward and backward substitution algorithm using L and L' 
# for solving linear system Kx=y, which equivalently means LL'x=y
# first solve Lb=y by forward substitution and then solve L'x=b by back substitution

# Forward substitution
def forward(L, y):
    N = len(y)
    b = np.zeros(N)
    for i in range(N):
        b[i] = ( y[i] - sum(L[i,:]*b) ) / L[i,i] 
    
    return b

# Back substitution
def back(U, b):
    N = len(b)
    x = np.zeros(N)
    for n in reversed(range(N)):
        x[n] = ( b[n] - sum(U[n,:]*x) ) / U[n,n]
    
    return x
    
# Gussian Process Prediction:
# Input: X (inputs); y (targets); K (covariance function); sigma_2 (noise level); x_new (test input)

def GP(X, y, Kernel, sigma_2, x_new):
    N = len(y)
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            K[i,j] = Kernel(X[i], X[j])
            K[j,i] = K[i,j]

    K = K + sigma_2*np.eye(N)
    L = np.linalg.cholesky(K)
    b = forward(L, y)
    alpha = back(L.transpose(), b)
    
    mu_new = []
    var_new = []
    num = 0
    for point in x_new:
        num += 1
        kx = np.zeros(N)
        for i in range(N):
            kx[i] = Kernel(point, X[i])
            
        mu_point = sum(kx*alpha)
        v = forward(L, kx)
        var_point = Kernel(point, point) - sum(v*v) + sigma_2
        
        mu_new.append(mu_point)
        var_new.append(var_point)
        if num % 10 == 0:
            print(num)
    
    return np.array(mu_new), np.array(var_new)

# The following example kernels are used in question 4 of hw2 Stat547P
tau_2 = 20
nu_2 = 200
ell = 1
def Kernel_b(x_1, x_2):
    dif = x_1 - x_2
    k = tau_2 * np.exp(-dif**2/(2*nu_2) - (np.sin(np.pi*dif/ell))**2)
    return k

eta_2 = 1000

def Kernel_d(x_1, x_2):
    k = Kernel_b(x_1, x_2) + eta_2*(1 + x_1*x_2)         
    return k

f = open("maunaloa_clean.txt")
maunaloa = np.loadtxt(f, delimiter=" ")
year = maunaloa[:,0]
co2 = maunaloa[:,1]
year_new = np.linspace(2000, 2040, 1000)
mu_1, var_1 = GP(year, co2, Kernel_b, 1, year_new)
mu_2, var_2 = GP(year, co2, Kernel_d, 1, year_new)
plt.plot(year,co2)
plt.plot(year_new, mu_2)
plt.fill_between(year_new, mu_2-var_2, mu_2+var_2, alpha=0.5)
plt.axvline(x = 2025, linewidth=4, color='r')
