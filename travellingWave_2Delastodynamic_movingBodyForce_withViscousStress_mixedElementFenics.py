# Janel Chua
# Able to handle supersonic (with respect to pressure wave speed)
##################################################################################
# Preliminaries and mesh
from dolfin import *
import numpy as np
mesh_half = Mesh('Trial19.xml')

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Manually introduce the material parameters
class GC(UserExpression):
    def set_Gc_values(self, Gc_0, Gc_1):
        self.Gc_0, self.Gc_1 = Gc_0, Gc_1
    def eval(self, value, x):
        "Set value[0] to value at point x"
        tol = 1E-14
        if x[1] >= 0.015 + tol:
            value[0] = self.Gc_0
        elif x[1] <= -0.015 + tol:
            value[0] = self.Gc_0
        else:
            value[0] = self.Gc_1 #middle layer
# Initialize Gc
Gc = GC()
Gc.set_Gc_values(0.081, 0.081) #N/mm 

l     = 0.04 # mm
E     = 5.3*1000 # MPa = MKg.m^-1.s^-2 # we are using N/mm^2
nu    = 0.35 # Poisson's ratio
mu    = Constant(E / (2.0*(1.0 + nu))) # N/mm^2
lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))) # N/mm^2
mu2   = 0 # this mu is used for the evolution equation

# Mass density
rho   = Constant(1.23*10**(-9)) # Mkg.m^-3  # we are using (N.s^2)/mm^4

# Rayleigh damping coefficients
eta_m = Constant(0)
eta_k = Constant(1*10**(-9)) 

# Small eta for non 0 stiffness
eta_e = 1e-3

# Generalized-alpha method parameters
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant(0.5+alpha_f-alpha_m)
beta    = Constant((gamma+0.5)**2/4.)

# Time-stepping parameters
T       = 20 
Nsteps  = 100 
dt = Constant(T/Nsteps)

tract = Constant((-0.001, 0)) # horizontal traction

##################################################################################
# Define Function Spaces and mixed elements
FE_A  = VectorElement("CG", mesh_half.ufl_cell(), 1)
FE_U  = VectorElement("CG", mesh_half.ufl_cell(), 1)
Z = FunctionSpace(mesh_half, MixedElement([FE_A, FE_U]))

dU = TrialFunction(Z)
(A, U) = split(dU)
U_ = TestFunction(Z)
(V, W) = split(U_)

old = Function(Z)
(Aold, Uold) = old.split(True)
new = Function(Z)
(Anew, Unew) = new.split(True)

strain = TensorFunctionSpace(mesh_half, "CG", 1)
strain_total = Function(strain, name='Strain')
sigma_fs = TensorFunctionSpace(mesh_half, "CG", 1)
stress_elas = Function(sigma_fs, name='Stress')
VV = FunctionSpace(mesh_half, 'CG', 1)
normSquared = Function(VV)

# Defining velocity of moving defect
vel = as_vector([1000000,0]) 
velx = vel[0] 

# Defining bodyForce
Amp = 10
x0 = 2.5
y0 = 0
sigx = 0.01
sigy = 0.01
bodyForce = Expression(('Amp*exp( -( pow((x[0] - x0),2)/(2*pow(sigx,2)) + pow((x[1] - y0),2)/(2*pow(sigy,2)) ) )','0'), degree =2, Amp=Amp, x0=x0,y0=y0,sigx=sigx,sigy=sigy) # Gaussian bodyForce
#bodyForce = Expression(('x[1] >= -0.05 and x[1] <= 0.05 and x[0] >= 2.45 and x[0] <= 2.55? 10: x[1]< 0 ? 0: 0','0'), degree = 2) # Square bodyForce 

##################################################################################
# Boundary conditions
def top(x,on_boundary):
    return near(x[1],1.2) and on_boundary #(x[0], 1.2)
def bot(x,on_boundary):
    return near(x[1],-1.2) and on_boundary #(x[0], -1.2)

def left(x,on_boundary):
    return near(x[0],-0.5) and on_boundary #(-0.5, x[1])
def leftTopHalf(x,on_boundary):
    return near(x[0],-0.5) and (x[1] > 0) and on_boundary #(-0.5, x[1])
def leftBotHalf(x,on_boundary):
    return near(x[0],-0.5) and (x[1] < 0) and on_boundary #(-0.5, x[1])
def right(x,on_boundary):
    return near(x[0],5) and on_boundary #(5, x[1])

def leftcorner(x,on_boundary):
    tol=1E-15
    return (abs(x[0]+0.5) < tol) and (abs(x[1]+1.2)<tol) #(-0.5,-1.2)
def rightcorner(x,on_boundary):
    tol=1E-15
    return (abs(x[0]-5) < tol) and (abs(x[1]+1.2)<tol) #(5,-1.2)

def Crack(x):
    return abs(x[1]) < 5e-03 and x[0] <= -0.25 # no initial boundary crack
def imposedPhi(x):
    return abs(x[1]) < 5e-03 and x[0] <= 2 # no initial boundary crack

loadtop = Expression("t", t = 0.0, degree=1)
loadbot = Expression("t", t = 0.0, degree=1)
loadleft = Expression("t", t = 0.0, degree=1)
loadright = Expression("t", t = 0.0, degree=1)

stretch0 = Expression(("0.1*x[0]"),degree=1)
stretch1 = Expression(("0.0015"),degree=1)
stretch2 = Expression(("-0.0015"),degree=1)
stretch3 = Expression(("0"),degree=1) 

bcright = DirichletBC(Z.sub(1).sub(0), Constant(0), right) #Right displacement loaded
bcright2 = DirichletBC(Z.sub(1).sub(1), Constant(0),  right) #Right displacement loaded

# Dirichlet Boundary Condition used:
bc_U = [bcright, bcright2]  
bc_A = []

n = FacetNormal(mesh_half)
# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh_half, mesh_half.topology().dim() - 1)
boundary_subdomains.set_all(0)

AutoSubDomain(top).mark(boundary_subdomains, 1) # top boundary
AutoSubDomain(bot).mark(boundary_subdomains, 2) # bottom boundary
AutoSubDomain(left).mark(boundary_subdomains, 3) # left boundary
AutoSubDomain(leftTopHalf).mark(boundary_subdomains, 31) # leftTop boundary
AutoSubDomain(leftBotHalf).mark(boundary_subdomains, 32) # leftBot boundary
AutoSubDomain(right).mark(boundary_subdomains, 4) # right boundary

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

##################################################################################
# Constituive functions
def epsilon(U):
    return sym(grad(U))
def epsilonDot(a):
    return -sym(grad(a))
def sigma(U, a):
    elastic = 2.0*mu*epsilon(U)+lmbda*tr(epsilon(U))*Identity(len(U))
    dissipative = eta_k*( 2.0*mu*epsilonDot(a)+lmbda*tr(epsilonDot(a))*Identity(len(U)) )
    return (elastic + dissipative)
def psi(U):
    return 0.5*lmbda*(tr(epsilon(U)))**2 + mu*inner(epsilon(U),epsilon(U)) # isotropic linear elastic

def W0(p, u):
    Epsilon = variable(sym(grad(u)))
    P = variable(p)
    Psi = 0.5*lmbda*(tr(Epsilon))**2 + mu*inner(Epsilon,Epsilon) 
    w0 = ((1-P)**2 + eta_e)*Psi + (Gc/(2*l))*P**2 + ((Gc*l)/2)*(grad(P))**2
    stress = diff(w0, Epsilon)
    W0_diffPhi = diff(w0, P) # This is partial difference
    return [w0, stress, W0_diffPhi]

# Mass term weak form
def m(U, a, W, vel):
    return rho*inner(a, div(outer(W, vel)) )*dx
# Elastic stiffness weak form
def k(U, a, W):
    return inner(sigma(U, a), grad(W))*dx 
# Body Force weak form
def b(W):
    return inner(bodyForce,W)*dx # Body Force, a gaussian
# Rayleigh damping form
def c(U, a, W):
    return eta_m*m(U, W) + eta_k*k(U, a, W)
# Work of external forces
def Wext(Pold, W):
    return ((1.0-Pold)**2)*dot(W, tract)*dss(1)
def H_l(x):
    return (1/2)*(1 + tanh(x/l))

# Boundary Terms
def boundaryTop(U, W, a):
    stress = sigma(U, a)
    e1 = as_vector([1,0])
    e2 = as_vector([0,1])
    n = as_vector([0,1])
    return +( inner( (stress[0,1]*n[1]), (dot(W,e1)) ) + inner( (stress[1,1]*n[1]), (dot(W,e2)) )   )

def boundaryBot(U, W, a):
    stress = sigma(U, a)
    e1 = as_vector([1,0])
    e2 = as_vector([0,1])
    n = as_vector([0,-1])
    return +( inner( (stress[0,1]*n[1]), (dot(W,e1)) ) + inner( (stress[1,1]*n[1]), (dot(W,e2)) )   )

def boundaryLeft(U, W, velx, a):
    stress = sigma(U, a)
    e1 = as_vector([1,0])
    e2 = as_vector([0,1])
    n = as_vector([-1,0])
    return -rho*velx*a[0]*dot(W,e1)*n[0] - rho*velx*a[1]*dot(W,e2)*n[0] #\
    #+( inner( (stress[0,0]*n[0]), (dot(W,e1)) ) + inner( (stress[1,0]*n[0]), (dot(W,e2)) )   )

##################################################################################
# Weak form for a
E_A = ( inner(dot(grad(U),vel), V) - inner(A, V) )*dx
# Weak form for u
E_U = m(U, A, W, vel) - k(U, A, W) + b(W)\
        + ( boundaryLeft(U, W, velx, A) )*dss(3) 

form = E_A + E_U
a_form = lhs(form)
L_form = rhs(form)

##################################################################################
# Initialization of the iterative procedure and output requests
time = np.linspace(0, T, Nsteps+1)
tol = 1e10 

#Name of file used for storage
store_u = File ("mydata/u.pvd")
store_a = File ("mydata/a.pvd")
store_stress_elas = File ("mydata/stress_elas.pvd")
store_strain_total = File ("mydata/strain_total.pvd")
store_normSquared = File ("mydata/normSquared.pvd")

##################################################################################
# Storing things at t = 0:
store_a << Aold
store_u << Uold # mm

stress_elas.assign(project(sigma(Uold, Aold) ,sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
store_stress_elas << stress_elas

strain_total.assign( project(epsilon(Uold), strain, solver_type="cg", preconditioner_type="amg"))                     
store_strain_total << strain_total

normSquared.assign( project( dot(Uold, Uold), VV, solver_type="cg", preconditioner_type="amg"))
store_normSquared << normSquared
print ('Saving initial condition')

##################################################################################
# Looping through time here.
for (i, dt) in enumerate(np.diff(time)):

    t = time[i+1]
    print("Time: ", t)
    iter = 0
    err = 1e11

    while err > tol:
        iter += 1
        # Solve for new displacement  
        solve(a_form == L_form, new, bc_U, solver_parameters={'linear_solver': 'mumps'})
        (Anew, Unew) = new.split(True)
        
        # Calculate error
        err_a = errornorm(Anew,Aold,norm_type = 'l2',mesh = None)
        err_u = errornorm(Unew,Uold,norm_type = 'l2',mesh = None)
        err = max(err_a, err_u)
        print(err_a, err_u)

        # Update new fields in same timestep with new calculated quantities
        Aold.assign(Anew)
        Uold.assign(Unew)
        print ('Iterations:', iter, ', Total time', t, ', error', err)
	
        if err < tol:
            # Update old fields from previous timestep with new quantities
            Aold.assign(Anew)
            Uold.assign(Unew)
            print ('err<tol :D','Iterations:', iter, ', Total time', t, ', error', err)

            if round(t*1e1) % 2 == 0: # each saved data point is 2e-9s
                store_a << Aold
                store_u << Uold #mm
                
                stress_elas.assign(project(sigma(Uold, Aold), sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
                store_stress_elas << stress_elas

                strain_total.assign( project(epsilon(Uold), strain, solver_type="cg", preconditioner_type="amg"))                     
                store_strain_total << strain_total

                normSquared.assign( project( dot(Uold, Uold), VV, solver_type="cg", preconditioner_type="amg"))
                store_normSquared << normSquared

                File('mydata/saved_mesh.xml') << mesh_half
                File('mydata/saved_a.xml') << Aold
                File('mydata/saved_u.xml') << Uold
                print ('Iterations:', iter, ', Total time', t, 'Saving datapoint')
 	    
print ('Simulation completed') 
##################################################################################
