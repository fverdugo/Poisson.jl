module Poisson

using TimerOutputs

using Gridap
import Gridap: ∇

export poisson

# Define manufactured functions
ufun(x) = x[1] + x[2]
ufun_grad(x) = VectorValue(1.0,1.0,0.0)
∇(::typeof(ufun)) = ufun_grad
bfun(x) = 0.0

# Define forms of the problem
a(v,u) = inner(∇(v), ∇(u))

# Define norms to measure the error
l2(u) = inner(u,u)
h1(u) = a(u,u) + l2(u)

"""
n: number of elements per direction
"""
function poisson(n::Integer)

  reset_timer!()

  # Construct the discrete model
  @timeit "model" model = CartesianDiscreteModel(partition=(n,n,n))
  
  # Construct the FEspace
  order = 1
  diritag = "boundary"
  @timeit "fespace" fespace = ConformingFESpace(Float64,model,order,diritag)
  
  # Define test and trial spaces
  @timeit "V" V = TestFESpace(fespace)

  @timeit "U" U = TrialFESpace(fespace,ufun)
  
  # Define integration mesh and quadrature
  @timeit "trian" trian = Triangulation(model)

  @timeit "quad" quad = CellQuadrature(trian,order=2)
  
  # Define the source term
  @timeit "bfield" bfield = CellField(trian,bfun)
  
  # Define Assembler
  @timeit "assem" assem = SparseMatrixAssembler(V,U)
  
  # Define the FEOperator
  b(v) = inner(v,bfield)
  @timeit "op" op = LinearFEOperator(a,b,V,U,assem,trian,quad)
  
  # Define the FESolver
  ls = LUSolver()
  solver = LinearFESolver(ls)
  
  # Solve!
  @timeit "uh" uh = solve(solver,op)
  
  # Define exact solution and error
  @timeit "u" u = CellField(trian,ufun)

  @timeit "e" e = u - uh
  
  # Compute errors
  @timeit "el2" el2 = sqrt(sum( integrate(l2(e),trian,quad) ))

  @timeit "eh1" eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))
  
  @assert el2 < 1.e-8
  @assert eh1 < 1.e-8

  print_timer()

end

end # module
