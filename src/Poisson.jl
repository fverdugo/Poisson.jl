module Poisson

using TimerOutputs
using IterativeSolvers: cg
using Preconditioners: AMGPreconditioner, SmoothedAggregation

using Gridap
import Gridap: ∇

export poisson

# Define manufactured functions
u(x) = x[1] + x[2]
∇u(x) = VectorValue(1.0,1.0,0.0)
∇(::typeof(u)) = ∇u
f(x) = 0.0

# Define forms of the problem
a(v,u) = inner(∇(v), ∇(u))
b(v) = inner(v,f)

# Define norms to measure the error
l2(u) = inner(u,u)
h1(u) = a(u,u) + l2(u)

"""
n: number of elements per direction
"""
function poisson(n::Integer)

  reset_timer!()

  @timeit "model" model = CartesianDiscreteModel(partition=(n,n,n))
  
  order = 1
  diritag = "boundary"
  @timeit "fespace" fespace = CLagrangianFESpace(Float64,model,order,diritag)
  
  @timeit "V" V = TestFESpace(fespace)

  @timeit "U" U = TrialFESpace(fespace,u)
  
  @timeit "trian" trian = Triangulation(model)

  @timeit "quad" quad = CellQuadrature(trian,order=2)
  
  @timeit "assem" assem = SparseMatrixAssembler(V,U)
  
  @timeit "op" op = LinearFEOperator(a,b,V,U,assem,trian,quad)
  
  @timeit "AMG setup" p = AMGPreconditioner{SmoothedAggregation}(op.mat)

  @timeit "PCG"  x = cg(op.mat,op.vec,verbose=true,Pl=p)

  @timeit "uh" uh = FEFunction(U,x)
  
  @timeit "e" e = u - uh
  
  @timeit "el2" el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @timeit "eh1" eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))
  
  @show el2
  @show eh1

  print_timer()

end

end # module
