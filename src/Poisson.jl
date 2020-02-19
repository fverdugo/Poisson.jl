module Poisson

using TimerOutputs
using IterativeSolvers: cg
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using SparseArrays

using Gridap
import Gridap: ∇

export poisson

# Define manufactured functions
u(x) = x[1] + x[2]
∇u(x) = VectorValue(1.0,1.0,0.0)
∇(::typeof(u)) = ∇u
const f = 0.0

# Define forms of the problem
a(v,u) = ∇(v)*∇(u)
l(v) = v*f

# Define norms to measure the error
l2(u) = inner(u,u)
h1(u) = a(u,u) + l2(u)

"""
n: number of elements per direction
"""
function poisson(n::Integer)

  reset_timer!()

  domain = (0,1,0,1,0,1)
  partition = (n,n,n)
  @timeit "model" model = CartesianDiscreteModel(domain,partition)
  
  order = 1
  @timeit "V" V = TestFESpace(
    model=model,dirichlet_tags="boundary", conformity=:H1,
    reffe=:Lagrangian, order=order, valuetype=Float64)
  
  @timeit "U" U = TrialFESpace(V,u)
  
  @timeit "trian" trian = Triangulation(model)

  degree = 2
  @timeit "quad" quad = CellQuadrature(trian,degree)

  @timeit "t_Ω" t_Ω = AffineFETerm(a,l,trian,quad)
  
  @timeit "op" op = AffineFEOperator(SparseMatrixCSC{Float64,Int32},V,U,t_Ω)

  A = get_matrix(op)
  b = get_vector(op)
  
  @timeit "AMG setup" p = AMGPreconditioner{SmoothedAggregation}(A)

  @timeit "PCG"  x = cg(A,b,verbose=true,Pl=p)

  @timeit "uh" uh = FEFunction(U,x)
  
  @timeit "e" e = u - uh
  
  @timeit "el2" el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @timeit "eh1" eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))
  
  @show el2
  @show eh1

  print_timer()

end

end # module
