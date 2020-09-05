begin
	using LightGraphs # create network topologies
	using LinearAlgebra
	using DifferentialEquations
	using GraphPlot
	using Plots
	using Parameters
	using LSODA
end

begin
	N = 4
	num_days = 10
	l_day = 3600*24
end

begin
	graph = random_regular_graph(iseven(3N) ? N : (N-1), 3)
end

@with_kw mutable struct parameterss
	N
	K
	R
	L_inv
	C_inv
	v_ref
	incidence
	n_prod
	n_cons

end

################# parameters initialization ###################################
param = parameterss(N = 4, K = 1. , R = 0.0532, L_inv = 1/0.2, C_inv = 1/0.01,  v_ref = 48. , incidence = incidence_matrix(graph,oriented=true), n_prod = 2, n_cons = 2)
###############################################################################



begin
	current_filter = 1:Int(1.5N)
	voltage_filter = Int(1.5N)+1:Int(2.5N)
	energy_filter = Int(2.5N)+1:Int(3.5N)#3N+1:4N
	energy_abs_filter = Int(3.5N)+1:Int(4.5N) #4N+1:5N
end


function DCToymodel!(du, u, p, t)
   #DC Microgrids swarm type network implemented by Lia Strenge in python, equations 4.12
   #print("du",du)
   P = -12.
   n_lines = Int(1.5*p.N)
   #print(u)

   i = u[1:n_lines]
   v = u[(n_lines+1):end]
   di = @view du[1:n_lines]
   dv = @view du[(n_lines+1):end]
   #i = view(u, 1:Int(1.5*p.N)) # u = n_lines
   #v = view(u, (Int(1.5*p.N)+1):Int(2.5*p.N))
   #di = view(du, 1:Int(1.5*p.N))
   #dv = view(du, (Int(1.5*p.N)+1):Int(2.5*p.N))
   inc_v = p.incidence' * v
   inc_i = p.incidence * i

   @. di .= p.L_inv.*(inc_v .-(p.R.*i))
   @. dv .= -1. .* inc_i
   @. dv[1:p.n_prod] += p.K .* (p.v_ref.- v[1:p.n_prod])
   @. dv[p.n_prod+1:end] += P ./ (v[p.n_prod+1:end].+1)
   @. dv .*= p.C_inv

   return nothing
end


##### solving ###############################

begin
	fp = [0. 0. 0. 0. 0. 0. 48. 48. 48. 48.] #initial condition
	tspan = (0. , num_days * l_day)
	ode = ODEProblem(DCToymodel!, fp, tspan, param)
end

sol = solve(ode, lsoda())

plot(sol, vars = current_filter)
plot(sol, vars = voltage_filter)

end
