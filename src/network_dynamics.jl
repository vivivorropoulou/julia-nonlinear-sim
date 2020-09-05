@doc """
This is a module that contains the system and control dynamics.
# Examples
```julia-repl
julia> include("src/network_dynamics.jl")
Main.network_dynamics
```
"""
module network_dynamics

begin
	using Random # random numbers
	using LightGraphs # create network topologies
	using LinearAlgebra
	using DifferentialEquations: reinit!
	using DSP
	using ToeplitzMatrices
end

@doc """
    ACtoymodel!(du, u, p, t)
Lower-layer dynamics with controller from [Dörfler et al. 2017] eqns. 15a,b,c.
with kp = D, kI = K and chi = -p
[Dörfler et al. 2017]: https://arxiv.org/pdf/1711.07332.pdf
"""

function DCtoymodel!(du, u, p, t)
	#VARIABLEN
 	K = 1.
	R = 0.0532 .* ones(Int(1.5*p.N))
	v_ref = 48.

	i= view(u, 1:Int(1.5*p.N)) # u = n_lines
	v = view(u, (Int(1.5*p.N)+1):Int(2.5*p.N))

	di = view(du, 1:Int(1.5*p.N))
	dv = view(du, (Int(1.5*p.N)+1):Int(2.5*p.N))

	control_power_integrator = view(du,Int(2.5*p.N+1):Int(3.5*p.N))
	control_power_integrator_abs = view(du,Int(3.5*p.N+1):Int(4.5*p.N))
	#i = (K * (v_ref .- v)).*i
	# demand = - p.periodic_demand(t) .- p.residual_demand(t)
	power_ILC = p.hl.current_background_power #(t)
	#power_LI = v .* i_gen
	#K .* (v_ref - v) #K = droop coefficient #low-layer controller
	 #uncontrolled net power demand at node p
	periodic_power = - p.periodic_demand(t) .+ p.periodic_infeed(t) #determine the update cycle of the hlc
	fluctuating_power = - p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	Pd = periodic_power + fluctuating_power

	di= p.ll.L_inv .*(-R.*i .-(p.incidence' * v))

	dv = p.ll.C_inv.*( K.* (v_ref .- v) .+ (p.incidence * i) .- Pd./(v.+1))

	control_power_integrator =  v.* 1 .* (v_ref .- v)
	control_power_integrator_abs = abs.(v.* 1 .* (v_ref .- v))

	  #Compared with python model mg_new.py, line 141
	return nothing
end


function DCtoymodelstrenge!(du, u, p, t)
	#DC Microgrids swarm type network implemented by Lia Strenge in python, equations 4.12

	n_prod = 2
	n_cons = p.N - n_prod
	P = -12.
	n_lines = Int(1.5*p.N)

	i = u[1:n_lines]
    v = u[(n_lines+1):Int(2.5*p.N)]
	print(v)

    di = @view du[1:n_lines]
    dv = @view du[(n_lines+1):Int(2.5*p.N)]


	inc_v = p.incidence' * v
    inc_i = p.incidence * i

    @. di .= p.ll.L_inv.*(inc_v .-(p.ll.R.*i))
    @. dv .= -1. .* inc_i
    @. dv[1:n_prod] += p.ll.K .* (p.ll.v_ref.- v[1:n_prod])
    @. dv[n_prod+1:end] += P ./ (v[n_prod+1:end].+1)
    @. dv .*= p.ll.C_inv
end

function ACtoymodel!(du, u, p, t)
	theta = view(u, 1:p.N)
	omega = view(u, (p.N+1):(2*p.N))
	chi = view(u, (2*p.N+1):(3*p.N))


	dtheta = view(du, 1:p.N)
	domega = view(du, (p.N+1):(2*p.N))
	dchi = view(du, (2*p.N+1):(3*p.N))

	control_power_integrator = view(du,(3*p.N+1):(4*p.N))
	control_power_integrator_abs = view(du,(4*p.N+1):(5*p.N))

	# demand = - p.periodic_demand(t) .- p.residual_demand(t)
	power_ILC = p.hl.current_background_power #(t)
	power_LI =  chi .- p.ll.kP .* omega
	periodic_power = - p.periodic_demand(t) .+ p.periodic_infeed(t)
	fluctuating_power = - p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	# avoid *, use mul! instead with pre-allocated cache http://docs.juliadiffeq.org/latest/basics/faq.html
	#cache1 = zeros(size(p.coupling)[1])
	#cache2 = similar(cache1)
	#flows = similar(theta)
	#mul!(cache1, p.incidence', theta)
	#mul!(cache2, p.coupling, sin.(cache1))
	#mul!(flows, - p.incidence, cache2 )
	flows = - (p.incidence * p.coupling * sin.(p.incidence' * theta))
 	#println("periodic power: ", periodic_power)

	@. dtheta = omega
	@. domega = p.ll.M_inv .* (power_ILC .+ power_LI
						.+ periodic_power .+ fluctuating_power .+ flows)
						# signs checked (Ruth)
    @. dchi = p.ll.T_inv .* (- omega .- p.ll.kI .* chi) # Integrate the control power used.
	@. control_power_integrator = power_LI
	@. control_power_integrator_abs = abs.(power_LI)
	return nothing
end

@doc """
    ACtoymodel!(du, u, p, t)
Lower-layer dynamics with controller from [Dörfler et al. 2017] eqns. 15a,b,c.
with kp = D, kI = K and chi = -p
[Dörfler et al. 2017]: https://arxiv.org/pdf/1711.07332.pdf
with DC approx (sin removed)
"""

function ACtoymodel_lin!(du, u, p, t)
	theta = view(u, 1:p.N)
	omega = view(u, (p.N+1):(2*p.N))
	chi = view(u, (2*p.N+1):(3*p.N))
	dtheta = view(du, 1:p.N)
	domega = view(du, (p.N+1):(2*p.N))
	dchi = view(du, (2*p.N+1):(3*p.N))

	control_power_integrator = view(du,(3*p.N+1):(4*p.N))
	control_power_integrator_abs = view(du,(4*p.N+1):(5*p.N))

	# demand = - p.periodic_demand(t) .- p.residual_demand(t)
	power_ILC = p.hl.current_background_power
	power_LI =  chi .- p.ll.kP .* omega
	periodic_power = - p.periodic_demand(t) .+ p.periodic_infeed(t)
	fluctuating_power = - p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	# avoid *, use mul! instead with pre-allocated cache http://docs.juliadiffeq.org/latest/basics/faq.html
	#cache1 = zeros(size(p.coupling)[1])
	#cache2 = similar(cache1)
	#flows = similar(theta)
	#mul!(cache1, p.incidence', theta)
	#mul!(cache2, p.coupling, sin.(cache1))
	#mul!(flows, - p.incidence, cache2 )
	flows = - (p.incidence * p.coupling * p.incidence' * theta)


	@. dtheta = omega
	@. domega = p.ll.M_inv .* (power_ILC .+ power_LI
						.+ periodic_power .+ fluctuating_power .+ flows)
						# signs checked (Ruth)
    @. dchi = p.ll.T_inv .* (- omega .- p.ll.kI .* chi) # Integrate the control power used.
	@. control_power_integrator = power_LI
	@. control_power_integrator_abs = abs.(power_LI)
	return nothing
end


@doc """
    HourlyUpdate()
Store the integrated control power in memory.
See also [`(hu::HourlyUpdate)`](@ref).
"""
struct HourlyUpdate
	integrated_control_power_history
	HourlyUpdate() = new([])
end



@doc """
    HourlyUpdate(integrator)
PeriodicCallback function acting on the `integrator` that is called every simulation hour (t = 1,2,3...).
"""
function (hu::HourlyUpdate)(integrator)
	hour = mod(round(Int, integrator.t/3600.), 24) + 1
	last_hour = mod(hour-2, 24) + 1

	power_idx = 3*integrator.p.N+1:4*integrator.p.N
	power_abs_idx = 4*integrator.p.N+1:5*integrator.p.N
	# For the array  of arrays append to work correctly we need to give append!
	# an array of arrays. Otherwise obscure errors follow. Therefore u[3...] is
	# wrapped in [].

	# append!(hu.integrated_control_power_history, [integrator.u[power_idx]])

	# println("===========================")
	# println("Starting hour $hour, last hour was $last_hour")
	# println("Integrated power from the last hour:")
	# println(integrator.u[power_idx])
	# println("Yesterdays mismatch for the last hour:")
	# println(integrator.p.hl.mismatch_yesterday[last_hour,:])
	# println("Background power for the next hour:")
	# println(integrator.p.hl.daily_background_power[hour, :])

	integrator.p.hl.mismatch_yesterday[last_hour,:] .= integrator.u[power_idx]
	integrator.u[power_idx] .= 0.
	integrator.u[power_abs_idx] .= 0.

	# println("hour $hour")
	integrator.p.hl.current_background_power .= integrator.p.hl.daily_background_power[hour, :]
	# integrator.p.residual_demand = 0.1 * (0.5 + rand())
	# reinit!(integrator, integrator.u, t0=integrator.t, erase_sol=true)

	#now = copy(integrator.t)
	#state = copy(integrator.u)
	#reinit!(integrator, state, t0=now, erase_sol=true)
	nothing
end



function DailyUpdate_X(integrator)
	#println("mismatch ", integrator.p.hl.daily_background_power)
	#println("Q ", integrator.p.hl.Q)
	integrator.p.hl.daily_background_power = integrator.p.hl.Q * (integrator.p.hl.daily_background_power + integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday)
	#println("mismatch ", integrator.p.hl.daily_background_power)
	nothing
end

end
