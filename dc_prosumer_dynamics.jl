begin
	using JLD2, FileIO, GraphIO, CSV, DataFrames
	using ForwardDiff
	using Distributed
	using LightGraphs # create network topologies
	using LinearAlgebra
	using DifferentialEquations
	using GraphPlot
	using Random
	using Plots
	using Parameters
	using LSODA
	using ToeplitzMatrices
	using DSP
	using LaTeXStrings
	using Distributions
	using StatsBase
	using Roots
	using Interpolations
	Random.seed!(42)
end

begin
	dir = @__DIR__
	N = 4 #Number of nodes
	num_prod = 2 # producer nodes
	nom_cons = N - num_prod
	N_half = Int(N/2)
	num_days = 7
	l_day = 3600*24 # DemCurve.l_day
	l_hour = 3600 # DemCurve.l_hour
	l_minute = 60

	kappa = 1.0 / l_hour
	vc1 = 1:N # ilc_nodes (here: without communication)
	cover1 = Dict([v => [] for v in vc1])# ilc_cover
	u = [zeros(1000,1);1;zeros(1000,1)];
	fc = 1/6;
	a = digitalfilter(Lowpass(fc),Butterworth(2));
	Q1 = filtfilt(a,u);# Markov Parameter
	Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);

end


struct demand_amp_var
	demand
end


function (dav::demand_amp_var)(t)
	index = Int(floor(t / (24*3600)))
	dav.demand[index + 1,:]
end



begin
	graph = random_regular_graph(iseven(3N) ? N : (N-1), 3)
end


@with_kw mutable struct LeakyIntegratorPars
	K
	R
	L_inv
	C_inv
	v_ref
	n_prod
	n_cons
end

@with_kw mutable struct ILCPars
	kappa
	mismatch_yesterday
	daily_background_power
	current_background_power
	ilc_nodes
	ilc_covers
	Q
end
 @with_kw mutable struct incidences
	 inc_i
	 inc_v
end

@with_kw mutable struct UlMoparss
	N::Int
	ll::LeakyIntegratorPars
	hl::ILCPars
	inc::incidences
	periodic_infeed
	periodic_demand
	fluctuating_infeed
	residual_demand
	incidence

	function UlMoparss(N::Int,
						ll::LeakyIntegratorPars,
						hl:: ILCPars,
						inc::incidences,
						periodic_infeed,
						periodic_demand,
						fluctuating_infeed,
						residual_demand)
			new(N, ll,
			hl,
			inc,
			periodic_infeed,
			periodic_demand,
			fluctuating_infeed,
			residual_demand,
			incidence_matrix(graph,oriented=true))
	end
end

function set_parameters(N)
	low_layer_control = LeakyIntegratorPars(K = 1. , R = 0.0532, L_inv = 1/0.237e-4, C_inv = 1/0.01,  v_ref = 48. ,  n_prod = num_prod, n_cons = N.-num_prod)
	u = [zeros(1000,1);1;zeros(1000,1)];
	fc = 1/6;
	a = digitalfilter(Lowpass(fc), Butterworth(2));
	Q1 = filtfilt(a,u);
	control_incidences = incidences(inc_i = zeros(N), inc_v = zeros(Int(1.5*N)))
	higher_layer_control = ILCPars(kappa = 0.35, mismatch_yesterday=zeros(24,N), daily_background_power=zeros(24,N), current_background_power=zeros(N), ilc_nodes=1:N, ilc_covers = [], Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]))
	periodic_infeed = t -> zeros(N)
	peak_demand = rand(N)
	periodic_demand = t -> zeros(N)
	fluctuating_infeed = t -> zeros(N)
	residual_demand = t -> zeros(N)

	return UlMoparss(N,low_layer_control,
					higher_layer_control,
					control_incidences,
					periodic_infeed,
					periodic_demand,
					fluctuating_infeed,
					residual_demand)
end

begin
	current_filter = 1:Int(1.5N)
	voltage_filter = Int(1.5N)+1:Int(2.5N)
	energy_filter = Int(2.5N)+1:Int(3.5N)#3N+1:4N
end

function prosumerToymodel!(du, u, p, t)

	n_lines = Int(1.5*p.N)

    #state variables
    i = u[1:n_lines]
    v = u[(n_lines+1):Int(2.5*p.N)]

    di = @view du[1:n_lines]
    dv = @view du[(n_lines+1):Int(2.5*p.N)]
    control_power_integrator = @view du[Int(2.5*p.N)+1:Int(3.5*p.N)]

	periodic_power = - p.periodic_demand(t) .+ p.periodic_infeed(t) #determine the update cycle of the hlc
	fluctuating_power = -  p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	Pd = periodic_power + fluctuating_power

    i_ILC = p.hl.current_background_power./v

	p.inc.inc_v = p.incidence' * v
    p.inc.inc_i = p.incidence * i

	i_gen = p.ll.K .* (p.ll.v_ref .- v)
	i_load = Pd./(v.+1)

	@. di = p.ll.L_inv .*(-(p.ll.R.*i) .+ p.inc.inc_v)
	@. dv = p.ll.C_inv.*(i_gen .- p.inc.inc_i .+i_ILC .+ i_load)

	@. control_power_integrator= p.inc.inc_i.* v 				#power LI


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
	power_idx = Int(2.5*integrator.p.N)+1:Int(3.5*integrator.p.N) # power index

	#Define current background power in ac power ILC
	integrator.p.hl.current_background_power .= integrator.p.hl.daily_background_power[hour, :]

	#power calculation y^c
	integrator.u[power_idx].= integrator.p.incidence *  integrator.u[current_filter].*  integrator.u[voltage_filter]# Node current is multiplied here with voltage
	integrator.p.hl.mismatch_yesterday[last_hour,:] .= integrator.u[power_idx]


	integrator.u[power_idx] .= 0.

	nothing
end



function DailyUpdate_X(integrator)
#ilc
	integrator.p.hl.daily_background_power = integrator.p.hl.Q * (integrator.p.hl.daily_background_power + (integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday)) # mismatch is horuly energy
	nothing
end

demand_amp1 = demand_amp_var(repeat([80 80 80 10 10 10 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp2 = demand_amp_var(repeat([10 10 10 80 80 80 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp3 = demand_amp_var(repeat([60 60 60 60 10 10 10 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp4 = demand_amp_var(repeat([30 30 30 30 10 10 10 80 80 80 80], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp = t->vcat(demand_amp1(t), demand_amp2(t), demand_amp3(t), demand_amp4(t))

periodic_demand =  t-> demand_amp(t) .* sin(t*pi/(24*3600))^2

samples = 24*4

inter = interpolate([.2 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples)



################### set parameters ############################
begin
	param = set_parameters(N)
	param.periodic_demand = periodic_demand
	param.residual_demand = residual_demand
	param.hl.daily_background_power .= 0
	param.hl.current_background_power .= 0
	param.hl.mismatch_yesterday .= 0.
end
####################### solving ###############################
begin
	fp = [0. 0. 0. 0. 0. 0. 48. 48. 48. 48. 0. 0. 0. 0.] #initial condition
	factor = 0.05
	ic = factor .* ones(14)
	tspan = (0. , num_days * l_day) # 7 Tage
	tspan2 = (0., 1.0)
	#tspan3 = (0., 200.)
	ode = ODEProblem(prosumerToymodel!, fp, tspan, param,
	callback=CallbackSet(PeriodicCallback(HourlyUpdate(), l_hour),
						 PeriodicCallback(DailyUpdate_X, l_day)))
end
sol = solve(ode, Rodas4())

######################## PLOTTING ########################################

#K_sol = sol.prob.p.ll.K
#v_ref_sol = sol.prob.p.ll.v_ref
#inc_i_sol = sol.prob.p.inc.inc_i
#print(inc_i_sol)
#energy_ll_1 = (K_sol.*(v_ref_sol.- sol[voltage_filter[1],:])) .* sol[voltage_filter[1],:]#
#plot(energy_ll_1)

###################### Hourly energy #####################################
hourly_energy = zeros(24*num_days+1,N)
for i=1:24*num_days+1
	for j = 1:N
		hourly_energy[i,j] = sol((i-1)*3600)[energy_filter[j]]
	end
end
p1 = plot()
plot!(hourly_energy[:,1]+ hourly_energy[:,2] + hourly_energy[:,3] + hourly_energy[:,4], label = "sum")
plot!(hourly_energy, title = "Lower-layer energy per node ", label = ["Node 1" "Node 2" "Node 3" "Node 4"]) #Sum of node powers should be zero
xlabel!("Time in h")
ylabel!("Energy in W")
savefig("$dir/plots/DC_prosumer_lower_layer_energy.png")
###########################################################################
######################ILC power ##########################################


ILC_power = zeros(num_days+2,24,N)
for j = 1:N
	ILC_power[2,:,j] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,j])
end
norm_energy_d = zeros(num_days,N)
for j = 1:N
	norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
end

for i=2:num_days
	for j = 1:N
		ILC_power[i+1,:,j] = Q*(ILC_power[i,:,j] +  kappa*hourly_energy[(i-1)*24+1:i*24,j])
		norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
	end
end

#ILC_power_agg = maximum(mean(ILC_power.^2,dims=3),dims=2)
ILC_power_agg = [norm(mean(ILC_power,dims=3)[d,:]) for d in 1:num_days+2]
ILC_power_hourly_mean = vcat(mean(ILC_power,dims=3)[:,:,1]'...)
ILC_power_hourly_mean_node1 = vcat(ILC_power[:,:,1]'...)
ILC_power_hourly = [norm(reshape(ILC_power,(num_days+2)*24,N)[h,:]) for h in 1:24*(num_days+2)]
ILC_power_hourly_node1 = [norm(reshape(ILC_power,(num_days+2)*24,N)[h,1]) for h in 1:24*(num_days+2)]
dd = t->((-periodic_demand(t) .- residual_demand(t))./100)
load_amp = [first(maximum(dd(t))) for t in 1:3600*24:3600*24*num_days]

norm_hourly_energy = [norm(hourly_energy[h,:]) for h in 1:24*num_days]

###########################################################################


node = 1
p1 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"))
savefig("$dir/plots/DC_prosumer_demand_seconds_node_$(node)_hetero.png")
#plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
#               xtickfontsize=14,
#    		   legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)




node = 2
p2 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,3]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"))
savefig("$dir/plots/DC_prosumer_demand_seconds_node_$(node)_hetero.png")


node = 3
p3 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,3]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"))
savefig("$dir/plots/DC_prosumer_demand_seconds_node_$(node)_hetero.png")


node = 4
p4 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,3]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"))
savefig("$dir/plots/DC_prosumer_demand_seconds_node_$(node)_hetero.png")



plot(1:num_days, ILC_power_agg[1:num_days,1,1] ./ maximum(ILC_power_agg), label=L"$\max_h \Vert P_{ILC, k}\Vert$")
plot!(1:num_days, mean(norm_energy_d,dims=2) ./ maximum(norm_energy_d), label=L"norm(y_h)")
plot!(1:num_days, load_amp  ./ maximum(load_amp), label = "demand amplitude")
xlabel!("day d [d]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/DC_prosumer_demand_daily_hetero.png")
###########################################################################


energy_cons = plot(sol, vars = energy_filter,title = "Energy per node ", label = ["Node 3" "Node 4"])
xlabel!("Time in s")
ylabel!("Energy in W")
savefig("$dir/plots/Energy_with_callbacks_producer_consumer.png")
energy_prod= plot(sol, vars = energy_filter[1:2],title = "Producer Energy per node ", label = ["Node 1" "Node 2"])
#sum up the node powers
energy_sum = sol[energy_filter[1],:]+sol[energy_filter[2],:]+sol[energy_filter[3],:]+sol[energy_filter[4],:]

plot(energy_sum ,title = "Sum of Energy ",label = "Energy sum")
xlabel!("Time in s")
ylabel!("Energy in W")
savefig("$dir/plots/Energy_sum_with_callbacks_producer_consumer.png")
############################################################################
plot(sol, vars = voltage_filter,title = "Voltage per node ", label = ["Node 1" "Node 2" "Node 3" "Node 4"])
xlabel!("Time in s")
ylabel!("Voltage in V")
savefig("$dir/plots/DC_prosumer_periodic_voltage_julia.png")


############ DC CURRENT plot #########################################

cur = plot(sol, vars = current_filter, title = "Current per edge ", label = ["Edge 1" "Edge 2" "Edge 3" "Edge 4" "Edge 5" "Edge 6"])
xlabel!("Time in s")
ylabel!("Current in A")
savefig("$dir/plots/DC_prosumer_periodic_current_julia.png")
plot!(cur,energy)
end
