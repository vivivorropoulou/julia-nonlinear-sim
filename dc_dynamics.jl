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

	num_days = 10
	l_day = 3600*24 # DemCurve.l_day
	l_hour = 3600 # DemCurve.l_hour
	l_minute = 60


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

@with_kw mutable struct UlMopars
	N::Int
	ll::LeakyIntegratorPars
	hl::ILCPars
	periodic_infeed
	periodic_demand
	fluctuating_infeed
	residual_demand
	incidence

	function UlMopars(N::Int,
						ll::LeakyIntegratorPars,
						hl:: ILCPars,
						periodic_infeed,
						periodic_demand,
						fluctuating_infeed,
						residual_demand)
			new(N, ll,
			hl,
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

	higher_layer_control = ILCPars(kappa = 0.35, mismatch_yesterday=zeros(24,num_prod), daily_background_power=zeros(24,num_prod), current_background_power=zeros(num_prod), ilc_nodes=1:N, ilc_covers = [], Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]))
	periodic_infeed = t -> zeros(num_prod)
	periodic_demand = t -> zeros(num_prod)
	fluctuating_infeed = t -> zeros(num_prod)
	residual_demand = t -> zeros(num_prod)

	return UlMopars(N,low_layer_control,
					higher_layer_control,
					periodic_infeed,
					periodic_demand,
					fluctuating_infeed,
					residual_demand)
end

begin
	current_filter = 1:Int(1.5N)
	voltage_filter = Int(1.5N)+1:Int(2.5N)
	energy_filter = Int(2.5N)+1:Int(3.5N)#3N+1:4N
	energy_abs_filter = Int(3.5N)+1:Int(4.5N) #4N+1:5N
end


function DCToymodel!(du, u, p, t)
   #P = -12
   n_lines = Int(1.5*p.N)

   #state variables
   i = u[1:n_lines]
   v = u[(n_lines+1):Int(2.5*p.N)]

   di = @view du[1:n_lines]
   dv = @view du[(n_lines+1):Int(2.5*p.N)]
   control_power_integrator = @view du[Int(2.5*p.N)+1:Int(3.5*p.N)]

   #ILC power is calculated in the callback functions, since the power is not a state of the DC microgrids system
   inc_v = p.incidence' * v
   inc_i = p.incidence * i

   #periodic demand instead of constant demand
   periodic_power = -p.periodic_demand(t) .+p.periodic_infeed(t)
   fluctuating_power = -p.residual_demand(t) .+ p.fluctuating_infeed(t)
   Pd = fluctuating_power + periodic_power

   #network topology
   @. di .= p.ll.L_inv.*(inc_v .-(p.ll.R.*i))
   @. dv .= -1. .* inc_i
   @. dv[1:p.ll.n_prod] += p.ll.K .* (p.ll.v_ref.- v[1:p.ll.n_prod]) #integrator.u[voltage_filter] power ilc hinzufügen
   @. dv[p.ll.n_prod+1:end] += Pd ./ (v[p.ll.n_prod+1:end].+1)
   @. dv .*= p.ll.C_inv

   #@. control_power_integrator .= v .* inc_i


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

	n_lines = Int(1.5*integrator.p.N)

	#indexes
	power_idx = Int(2.5*integrator.p.N)+1:Int(3.5*integrator.p.N) # power index
	producer_idx = 1:integrator.p.ll.n_prod # producer index
	consumer_idx = integrator.p.N .- producer_idx # consumer index


	#Define current background power
	integrator.p.hl.current_background_power[producer_idx] .= integrator.p.hl.daily_background_power[hour, :]

	integrator_inc_i = integrator.p.incidence *  integrator.u[1:n_lines] #edge currents per node summed up

	#sum up all node currents
	integrator_inc_i_sum = 0

	for j = 1:integrator.p.N
		integrator_inc_i_sum += integrator_inc_i[j]
		#print(integrator_inc_i_sum)
		#print("           ")
	end

	#power calculation
	integrator.u[power_idx] .= integrator_inc_i .*  integrator.u[voltage_filter] # Current is multiplied here with voltage
	integrator_power = integrator.u[power_idx]

	#ILC power in form of ILC current is calculated here
	integrator_i_ILC = integrator.p.hl.current_background_power[producer_idx]./integrator_power[producer_idx] #ILC power in form of current
	integrator_inc_i[producer_idx] .+= integrator_i_ILC #We get the ILC power from the producer nodes

	#update the power per node
	integrator.u[power_idx] .= integrator_inc_i .*  integrator.u[voltage_filter]


	integrator.p.hl.mismatch_yesterday[last_hour,:] .= integrator_power[producer_idx]



	nothing
end



function DailyUpdate_X(integrator)

	integrator.p.hl.daily_background_power = integrator.p.hl.Q * (integrator.p.hl.daily_background_power + integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday)
	nothing
end

demand_amp1 = demand_amp_var(repeat([80 80 80 10 10 10 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp2 = demand_amp_var(repeat([10 10 10 80 80 80 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp = t->vcat(demand_amp1(t), demand_amp2(t))

periodic_demand =  t-> demand_amp(t) .* sin(t*pi/(24*3600))^2

samples = 24*2

inter = interpolate([.2 * randn(2) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples)


################### set parameters ############################
param = set_parameters(N)
param.periodic_demand = periodic_demand
param.residual_demand = residual_demand
param.hl.daily_background_power .= 0
param.hl.current_background_power .= 0
param.hl.mismatch_yesterday .= 0.

####################### solving ###############################
begin
	fp = [0. 0. 0. 0. 0. 0. 48. 48. 48. 48. 0. 0. 0. 0.] #initial condition
	factor = 0.05
	ic = factor .* ones(14)
	tspan = (0. , num_days * l_day)
	tspan2 = (0., 1.0)
	#tspan3 = (0., 200.)
	ode = ODEProblem(DCToymodel!, fp, tspan, param,
	callback=CallbackSet(PeriodicCallback(HourlyUpdate(), l_hour),
						 PeriodicCallback(DailyUpdate_X, l_day)))
end
sol = solve(ode, Rodas4())

######################## Plotting ########################################
hourly_energy = zeros(24*num_days+1,N)
for i=1:24*num_days+1
	for j = 1:N
		hourly_energy[i,j] = sol((i-1)*3600)[energy_filter[j]]
	end
end
plot(hourly_energy)
plot(sol, vars = energy_filter,title = "Energy per node ", label = ["Node 1" "Node 2" "Node 3" "Node 4"])
xlabel!("Time in s")
ylabel!("Energy in W")
savefig("$dir/plots/Energy_with_callbacks_producer_consumer.png")

#sum up the node powers
energy_sum = sol[11,:]+sol[12,:]+sol[13,:]+sol[14,:]

plot(energy_sum ,title = "Sum of Energy ",label = "Energy sum")
xlabel!("Time in s")
ylabel!("Energy in W")
savefig("$dir/plots/Energy_sum_with_callbacks_producer_consumer.png")

end
