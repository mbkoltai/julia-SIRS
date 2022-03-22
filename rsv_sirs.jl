# vectorised ODE system for age structured model
using LinearAlgebra, Statistics, Random, SparseArrays, OrdinaryDiffEq, Parameters, Plots
using BenchmarkTools, DelimitedFiles, DataFrames CSV

cd("Desktop/research/models/RSV_model/transmission_model/2022_new_project/julia_SIRS/")

# birthrates=vcat(2314,zeros(98))
# deathrates=vcat(ones(18)*0.00003,ones(45)*0.000001,zeros(18),ones(9)*0.000001,ones(9)*0.000179)
# K_m = readdlm("../../RSV-resurgence-model/extras/K_m.csv", ',', Float64)
# contmatr_rowvector = readdlm("../../RSV-resurgence-model/extras/contmatr_rowvector.csv",',',Float64)
# infvars_inds = reshape([4,5,6,13,14,15,22,23,24,31,32,33,40,41,42,49,50,51,58,59,60,67,68,69,76,77,78,85,86,87,94,95,96],3,11)
# suscvars_inds=[1 2 3 10 11 12 19 20 21 28 29 30 37 38 39 46 47 48 55 56 57 64 65 66 73 74 75 82 83 84 91 92 93]
# deltasusc=readdlm("../../RSV-resurgence-model/extras/deltasusc.csv", ',', Float64)
# prot_inf_ind = 7
# prot_adults_childb=[79 80 81]
# susc_adults_childb=[73 74 75]

#################################################################

# ODE system
function vector_ode_sirs!(du,u,p,t)
    birthrates=copy(p[1]); deathrates=p[2]; K_m=p[3]; contmatr_rowvector=p[4]; infvars_inds=p[5]; suscvars_inds=p[6];
    deltasusc=p[7]; prot_inf_ind=p[8]; prot_adults_childb=p[9]; susc_adults_childb=p[10]; lockdown_yr=p[11]; 
    p_t_input=p[12]
    # proportion of adults susceptible
    proport_adult_susc=sum(u[susc_adults_childb])/(sum(u[susc_adults_childb])+sum(u[prot_adults_childb]))
    birthrates[prot_inf_ind]=(1-proport_adult_susc)*birthrates[1]
    birthrates[1]=proport_adult_susc*birthrates[1]
    # infection terms
    inf_vars_stacked = hcat(u[infvars_inds[:,1],:],u[infvars_inds[:,2],:],u[infvars_inds[:,3],:],
                        u[infvars_inds[:,4],:],u[infvars_inds[:,5],:],u[infvars_inds[:,6],:],
                        u[infvars_inds[:,7],:],u[infvars_inds[:,8],:],u[infvars_inds[:,9],:],
                        u[infvars_inds[:,10],:],u[infvars_inds[:,11],:]);
    # "lockdown"
    scale_seas_force=1
    if t > (lockdown_yr*365+80) && t<((lockdown_yr+1)*365+80)
       scale_seas_force=0.1
    end
    seas_force_t=p_t_input(t)*scale_seas_force; # (1+cos(pi*((t-330)/365))^2) # print(round(seas_force_t,digits=2),", ")
    # force of infection
    lambda_vect=seas_force_t*Diagonal(reshape(deltasusc,33)) * contmatr_rowvector * ([1 1 1] * inf_vars_stacked)';
    infection_vect=Diagonal(vec(u[suscvars_inds])) * lambda_vect;
    F_vect = zeros(99); 
    F_vect[hcat(suscvars_inds,reshape(infvars_inds,1,33))']=vcat(-infection_vect,infection_vect);
    du_test = vcat(birthrates + F_vect + K_m * u[1:99] - deathrates .* u[1:99],infection_vect);
    for k_u=1:length(u)
        du[k_u]=du_test[k_u]
    end
end

##### 
# parameters
# seasonal forcing
# p_t = t-> 1 + 3*cos(pi*((t - 330)/365))^2
seas_st_dev =  7*5; t_peak = 330
p_t_normal = t -> 1 + exp(-0.5*(min((t % 365) - t_peak,(t % 365) + 365 - t_peak)/seas_st_dev)^2)
ode_inputs = [vcat(2314,zeros(98)), # 1) birth rates
    vcat(ones(18)*0.00003,ones(45)*0.000001,zeros(18),ones(9)*0.000001,ones(9)*0.000179), #2) deathrates
    readdlm("K_m.csv", ',', Float64), # 3) K_m
    readdlm("contmatr_rowvector.csv",',',Float64), # 4) contmatr_rowvector
    reshape([4,5,6,13,14,15,22,23,24,31,32,33,40,41,42,49,50,51,58,59,
                60,67,68,69,76,77,78,85,86,87,94,95,96],3,11), # 5) infvars_inds
    [1 2 3 10 11 12 19 20 21 28 29 30 37 38 39 46 47 48 55 56 57 64 65 66 73 74 75 82 83 84 91 92 93], # 6) suscvars_inds
    readdlm("deltasusc.csv", ',', Float64), # 7) deltasusc
                7, # 8) prot_inf_ind
                [79 80 81], # 9) prot_adults_childb
                [73 74 75], # 10) susc_adults_childb
                3, # ]; # 11) lockdown year
                p_t_normal]; # 12) seasonal forcing

################## ################## ##################
# WITHOUT IMPORTATIONS
# initial values
u0=readdlm("initvals_sirs_model_stationary.csv", ',', Float64)
# use last state of simul:
# reshape([sol_vector_ode_prob[1:99,1,size(sol_vector_ode_prob)[3]]; zeros(33)],132,1);
# writedlm( "FileName.csv",  A, ',')
# 
tspan = (0.0,3650.0); 
vector_ode_prob = ODEProblem(vector_ode_sirs!,u0,tspan,ode_inputs);
@time begin; sol_vector_ode_prob = solve(vector_ode_prob, Tsit5(),saveat=1); end
##################
# plot prevalence of infections
plot(sol_vector_ode_prob[reshape(ode_inputs[5][:,1:4],12),1,:]')

# susceptibles
# plot(sol_vector_ode_prob[(1:3).+9,1,:]')
# S + I + R for given age group
plot((sol_vector_ode_prob[(1:3).+9,1,:] + sol_vector_ode_prob[(4:6).+9,1,:] + 
      sol_vector_ode_prob[(7:9).+9,1,:])')
# their sum
k_age_sel=10; k_shift=(k_age_sel-1)*9
plot(sum(sol_vector_ode_prob[(1:9).+k_shift,1,:],dims=1)')
# plot multiple age groups together
plot(vcat(sum(sol_vector_ode_prob[(1:9),1,:],dims=1),
          sum(sol_vector_ode_prob[(1:9).+k_shift,1,:],dims=1),
          sum(sol_vector_ode_prob[(1:9).+k_shift.+9,1,:],dims=1))')

################## ################## ##################
# callback to introduce monthly importations (otherwise weird dynamics)
k_yr=10; tspan=(0.0,k_yr*365.0)
# timepoints for importations
dosetimes=collect((1:(k_yr*12))*30); condition(u,t,integrator) = t ∈ dosetimes;
affect!(integrator) = integrator.u[[reshape(ode_inputs[5],33); collect(100:132)]] .+= 10
cb = DiscreteCallback(condition,affect!);

# call with import events
vector_ode_prob = ODEProblem(vector_ode_sirs!,u0,tspan,ode_inputs);
@time begin; sol_event=solve(vector_ode_prob,Tsit5(),callback=cb,tstops=dosetimes,saveat=1); end

# plot prevalence of infections
k_col=5; plot(sol_event[reshape(ode_inputs[5][:,1:k_col],k_col*3),1,:]',size=(1200,600))
# prevalence susceptibles
# plot(sol_event[(1:3).+9,1,:]')

# incident infections
cumul_inf = sol_event[100:132,1,:];
incid_inf=cumul_inf[:,2:(size(sol_event)[3])] - cumul_inf[:,1:(size(sol_event)[3]-1)];
incid_inf[incid_inf.==0] .= NaN;
# plot nth infections for all age groups
k_inf_sel=2; plot(incid_inf[collect((0:10)*3) .+ k_inf_sel,:]')
# savefig("plot_2nd_inf_line.svg")

# plot(1:3771,incid_inf[20,:],seriestype=:scatter,markersize=2)
k_t=k_yr*365
plot(1:k_t,incid_inf[(1:3).+3,1:k_t]',size=(1200,600),
        marker=(:circle,1.5),linestyle=:dash,markerstrokewidth=0)
#
# savefig("plot_inf1_3_scatter_line.svg")

################## ################## ##################
# benchmark
@benchmark sol_event = solve(vector_ode_prob,Tsit5(),callback=cb,tstops=dosetimes)
# profile
@profview solve(vector_ode_prob,Tsit5(),callback=cb,tstops=dosetimes)