# vectorised ODE system for age structured model
using LinearAlgebra, Statistics, Random, SparseArrays, OrdinaryDiffEq, Parameters, Plots
using BenchmarkTools, DelimitedFiles, DataFrames

cd("Desktop/research/models/RSV_model/transmission_model/2022_new_project/julia_SIRS/")

#################################################################

# parameters
# seasonal forcing
seas_st_dev=7*5; t_peak=7*48; forcing_term=1
p_t_normal = t -> 1 + forcing_term*exp(-0.5*(min(abs((t % 365) - t_peak),(t % 365) + 365 - t_peak)/seas_st_dev)^2);
# importation of new infections
p_t_import = t -> (abs(t % 30 - 30)<1)*5
# ODE system
ode_inputs = [vcat(2314,zeros(98)), # 1) birth rates
    vcat(ones(18)*0.00003,ones(45)*0.000001,zeros(18),ones(9)*0.000001,ones(9)*0.000179), #2) deathrates
    readdlm("K_m.csv", ',', Float64),              # 3) K_m
    readdlm("contmatr_rowvector.csv",',',Float64), # 4) contmatr_rowvector
    reshape([4,5,6,13,14,15,22,23,24,31,32,33,40,41,42,49,50,51,58,59,
                60,67,68,69,76,77,78,85,86,87,94,95,96],3,11), # 5) infvars_inds
    [1 2 3 10 11 12 19 20 21 28 29 30 37 38 39 46 47 48 55 56 57 64 65 66 73 74 75 82 83 84 91 92 93], # 6) suscvars_inds
    readdlm("delta_susc.csv", ',', Float64), # 7) deltasusc
                7,           # 8) prot_inf_ind
                [79 80 81],  # 9) prot_adults_childb
                [73 74 75],  # 10) susc_adults_childb
                9,           # 11) lockdown year
                p_t_normal, # 12) seasonal forcing
                0];

# ODE system
function vector_ode_sirs!(du,u,p,t)
    birthrates=copy(p[1]); deathrates=p[2]; K_m=p[3]; contmatr_rowvector=p[4]; infvars_inds=p[5]; suscvars_inds=p[6];
    deltasusc=p[7]; prot_inf_ind=p[8]; prot_adults_childb=p[9]; susc_adults_childb=p[10]; lockdown_yr=p[11]; 
    p_t_input=p[12]; p_import=copy(p[13]);
    # proportion of adults susceptible 
    proport_adult_susc=sum(u[susc_adults_childb])/(sum(u[susc_adults_childb])+sum(u[prot_adults_childb]))
    # babies born with protection
    birthrates[prot_inf_ind]=(1-proport_adult_susc)*birthrates[1]
    # born without
    birthrates[1]=proport_adult_susc*birthrates[1]
    # infection terms
    inf_vars_stacked = hcat(u[infvars_inds[:,1],:],u[infvars_inds[:,2],:],u[infvars_inds[:,3],:],
                        u[infvars_inds[:,4],:],u[infvars_inds[:,5],:],u[infvars_inds[:,6],:],
                        u[infvars_inds[:,7],:],u[infvars_inds[:,8],:],u[infvars_inds[:,9],:],
                        u[infvars_inds[:,10],:],u[infvars_inds[:,11],:]);
    # "lockdown"
    scale_seas_force=1
    if t >= (lockdown_yr*365+86) && t <= ((lockdown_yr+1)*365+137)
       scale_seas_force=0.1
    end
    seas_force_t=p_t_input(t)*scale_seas_force; # (1+cos(pi*((t-330)/365))^2) # print(round(seas_force_t,digits=2),", ")
    # force of infection
    lambda_vect=seas_force_t*Diagonal(reshape(deltasusc,33)) * contmatr_rowvector * ([1 1 1] * inf_vars_stacked)';
    infection_vect=Diagonal(vec(u[suscvars_inds])) * lambda_vect;
    F_vect = zeros(99); 
    F_vect[hcat(suscvars_inds,reshape(infvars_inds,1,33))']=vcat(-infection_vect,infection_vect); # 
    # print(t,", ",p_import,"\n")
    du_test = vcat(birthrates + F_vect + K_m * u[1:99] - deathrates .* u[1:99],infection_vect);
    for k_u=1:length(u)
        du[k_u]=du_test[k_u]
    end
end

##### 

################## ################## ##################
# initial conditions
u0=readdlm("initvals_sirs_model_startlow.csv", ',', Float64); # initvals_sirs_model_stationary.csv
# use last state of simul:
# reshape([sol_vector_ode_prob[1:99,1,size(sol_vector_ode_prob)[3]]; zeros(33)],132,1);
# writedlm( "FileName.csv",  A, ',')
# 
tspan = (0.0,3650.0);
################## ################## ##################
# simulation duration
# k_yr=10; tspan=(0.0,k_yr*365.0)
tspan=(0.0,4749.0)
# callback to introduce monthly importations (otherwise weird dynamics)
# timepoints for importations and condition for callback
dosetimes=collect((1:(k_yr*12))*30); 
condition(u,t,integrator) = t ∈ dosetimes;
# otherwise
# condition2(u,t,integrator)= ~(t ∈ dosetimes)
# increment state variables
affect!(integrator) = integrator.u[reshape(ode_inputs[5],33)] .+= 5; #
# add a vector of new infections
# affect!(integrator) = integrator.p[13] = 5;
# affect2!(integrator) = integrator.p[13] = 0;
cb=DiscreteCallback(condition,affect!);
# cb2=DiscreteCallback(condition2,affect2!);

# create ODE object
vector_ode_prob = ODEProblem(vector_ode_sirs!,u0,tspan,ode_inputs);
# call with import events by CALLBACK
@time begin; sol_event=solve(vector_ode_prob,Tsit5(),callback=cb,tstops=dosetimes,saveat=1); end; # CallbackSet(
# without CALLBACK
# @time begin; sol_event=solve(vector_ode_prob,Tsit5(),saveat=1); end;

# plot prevalence of infections
sel_vars = [4,5,6]; # reshape(ode_inputs[5][:,1:k_col],k_col*3)
k_col=5; 
plot(sol_event[[4],1,:]',xticks=0:200:(10*365),yticks=0:5e3:100e3,size=(1200,600))
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
# parameter sampling
# p_new=copy(ode_inputs); 
@time begin; 
for k_parsampl in 1:100
p_new[11]=rand(3:8); print(k_parsampl,", ")
problem_update = remake(vector_ode_prob; p=p_new) # p_new being the new params being sampled
sol_event=solve(problem_update,Tsit5(),callback=cb,tstops=dosetimes,saveat=1); 
end
print("\n end \n")
end

################## ################## ##################
# benchmark
@benchmark sol_event = solve(vector_ode_prob,Tsit5(),callback=cb,tstops=dosetimes)
# profile
@profview solve(vector_ode_prob,Tsit5(),callback=cb,tstops=dosetimes)

# using StaticArrays
