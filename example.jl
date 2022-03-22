import Pkg; Pkg.add("OrdinaryDiffEq")
Pkg.add("Parameters"); Pkg.add("Plots")
using LinearAlgebra, Statistics, Random, SparseArrays, OrdinaryDiffEq, Parameters, Plots

function F_simple(x,p,t;γ =1/18,R₀=3.0,σ=1/5.2)
    s, e, i, r = x
    return [-γ*R₀*s*i; γ*R₀*s*i -  σ*e; σ*e - γ*i; γ*i;]
    # ds/dt = -γR₀si # de/dt = γR₀si -σe # di/dt = σe -γi # dr/dt = γi
end

i_0 = 1E-7   # 33 = 1E-7 * 330 million population = initially infected
e_0 = 4.0 * i_0   # 132 = 1E-7 *330 million = initially exposed
s_0 = 1.0 - i_0 - e_0
r_0 = 0.0
x_0 = [s_0, e_0, i_0, r_0]  # initial condition

tspan = (0.0, 350.0)  # ≈ 350 days
prob = ODEProblem(F_simple, x_0, tspan)


@time begin
sol = solve(prob, Tsit5())
end

plot(sol, labels = ["s" "e" "i" "r"], title = "SEIR Dynamics", lw = 2)

vector=[1,2,3]
# vector product
vector' * vector

# matrix equation
A  = [1. 0  0 -5;  4 -2  4 -3; -4  0  0  1; 5 -2  2  3]
u0 = rand(4,1); tspan = (0.0,10.0)
f_matr_ode(u,p,t) = A*u
prob_matr_eq = ODEProblem(f_matr_ode,u0,tspan)

@time begin
sol_matr_eq = solve(prob_matr_eq, Tsit5())
end

size(sol_matr_eq)

plot(sol_matr_eq)

#########################################
# matrix equation non-linear terms

#########################################
# Lorenz equation

function parameterized_lorenz!(du,u,p,t)
    # du[1] = p[1]*(u[2]-u[1])
    # du[2] = u[1]*(p[2]-u[3]) - u[2]
    # du[3] = u[1]*u[2] - p[3]*u[3]
    du_test = vcat(p[1]*(u[2]-u[1]),u[1]*(p[2]-u[3]) - u[2],u[1]*u[2] - p[3]*u[3])
    for j=1:length(u)
        du[j]=du_test[j]
    end
    print(round(du[3],digits=1),", ")
end

u0 = [1.0,0.0,0.0]; tspan = (0.0,10.0); p = [10.0,28.0,8/3]
# solve
lorenz_prob = ODEProblem(parameterized_lorenz!,u0,tspan,p);
sol_lorenz = solve(lorenz_prob, Tsit5());
plot(sol_lorenz)
# p = [1,2,3]
# outputs = Array{Float64}(undef, 0, 2)
@time begin
for i in 1:1e3
    p=rand(3,1)
    prob = ODEProblem(parameterized_lorenz!,u0,tspan,p)
    sol_lorenz = solve(prob, Tsit5())
    # vcat(sol_lorenz.t', Array(sol_lorenz))
    if i==1
        outputs=vcat(sol_lorenz.t', Array(sol_lorenz)) # Array(sol_lorenz)
    else
        outputs=hcat(outputs,vcat(sol_lorenz.t', Array(sol_lorenz)))
    end
end
# end of for loop
end

using DataFrames

plot(outputs[1,:],outputs[4,:])

df_output = DataFrame(outputs', :auto)
rename!(df_output,:x1 => :t)

using BenchmarkTools

# fixed time step
@time begin
sol_fixed = solve(prob,ABM54(),adaptive=false,dt=0.1)
end
# t=0.000076s

@time begin
sol_lorenz = solve(prob, Tsit5())
end
# 0.000047s

x=solve(prob, Tsit5(),saveat=0.5);
@benchmark solve(prob, Tsit5(),saveat=0.5)

########################################
# vectorised ODE system for age structured model

using DelimitedFiles

# birthrates=vcat(2314,zeros(98))
# deathrates=vcat(ones(18)*0.00003,ones(45)*0.000001,zeros(18),ones(9)*0.000001,ones(9)*0.000179)
# K_m = readdlm("../../RSV-resurgence-model/extras/K_m.csv", ',', Float64)
# contmatr_rowvector = readdlm("../../RSV-resurgence-model/extras/contmatr_rowvector.csv",',',Float64)
# infvars_inds = reshape([4,5,6,13,14,15,22,23,24,31,32,33,40,41,42,49,50,51,58,59,60,67,68,69,76,77,78,85,86,87,94,95,96],3,11)
# suscvars_inds=[1 2 3 10 11 12 19 20 21 28 29 30 37 38 39 46 47 48 55 56 57 64 65 66 73 74 75 82 83 84 91 92 93]
# deltasusc=readdlm("../../RSV-resurgence-model/extras/deltasusc.csv", ',', Float64)
# prot_inf_ind = 7
#prot_adults_childb=[79 80 81]
# susc_adults_childb=[73 74 75]
ode_inputs = [vcat(2314,zeros(98)), # 1) birth rates
            vcat(ones(18)*0.00003,ones(45)*0.000001,zeros(18),ones(9)*0.000001,ones(9)*0.000179), # 2) deathrates
    readdlm("../../RSV-resurgence-model/extras/K_m.csv", ',', Float64), # 3) K_m
    readdlm("../../RSV-resurgence-model/extras/contmatr_rowvector.csv",',',Float64), # 4) contmatr_rowvector
    reshape([4,5,6,13,14,15,22,23,24,31,32,33,40,41,42,49,50,51,58,59,60,67,68,69,76,77,78,85,86,87,94,95,96],3,11), # 5) infvars_inds
    [1 2 3 10 11 12 19 20 21 28 29 30 37 38 39 46 47 48 55 56 57 64 65 66 73 74 75 82 83 84 91 92 93], # 6) suscvars_inds
    readdlm("../../RSV-resurgence-model/extras/deltasusc.csv", ',', Float64), # 7) deltasusc
                7, # 8) prot_inf_ind
                [79 80 81], # 9) prot_adults_childb
                [73 74 75], # 10) susc_adults_childb
                7]; # lockdown year
                # ODE system
function vector_ode_sirs!(du,u,ode_inputs,t)
    birthrates = ode_inputs[1]; deathrates=ode_inputs[2]; K_m=ode_inputs[3];
    contmatr_rowvector=ode_inputs[4]; infvars_inds=ode_inputs[5]; suscvars_inds=ode_inputs[6];
    deltasusc=ode_inputs[7]; prot_inf_ind=ode_inputs[8]; 
    prot_adults_childb=ode_inputs[9]; susc_adults_childb=ode_inputs[10]; lockdown_yr=ode_inputs[11]
    # proportion of adults susceptible
    proport_adult_susc=sum(u[susc_adults_childb])/(sum(u[susc_adults_childb])+sum(u[prot_adults_childb]))
    birthrates[prot_inf_ind]=(1-proport_adult_susc)*birthrates[1]
    birthrates[1]=proport_adult_susc*birthrates[1]
    # infection terms
    inf_vars_stacked = hcat(u[infvars_inds[:,1],:],u[infvars_inds[:,2],:],u[infvars_inds[:,3],:],
                        u[infvars_inds[:,4],:],u[infvars_inds[:,5],:],u[infvars_inds[:,6],:],
                        u[infvars_inds[:,7],:],u[infvars_inds[:,8],:],u[infvars_inds[:,9],:],
                        u[infvars_inds[:,10],:],u[infvars_inds[:,11],:] )
    # "lockdown"
    scale_seas_force=1
    if t > (lockdown_yr*365+80) && t<((lockdown_yr+1)*365+80)
        scale_seas_force=0.1
    end
    seas_force_t=(1 + cos(pi*((t - 330)/365))^2)*scale_seas_force
    # force of infection
    lambda_vect=seas_force_t*Diagonal(reshape(deltasusc,33)) * contmatr_rowvector * ([1 1 1] * inf_vars_stacked)'
    infection_vect=Diagonal(vec(u[suscvars_inds])) * lambda_vect
    # print(round(seas_force_t,digits=2),", ")
    F_vect = zeros(99)
    F_vect[hcat(suscvars_inds,reshape(infvars_inds,1,33))']=vcat(-infection_vect,infection_vect)
    du_test = vcat(birthrates + F_vect + K_m * u[1:99] - deathrates .* u[1:99],infection_vect)
    for k_u=1:length(u)
        du[k_u]=du_test[k_u]
    end
end

# initial values
u0 = readdlm("../../RSV-resurgence-model/extras/initvals_sirs_model.csv", ',', Float64);
tspan = (0.0,3650.0)
vector_ode_prob = ODEProblem(vector_ode_sirs!,u0,tspan,ode_inputs);
@time begin
sol_vector_ode_prob = solve(vector_ode_prob, Tsit5(),saveat=1);
end

# plot
plot(sol_vector_ode_prob[reshape(inf_vars_inds[:,1:4],12),1,:]')

# callback to introduce importations
# condition(u,t,integrator) = t>1175 & t<1540
dosetimes = collect((1:(10*12))*30)
condition(u,t,integrator) = t ∈ dosetimes
affect!(integrator) = integrator.u[reshape(ode_inputs[5],33)] .+= 10
cb = DiscreteCallback(condition,affect!)

# call with import events
@time begin
sol_event = solve(vector_ode_prob,Tsit5(),callback=cb,tstops=dosetimes)
end

plot(sol_event[reshape(inf_vars_inds[:,1:4],12),1,:]')

################################################################################
################################################################################
# time-dependent input

using UnPack

function F_seir(x, p, t)
    s, e, i, r, R₀, c, d = x
    @unpack σ, γ, R̄₀, η, δ = p

    return [-γ*R₀*s*i;        # ds/dt
            γ*R₀*s*i -  σ*e;  # de/dt
            σ*e - γ*i;        # di/dt
            γ*i;              # dr/dt
            η*(R̄₀(t, p) - R₀);# dR₀/dt
            σ*e;              # dc/dt
            δ*γ*i;            # dd/dt
            ]
end

# Parameters
p_gen = @with_kw (T=550.0, γ=1.0/18, σ=1/5.2, η=1.0/20,
R₀_n = 1.6, δ = 0.01, N = 3.3E8, R̄₀ = (t, p) -> p.R₀_n);

p = p_gen()  # use all default parameters
# init values
i_0 = 1E-7; e_0 = 4.0 * i_0; s_0 = 1.0 - i_0 - e_0
x_0 = [s_0, e_0, i_0, 0.0, p.R₀_n, 0.0, 0.0]
tspan = (0.0, p.T)
prob = ODEProblem(F, x_0, tspan, p)

sol = solve(prob, Tsit5())
@show length(sol.t);
