# ============================================================================ #
# ==================       Capacity Expansion Problem       ================== #
# ============================================================================ #
# using Pkg
# Pkg.add("CSV")
# Pkg.add("Distributions")
# Pkg.add("DataFrames")
# Pkg.add("JuMP")
# Pkg.add("GLPK")
using Distributions
using CSV
using DataFrames
using JuMP
using GLPK


filePath = @__DIR__;

# ======================     Parâmetros do Problema   ======================== #

nCenarios   = 5;
dmin = 50;
dmax = 150;
d = rand(Uniform(dmin, dmax), nCenarios);
Ω           = 1:nCenarios;


# ============================     Data Read    ============================== #

c           = 20;
u           = 150;
r           = 10;
q           = 60;
avg_q       = 60;


Pathread    = string(filePath, "\\IN_d.csv");
df          = CSV.read(Pathread, DataFrame);
d           = df.d[:];
avg_d       = mean(d);


# ============================================================================ #

# =================     Problema Amostral    ===================== #

CapExpModel = Model(with_optimizer(GLPK.Optimizer));
#CapExpModel = Model(with_optimizer(Gurobi.Optimizer));

# ========== Variáveis de Decisão ========== #

@variable(CapExpModel, x >= 0);
@variable(CapExpModel, y[Ω] >= 0);
@variable(CapExpModel, z[Ω] >= 0);

# ========== Restrições ========== #

@constraint(CapExpModel, Rest1, x <= u);
@constraint(CapExpModel, Rest2[ω in Ω], y[ω] <= d[ω]);
@constraint(CapExpModel, Rest3[ω in Ω], y[ω] + z[ω] <= x);

# ========== Função Objetivo ========== #

@objective(CapExpModel, Max, -c*x + sum((1/nCenarios)*(q*y[ω] + r*z[ω]) for ω in Ω));

optimize!(CapExpModel);

status      = termination_status(CapExpModel);
TotalCost   = JuMP.objective_value(CapExpModel);
xOpt        = JuMP.value.(x);
yOpt        = JuMP.value.(y);
zOpt        = JuMP.value.(z);

println("\n=================================")

println("\nStatus: ", status, "\n");

println(xOpt);

println("\nTotal Cost:      ", TotalCost);

println("\n=================================")

# ============================================================================ #

# =================     Informação Perfeita     ===================== #

global CostEVPI    = zeros(nCenarios);

for ω in Ω

    CapExpModel = Model(with_optimizer(GLPK.Optimizer));

    # ========== Variáveis de Decisão ========== #

    @variable(CapExpModel, x[I] >= 0);
    @variable(CapExpModel, y[I] >= 0);
    @variable(CapExpModel, z >= 0);

    # ========== Restrições ========== #

    @constraint(CapExpModel, Rest1[i in I], x[i] + x0[i] <= u[i]);
    @constraint(CapExpModel, Rest2[i in I], y[i] <= x[i] + x0[i]);
    @constraint(CapExpModel, Rest3, sum(y[i] for i in I) + z >= d[ω]);

    # ========== Função Objetivo ========== #

    @objective(CapExpModel, Min, sum(c[i]*x[i] for i in I) + q[ω]*z + sum(p[i]*y[i] for i in I) );

    optimize!(CapExpModel);

    status      = termination_status(CapExpModel);
    CostEVPI[ω]    = JuMP.objective_value(CapExpModel);

end;

TotalCostEVPI   = (1/nCenarios)*sum(CostEVPI[ω] for ω in Ω);
EVPI            = TotalCost - TotalCostEVPI;

println("\n=================================");
println("EVPI:      ", EVPI);
println("\n=================================");

# ============================================================================ #

# =================     Valor da Solução Estocástica     ===================== #

CapExpModel = Model(with_optimizer(GLPK.Optimizer));

# ========== Variáveis de Decisão ========== #

@variable(CapExpModel, x[I] >= 0);
@variable(CapExpModel, y[I] >= 0);
@variable(CapExpModel, z >= 0);

# ========== Restrições ========== #

@constraint(CapExpModel, Rest1[i in I], x[i] + x0[i] <= u[i]);
@constraint(CapExpModel, Rest2[i in I], y[i] <= x[i] + x0[i]);
@constraint(CapExpModel, Rest3, sum(y[i] for i in I) + z >= avg_d);

# ========== Função Objetivo ========== #

@objective(CapExpModel, Min, sum(c[i]*x[i] for i in I) + avg_q*z + sum(p[i]*y[i] for i in I) );

optimize!(CapExpModel);

status      = termination_status(CapExpModel);
xVSS        = JuMP.value.(x);

global CostVSS    = zeros(nCenarios);

for ω in Ω

    CapExpModel = Model(with_optimizer(GLPK.Optimizer));

    # ========== Variáveis de Decisão ========== #

    @variable(CapExpModel, x[I] >= 0);
    @variable(CapExpModel, y[I] >= 0);
    @variable(CapExpModel, z >= 0);

    # ========== Restrições ========== #

    @constraint(CapExpModel, Rest1[i in I], x[i] + x0[i] <= u[i]);
    @constraint(CapExpModel, Rest2[i in I], y[i] <= x[i] + x0[i]);
    @constraint(CapExpModel, Rest3, sum(y[i] for i in I) + z >= d[ω]);

    @constraint(CapExpModel, RestVSS[i in I], x[i] == xVSS[i])

    # ========== Função Objetivo ========== #

    @objective(CapExpModel, Min, sum(c[i]*x[i] for i in I) + q[ω]*z + sum(p[i]*y[i] for i in I) );

    optimize!(CapExpModel);

    status      = termination_status(CapExpModel);
    CostVSS[ω]  = JuMP.objective_value(CapExpModel);

end;

TotalCostVSS   = (1/nCenarios)*sum(CostVSS[ω] for ω in Ω);
VSS            = TotalCostVSS - TotalCost;

println("\n=================================");
println("VSS:      ", VSS);
println("\n=================================");

# ============================================================================ #

# =================     Print - Resultados Finais     ===================== #

println("\n=================================")

println("\nStatus: ", status, "\n");

for i in I
    println("   x[", i, "] = ", xOpt[i])
end

println("\nExpansion Cost:  ", sum(c[i]*xOpt[i] for i in I));
println("Operations Cost: ", (1/nCenarios)*sum(q[ω]*zOpt[ω] + sum(p[i]*yOpt[i,ω] for i in I) for ω in Ω));
println("\nTotal Cost:      ", TotalCost);

println("\n\nEVPI:      ", EVPI);
println("\n\nVSS:      ", VSS);

println("\n=================================")