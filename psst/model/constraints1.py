import numpy as np
from functools import partial

import logging

from pyomo.environ import *

logger = logging.getLogger(__file__)

eps = 1e-3

def fix_first_angle_rule(m,t, slack_bus=1):
    return m.Angle[m.Buses[slack_bus], t] == 0.0

def lower_line_power_bounds_rule(m, l, t):
    if m.EnforceLine[l] and np.any(np.absolute(m.ThermalLimit[l]) > eps):
        return -m.ThermalLimit[l] <= m.LinePower[l, t]
    else:
        return Constraint.Skip

def upper_line_power_bounds_rule(m, l, t):
    if m.EnforceLine[l] and np.any(np.absolute(m.ThermalLimit[l]) > eps):
        return m.ThermalLimit[l] >= m.LinePower[l, t]
    else:
        return Constraint.Skip

def line_power_ptdf_rule(m, l, t):
    return m.LinePower[l,t] == sum(float(m.PTDF[l, i]) * m.NetPowerInjectionAtBus[b, t] for i, b in enumerate(m.Buses))

def line_power_rule(m, l, t):
    if m.B[l] == 99999999:
        logger.debug(" Line Power Angle constraint skipped for line between {} and {} ".format(m.BusFrom[l], m.BusTo[l]))
        return Constraint.Skip
    else:
        return m.LinePower[l,t] == m.B[l] * (m.Angle[m.BusFrom[l], t] - m.Angle[m.BusTo[l], t])

def calculate_total_demand(m, t):
    return m.TotalDemand[t] == sum(m.Demand[b,t] for b in m.Buses)

def neg_load_generate_mismatch_tolerance_rule(m, b):
   return sum((m.negLoadGenerateMismatch[b,t] for t in m.TimePeriods)) >= 0.0

def pos_load_generate_mismatch_tolerance_rule(m, b):
   return sum((m.posLoadGenerateMismatch[b,t] for t in m.TimePeriods)) >= 0.0

def power_balance(m, b, t, has_storage=False, has_non_dispatchable_generators=False):
    # Power balance at each node (S)
    # bus b, time t (S)

    constraint = m.NetPowerInjectionAtBus[b, t] + sum(m.LinePower[l,t] for l in m.LinesTo[b]) \
           - sum(m.LinePower[l,t] for l in m.LinesFrom[b])  == 0
         #  + m.LoadGenerateMismatch[b,t] == 0

    return constraint

def net_power_at_bus_rule(m, b, t, has_storage=False, has_non_dispatchable_generators=False):
    constraint = sum((1 - m.GeneratorForcedOutage[g,t]) * m.GeneratorBusContributionFactor[g, b] * m.PowerGenerated[g, t] for g in m.GeneratorsAtBus[b])

    if has_storage is True:
        constraint = constraint + sum(m.PowerOutputStorage[s, t] for s in m.StorageAtBus[b]) \
           - sum(m.PowerInputStorage[s, t] for s in m.StorageAtBus[b])

    if has_non_dispatchable_generators is True:
        constraint = constraint + sum(m.NondispatchablePowerUsed[g, t] for g in m.NondispatchableGeneratorsAtBus[b])

    constraint = constraint + m.LoadGenerateMismatch[b,t]

    constraint = constraint - m.Demand[b, t]

    constraint = m.NetPowerInjectionAtBus[b, t] == constraint

    return constraint


# give meaning to the positive and negative parts of the mismatch
def posneg_rule(m, b, t):
    return m.posLoadGenerateMismatch[b, t] - m.negLoadGenerateMismatch[b, t] == m.LoadGenerateMismatch[b, t]

def global_posneg_rule(m, t):
    return m.posGlobalLoadGenerateMismatch[t] - m.negGlobalLoadGenerateMismatch[t] == m.GlobalLoadGenerateMismatch[t]

def enforce_reserve_requirements_rule(m, t, has_storage=False,
                                        has_non_dispatchable_generators=False,
                                        has_global_reserves=False):

    constraint = sum(m.MaximumPowerAvailable[g, t] for g in m.Generators)

    if has_non_dispatchable_generators is True:
        constraint = constraint + sum(m.NondispatchablePowerUsed[n,t] for n in m.NondispatchableGenerators)

    if has_storage is True:
        constraint = constraint + sum(m.PowerOutputStorage[s,t] for s in m.Storage)

    if has_global_reserves is True:
        constraint = constraint - m.ReserveRequirement[t]

    constraint = constraint >= m.TotalDemand[t] + m.GlobalLoadGenerateMismatch[t]

    return constraint
#+ m.RegulatingReserveUpAvailable[g, t]
def calculate_regulating_reserve_up_available_per_generator(m, g, t):
    return m.SpinningReserveUpAvailable[g, t]  == m.MaximumPowerAvailable[g,t] - m.PowerGenerated[g,t]

def enforce_zonal_reserve_requirement_rule(m, rz, t):
    return sum(m.SpinningReserveUpAvailable[g,t] for g in m.GeneratorsInReserveZone[rz]) >= m.ZonalReserveRequirement[rz, t]


# def enforce_regulating_up_reserve_requirement_rule(m, t):
#     return sum(m.RegulatingReserveUpAvailable[g,t] for g in m.Generators) >= m.RegulatingReserveRequirement[t]
# 
# def enforce_regulating_down_reserve_requirement_rule(m, t):
#     return sum(m.RegulatingReserveDnAvailable[g,t] for g in m.Generators) >= m.RegulatingReserveRequirement[t]

def enforce_unit_status_constraint(m,g,t):
    if t ==0:
        return m.UnitOn[g, t] - m.UnitOnT0[g] == m.UnitSu[g, t]  - m.UnitSd[g, t] 
    else:
        return m.UnitOn[g, t] - m.UnitOn[g, t-1] == m.UnitSu[g, t]  - m.UnitSd[g, t]     

def enforce_generator_output_limits_rule_part_a(m, g, t):
   return m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t]

def enforce_generator_output_limits_rule_part_b(m, g, t):
   return m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]

def enforce_generator_output_limits_rule_part_c(m, g, t):
   return m.MaximumPowerAvailable[g,t] <= m.MaximumPowerOutput[g] * m.UnitOn[g, t]

def enforce_max_available_ramp_up_rates_rule(m, g, t):
   # 4 cases, split by (t-1, t) unit status (RHS is defined as the delta from m.PowerGenerated[g, t-1])
   # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound due to unit being off)
   # (0, 1) - unit switching on:  RHS = startup ramp limit
   # (1, 0) - unit switching off: RHS = standard ramp limit minus startup ramp limit plus maximum power output (degenerate upper bound due to unit off)
   # (1, 1) - unit staying on:    RHS = standard ramp limit plus power generated in previous time period
   if t == 0:
      return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedT0[g] + \
                                              m.NominalRampUpLimit[g] * m.UnitOnT0[g] + \
                                              m.StartupRampLimit[g] * m.UnitSu[g, t] 
   else:
      return m.MaximumPowerAvailable[g, t] <= m.PowerGenerated[g, t-1] + \
                                              m.NominalRampUpLimit[g] * m.UnitOn[g, t-1] + \
                                              m.StartupRampLimit[g] * m.UnitSu[g, t] 

def enforce_max_available_ramp_down_rates_rule(m, g, t):
    # 4 cases, split by (t, t+1) unit status
    # (0, 0) - unit staying off:   RHS = 0 (degenerate upper bound)
    # (0, 1) - unit switching on:  RHS = maximum generator output minus shutdown ramp limit (degenerate upper bound) - this is the strangest case.
    # (1, 0) - unit switching off: RHS = shutdown ramp limit
    # (1, 1) - unit staying on:    RHS = maximum generator output (degenerate upper bound)
    #NOTE: As expressed in Carrion-Arroyo and subsequently here, this constraint does NOT consider ramp down from initial conditions to t=1!
    #if t == value(m.NumTimePeriods):
    #   return Constraint.Skip
    #else:
    #   return m.MaximumPowerAvailable[g, t] <= \
            #          m.MaximumPowerOutput[g] * m.UnitOn[g, t+1] + \
            #          m.ShutdownRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t+1])

    #This version fixes the problem with ignoring initial conditions mentioned in the above note

    if t == 0:
        # Not 100% sure of this one since there is no MaximumPowerAvailableT0
        return m.PowerGeneratedT0[g] - m.MaximumPowerAvailable[g, t] <= \
                m.NominalRampDownLimit[g] * m.UnitOn[g,t] + \
                m.ShutdownRampLimit[g] * m.UnitSd[g, t]

    else:
        return m.MaximumPowerAvailable[g, t-1] - m.MaximumPowerAvailable[g, t] <= \
                m.NominalRampDownLimit[g] * m.UnitOn[g, t] + \
                m.ShutdownRampLimit[g] * m.UnitSd[g, t]

def enforce_ramp_down_limits_rule(m, g, t):
    # 4 cases, split by (t-1, t) unit status:
    # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound)
    # (0, 1) - unit switching on:  RHS = standard ramp-down limit minus shutdown ramp limit plus maximum generator output - this is the strangest case
    #NOTE: This may never be physically true, but if a generator has ShutdownRampLimit >> MaximumPowerOutput, this constraint causes problems
    # (1, 0) - unit switching off: RHS = shutdown ramp limit
    # (1, 1) - unit staying on:    RHS = standard ramp-down limit -m.RegulatingReserveDnAvailable[g,t]
    if t == 0:
        return m.PowerGeneratedT0[g] - (m.PowerGenerated[g, t]) <= \
                m.NominalRampDownLimit[g] * m.UnitOn[g, t] + \
                m.ShutdownRampLimit[g]  * m.UnitSd[g, t]  
    else:
       return m.PowerGenerated[g, t-1] - (m.PowerGenerated[g, t]) <= \
               m.NominalRampDownLimit[g]  * m.UnitOn[g, t] + \
               m.ShutdownRampLimit[g]  * m.UnitSd[g, t]  

def enforce_ramp_up_limits_rule(m, g, t):
    # 4 cases, split by (t-1, t) unit status:
    # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound)
    # (0, 1) - unit switching on:  RHS = standard ramp-down limit minus shutdown ramp limit plus maximum generator output - this is the strangest case
    #NOTE: This may never be physically true, but if a generator has ShutdownRampLimit >> MaximumPowerOutput, this constraint causes problems
    # (1, 0) - unit switching off: RHS = shutdown ramp limit
    # (1, 1) - unit staying on:    RHS = standard ramp-down limit               
   if t == 0:
      return m.PowerGenerated[g, t] <= m.PowerGeneratedT0[g] + \
                                              m.NominalRampUpLimit[g] * m.UnitOnT0[g] + \
                                              m.StartupRampLimit[g] * m.UnitSu[g, t] 
   else:
      return m.PowerGenerated[g, t] <= m.PowerGenerated[g, t-1] + \
                                              m.NominalRampUpLimit[g] * m.UnitOn[g, t-1] + \
                                              m.StartupRampLimit[g] * m.UnitSu[g, t]           


# compute startup costs for each generator, for each time period
def compute_hot_start_rule(m, g, t):
    if t <= value(m.ColdStartHours[g]):
        if t - value(m.ColdStartHours[g]) <= value(m.UnitOnT0State[g]):
            m.HotStart[g, t] = 1
            m.HotStart[g, t].fixed = True
            return Constraint.Skip
        else:
            return m.HotStart[g, t] <= sum( m.UnitOn[g, i] for i in range(1, t) )
    else:
        return m.HotStart[g, t] <= sum( m.UnitOn[g, i] for i in range(t - m.ColdStartHours[g], t) )


def compute_startup_costs_rule_minusM(m, g, t):
    if t == 0:
        return m.StartupCost[g, t] >= m.ColdStartCost[g] - (m.ColdStartCost[g] - m.HotStartCost[g])*m.HotStart[g, t] \
                                      - m.ColdStartCost[g]*(1 - (m.UnitOn[g, t] - m.UnitOnT0[g]))
    else:
        return m.StartupCost[g, t] >= m.ColdStartCost[g] - (m.ColdStartCost[g] - m.HotStartCost[g])*m.HotStart[g, t] \
                                      - m.ColdStartCost[g]*(1 - (m.UnitOn[g, t] - m.UnitOn[g, t-1]))


# compute the per-generator, per-time period shutdown costs.
def compute_shutdown_costs_rule(m, g, t):
    if t == 0:
        return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (m.UnitOnT0[g] - m.UnitOn[g, t])
    else:
        return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t])


def enforce_up_time_constraints_initial(m, g):
   if value(m.InitialTimePeriodsOnLine[g]) == 0:
      return Constraint.Skip
   return sum((1 - m.UnitOn[g, t]) for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOnLine[g])) == 0.0


# constraint for each time period after that not involving the initial condition.
@simple_constraint_rule
def enforce_up_time_constraints_subsequent(m, g, t):
    if t > value(m.InitialTimePeriodsOnLine[g]) and t <= (len(m.NumTimePeriods)-1) :
        return sum(m.UnitSu[g, n] for n in m.TimePeriods if n >= (t - m.MinimumUpTime[g] +1) and n <= t ) -m.UnitOn[g, t] <= 0.0 


# constraint due to initial conditions.
def enforce_down_time_constraints_initial(m, g):
   if value(m.InitialTimePeriodsOffLine[g]) == 0:
      return Constraint.Skip
   return sum(m.UnitOn[g, t] for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOffLine[g])) == 0.0


# constraint for each time period after that not involving the initial condition.
@simple_constraint_rule
def enforce_down_time_constraints_subsequent(m, g, t):
    if t > value(m.InitialTimePeriodsOffLine[g])  and t <= (len(m.NumTimePeriods)-1) :
        return sum(m.UnitSd[g, n] for n in m.TimePeriods if n >= (t - m.MinimumDownTime[g] +1) and n <= t ) + m.UnitOn[g, t] <= 1 


def commitment_in_stage_st_cost_rule(m, st):
    return m.CommitmentStageCost[st] == (sum(m.StartupCost[g,t] + m.ShutdownCost[g,t] for g in m.Generators for t in m.CommitmentTimeInStage[st]) + sum(sum(m.UnitOn[g,t] for t in m.CommitmentTimeInStage[st]) * m.MinimumProductionCost[g] * m.TimePeriodLength for g in m.Generators))

def generation_in_stage_st_cost_rule(m, st):
    return m.GenerationStageCost[st] == sum(m.ProductionCost[g, t] for g in m.Generators for t in m.GenerationTimeInStage[st]) + m.LoadMismatchPenalty * \
                                   (sum(m.posLoadGenerateMismatch[b, t] + m.negLoadGenerateMismatch[b, t] for b in m.Buses for t in m.GenerationTimeInStage[st]) + \
                                     sum(m.posGlobalLoadGenerateMismatch[t] + m.negGlobalLoadGenerateMismatch[t] for t in m.GenerationTimeInStage[st]))   \
                                    + sum(m.reserve_price[g] * m.SpinningReserveUpAvailable[g, t] for g in m.Generators for t in m.GenerationTimeInStage[st]) \
                                    + sum(m.regulation_price[g] * m.RegulatingReserveUpAvailable[g, t]  for g in m.Generators for t in m.GenerationTimeInStage[st]) \
                                    + sum(m.regulation_price[g] * m.RegulatingReserveDnAvailable[g, t]  for g in m.Generators for t in m.GenerationTimeInStage[st])
    
 #   return m.GenerationStageCost[st] == c

def StageCost_rule(m, st):
    return m.StageCost[st] == m.GenerationStageCost[st] + m.CommitmentStageCost[st]
def total_cost_objective_rule(m):
   return sum(m.StageCost[st] for st in m.StageSet)



def constraint_net_power(model, has_storage=False, has_non_dispatchable_generators=False):
    partial_net_power_at_bus_rule = partial(net_power_at_bus_rule, has_storage=has_storage, has_non_dispatchable_generators=has_non_dispatchable_generators)
    model.CalculateNetPowerAtBus = Constraint(model.Buses, model.TimePeriods, rule=partial_net_power_at_bus_rule)


################################################

def constraint_line(model, ptdf=None, slack_bus=1):

    model.LinePowerConstraintLower = Constraint(model.TransmissionLines, model.TimePeriods, rule=lower_line_power_bounds_rule)
    model.LinePowerConstraintHigher = Constraint(model.TransmissionLines, model.TimePeriods, rule=upper_line_power_bounds_rule)

    if ptdf is not None:
        model.PTDF = ptdf
        model.CalculateLinePower = Constraint(model.TransmissionLines, model.TimePeriods, rule=line_power_ptdf_rule)
    else:
        partial_fix_first_angle_rule = partial(fix_first_angle_rule, slack_bus=slack_bus)
        model.FixFirstAngle = Constraint(model.TimePeriods, rule=partial_fix_first_angle_rule)
        model.CalculateLinePower = Constraint(model.TransmissionLines, model.TimePeriods, rule=line_power_rule)

def constraint_total_demand(model):
    model.CalculateTotalDemand = Constraint(model.TimePeriods, rule=calculate_total_demand)


def constraint_load_generation_mismatch(model):
    model.PosLoadGenerateMismatchTolerance = Constraint(model.Buses, rule=pos_load_generate_mismatch_tolerance_rule)
    model.NegLoadGenerateMismatchTolerance = Constraint(model.Buses, rule=neg_load_generate_mismatch_tolerance_rule)
    model.Defineposneg_Mismatch = Constraint(model.Buses, model.TimePeriods, rule = posneg_rule)
    model.Global_Defineposneg_Mismatch = Constraint(model.TimePeriods, rule = global_posneg_rule)


def constraint_power_balance(model, has_storage=False, has_non_dispatchable_generators=False):

    fn_power_balance = partial(power_balance, has_storage=has_storage, has_non_dispatchable_generators=has_non_dispatchable_generators)
    model.PowerBalance = Constraint(model.Buses, model.TimePeriods, rule=fn_power_balance)


def constraint_reserves(model, has_storage=False,
                            has_non_dispatchable_generators=False,
                            has_global_reserves=True,
                            has_spinning_reserves=True,
                            has_zonal_reserves=False,
                            has_regulating_reserves=True):

    if has_global_reserves is True:
        fn_enforce_reserve_requirements = partial(enforce_reserve_requirements_rule, has_storage=has_storage,
                                                has_non_dispatchable_generators=has_non_dispatchable_generators,
                                                has_global_reserves=has_global_reserves)
        model.EnforceReserveRequirements = Constraint(model.TimePeriods, rule=fn_enforce_reserve_requirements)

    if has_spinning_reserves is True:
        model.CalculateRegulatingReserveUpPerGenerator = Constraint(model.Generators, model.TimePeriods, rule=calculate_regulating_reserve_up_available_per_generator)

    if has_zonal_reserves is True:
        model.EnforceZonalReserveRequirements = Constraint(model.ReserveZones, model.TimePeriods, rule=enforce_zonal_reserve_requirement_rule)
        
    # if has_regulating_reserves is True:
    #     model.EnforceRegulatingUpReserveRequirements = Constraint(model.TimePeriods, rule=enforce_regulating_up_reserve_requirement_rule)
    #     model.EnforceRegulatingDnReserveRequirements = Constraint(model.TimePeriods, rule=enforce_regulating_down_reserve_requirement_rule)                


def constraint_generator_power(model):
    model.EnforceGeneratorStatus = Constraint(model.Generators, model.TimePeriods, rule=enforce_unit_status_constraint)
    model.EnforceGeneratorOutputLimitsPartA = Constraint(model.Generators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_a)
    model.EnforceGeneratorOutputLimitsPartB = Constraint(model.Generators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_b)
    model.EnforceGeneratorOutputLimitsPartC = Constraint(model.Generators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_c)
    model.EnforceMaxAvailableRampUpRates = Constraint(model.Generators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)
    model.EnforceMaxAvailableRampDownRates = Constraint(model.Generators, model.TimePeriods, rule=enforce_max_available_ramp_down_rates_rule)
    model.EnforceNominalRampDownLimits = Constraint(model.Generators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)
    model.EnforceNominalRampUpLimits = Constraint(model.Generators, model.TimePeriods, rule=enforce_ramp_up_limits_rule)


def constraint_up_down_time(model):

    model.EnforceUpTimeConstraintsInitial = Constraint(model.Generators, rule=enforce_up_time_constraints_initial)
    model.EnforceUpTimeConstraintsSubsequent = Constraint(model.Generators, model.TimePeriods, rule=enforce_up_time_constraints_subsequent)

    model.EnforceDownTimeConstraintsInitial = Constraint(model.Generators, rule=enforce_down_time_constraints_initial)
    model.EnforceDownTimeConstraintsSubsequent = Constraint(model.Generators, model.TimePeriods, rule=enforce_down_time_constraints_subsequent)


def production_cost_function(m, g, t, x):
    # a function for use in piecewise linearization of the cost function.
    return m.TimePeriodLength * m.PowerGenerationPiecewiseValues[g,t][x] * m.FuelCost[g]


def constraint_for_cost(model):

    model.ComputeProductionCosts = Piecewise(model.Generators * model.TimePeriods, model.ProductionCost, model.PowerGenerated, pw_pts=model.PowerGenerationPiecewisePoints, f_rule=production_cost_function, pw_constr_type='LB', warning_tol=1e-20)

    model.ComputeHotStart = Constraint(model.Generators, model.TimePeriods, rule=compute_hot_start_rule)
    model.ComputeStartupCostsMinusM = Constraint(model.Generators, model.TimePeriods, rule=compute_startup_costs_rule_minusM)
    model.ComputeShutdownCosts = Constraint(model.Generators, model.TimePeriods, rule=compute_shutdown_costs_rule)

    model.Compute_commitment_in_stage_st_cost = Constraint(model.StageSet, rule = commitment_in_stage_st_cost_rule)

    model.Compute_generation_in_stage_st_cost = Constraint(model.StageSet, rule = generation_in_stage_st_cost_rule)

    model.Compute_Stage_Cost = Constraint(model.StageSet, rule = StageCost_rule)

def objective_function(model):

    model.TotalCostObjective = Objective(rule=total_cost_objective_rule, sense=minimize)

def enforce_flexible_ramp_up_requirement_rule(m, t):
    return sum(m.FlexibleRampUpAvailable[g,t] for g in m.Generators) >= m.FlexibleRampUpRequirement[t]  
  
def enforce_flexible_ramp_down_requirement_rule(m, t):
    return sum(m.FlexibleRampDnAvailable[g,t] for g in m.Generators) >= m.FlexibleRampDnRequirement[t]    

def enforce_flexible_ramp_down_limits_rule(m, g, t):
    if t < (len(m.TimePeriods) -1):
        return  m.FlexibleRampDnAvailable[g,t]  <= \
               m.NominalRampDownLimit[g]  * m.UnitOn[g, t+1] + \
               m.ShutdownRampLimit[g]  * (m.UnitOn[g, t] - m.UnitOn[g, t+1]) + \
               m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
    else:
       return m.PowerGenerated[g, t] - m.FlexibleRampDnAvailable[g,t] >= m.MinimumPowerOutput[g] * m.UnitOn[g, t]

def enforce_flexible_ramp_up_limits_rule(m, g, t):
    if t < (len(m.TimePeriods) -1):
        return - m.FlexibleRampUpAvailable[g,t] >= \
               -1 * ( m.NominalRampUpLimit[g]  * m.UnitOn[g, t+1] ) + \
               -1 * ( m.StartupRampLimit[g]  * (m.UnitOn[g, t] - m.UnitOn[g, t+1]) ) + \
               -1 * ( m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t]) )
    else:
       return m.PowerGenerated[g, t] + m.FlexibleRampUpAvailable[g,t] <= m.MaximumPowerAvailable[g, t]
    
def constraint_for_Flexible_Ramping(model):
    model.EnforceFlexibleRampUpRates = Constraint(model.TimePeriods, rule=enforce_flexible_ramp_up_requirement_rule)
    model.EnforceFlexibleRampDownRates = Constraint(model.TimePeriods, rule=enforce_flexible_ramp_down_requirement_rule)
    model.EnforceFlexibleRampDownLimits = Constraint(model.Generators, model.TimePeriods, rule=enforce_flexible_ramp_down_limits_rule)
    model.EnforceFlexibleRampUpLimits = Constraint(model.Generators, model.TimePeriods, rule=enforce_flexible_ramp_up_limits_rule)
    
    



