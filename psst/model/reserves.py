from pyomo.environ import *



def _zone_generator_map(m, g):
    raise NotImplementedError("Zonal reserves not implemented yet")


def _form_generator_reserve_zones(m,rz):
    return (g for g in m.Generators if m.ReserveZoneLocation[g]==rz)


def _reserve_requirement_rule(m, t):
    return m.ReserveFactor * sum(value(m.Demand[b,t]) for b in m.Buses)

def initialize_global_reserves(model, reserve_factor=0.0, reserve_requirement=_reserve_requirement_rule):

    model.ReserveFactor = Param(within=Reals, initialize=reserve_factor, mutable=True)
    model.ReserveRequirement = Param(model.TimePeriods, initialize=reserve_requirement, within=NonNegativeReals, default=0.0, mutable=True)


def initialize_spinning_reserves(model):
    model.SpinningReserveUpAvailable = Var(model.Generators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)


def _regulating_requirement_rule(m, t):
    return m.RegulatingReserveFactor * sum(value(m.Demand[b,t]) for b in m.Buses)

def initialize_regulating_reserves_requirement(model, regulating_reserve_factor=0.0, regulating_reserve_requirement=_regulating_requirement_rule):

    model.RegulatingReserveFactor = Param(within=Reals, initialize=regulating_reserve_factor, mutable=True)
    model.RegulatingReserveRequirement = Param(model.TimePeriods, initialize=regulating_reserve_requirement, within=NonNegativeReals, default=0.0, mutable=True)
       
def initialize_regulating_reserves(model):
    model.RegulatingReserveUpAvailable = Var(model.Generators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)    
    model.RegulatingReserveDnAvailable = Var(model.Generators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)    

def _flexible_ramp_up_requirement_rule(m, t):
    if  t ==(len(m.TimePeriods) - 1) :
        return max(0.0,(m.FlexibleRampFactor * sum(value(m.Demand[b,t]) for b in m.Buses)))
    else:    
        return max(0.0,((1+m.FlexibleRampFactor) * sum(value(m.Demand[b,t+1]) for b in m.Buses) - sum(value(m.Demand[b,t]) for b in m.Buses)))

def _flexible_ramp_down_requirement_rule(m, t):
    if  t ==(len(m.TimePeriods) -1) :
        return max(0.0, (m.FlexibleRampFactor * sum(value(m.Demand[b,t]) for b in m.Buses)))
    else:    
        return max(0.0, (sum(value(m.Demand[b,t]) for b in m.Buses) - (1-m.FlexibleRampFactor) * sum(value(m.Demand[b,t+1]) for b in m.Buses)) ) 

def initialize_flexible_ramp(model, flexible_ramp_factor=0.0, flexible_ramp_Up_requirement=_flexible_ramp_up_requirement_rule, flexible_ramp_Dn_requirement=_flexible_ramp_down_requirement_rule):

    model.FlexibleRampFactor = Param(within=Reals, initialize=flexible_ramp_factor, mutable=True)
    model.FlexibleRampUpRequirement = Param(model.TimePeriods, initialize=flexible_ramp_Up_requirement, within=Reals, default=0.0, mutable=True)    
    model.FlexibleRampDnRequirement = Param(model.TimePeriods, initialize=flexible_ramp_Dn_requirement, within=Reals, default=0.0, mutable=True)       

def initialize_flexible_ramp_reserves(model):
    model.FlexibleRampUpAvailable = Var(model.Generators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.FlexibleRampDnAvailable = Var(model.Generators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)

def initialize_zonal_reserves(model, buses=None, generator_reserve_zones=_form_generator_reserve_zones, zone_generator_map=_zone_generator_map):
    if buses is None:
        buses = model.Buses
    model.ReserveZones = Set(initialize=buses)
    model.ZonalReserveRequirement = Param(model.ReserveZones, model.TimePeriods, default=0.0, mutable=True, within=NonNegativeReals)
    model.ReserveZoneLocation = Param(model.Generators, initialize=zone_generator_map)

    model.GeneratorsInReserveZone = Set(model.ReserveZones, initialize=generator_reserve_zones)

