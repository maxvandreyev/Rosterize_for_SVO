import math
import pyomo
import csv
import pandas as pd
import numpy as np
from pyomo.common.tempfiles import TempfileManager
#TempfileManager.tempdir = os.path.expanduser('~/nordstar-tmp')
from pyomo.environ import (value, ConcreteModel, ConstraintList, Var, NonNegativeIntegers,
        NonNegativeReals, Boolean, Reals, Constraint, Objective, SolverFactory,
        TerminationCondition, Expression, Param, Suffix)
from pyomo.opt.parallel import SolverManagerFactory
import datetime
from collections import defaultdict, Counter, OrderedDict


import argparse
import time
import random


def make_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w','--wide-algo', type=str, default = "", choices=["", "only_wide","wide_first"], help =' only_wide - can stay near "wide" jetbridge stands')
    parser.add_argument('-c','--case', type=int, default = 1, choices=[1,2,3], help =' case 1 - public, 2 - private, 3 - leaderbord ')
    parser.add_argument('-rm','--round-minutes', type=int, default = 3, help =' Minutes to round ')
    parser.add_argument('-sp','--soft-period', type=int, default = 30, help =' Soft period of slidig window which do not fix (in time ofr round minutes)')
    parser.add_argument('-is','--initial-size', type=int, default = 30, help =' Additional size of first step')
    parser.add_argument('-s','--solver', type=str, default = "cbc", choices=["cbc", "gurobi"], help =' solver ')
    parser.add_argument('-st','--steps', type=int, default = 4, help =' number of steps of time window ')
    parser.add_argument('-clm','--cut-long-minutes', type=int, default = 30,  help =' cut from model long bus trips ')
    parser.add_argument('-b','--baseline', default = False,  action='store_true', help =' load base line instead of real file ')
    parser.add_argument('-bt','--by_terminal', default = 0, type=int, help =' Decompose problem by terminal, if non zer - number of parts')
    parser.add_argument('-bw','--backward', default = False,  action='store_true', help =' sliding window backward. not real case but here possible ')
    return parser


def get_aircraft_class(capacity, ac):

    a= min( [(r.Max_Seats, r.Aircraft_Class) for r in ac.itertuples() if r.Max_Seats >= capacity])
    return a[1]

def intervals_not_intersects(i1, i2):
    #print(i1, i2)
    if i1[1] < i2[0] :
        return True
    if i2[1] < i1[0] :
        return True
    return False

def find_stand(curint, aircraft_stands, f, gst, stands_intervals, timetablerow,  wide=False, busy=(0,0), model=None):
    intersections = {}
    stands=[]
    for stand in aircraft_stands.itertuples() :
        st2 = stand.Aircraft_Stand
        #st2  = stand.group_index_old
        if stand.Aircraft_Stand == gst :
            stands.append(st2)
    wide_stands={}
    #print("stands", sorted(stands))
    for cst in sorted(stands) :
        if (cst-1) not in wide_stands or True :
            
            #print("cst", cst)
            wide_stands[cst] =1
    #print("wide stands", wide_stands)
    #exit()

    good_stands={}
    for stand in aircraft_stands.itertuples() :
        #print("stand", stand)
        st = stand.Index
        st2 = stand.Aircraft_Stand
        if stand.Aircraft_Stand == gst and (not wide or st2 in wide_stands) :
            #print("intervals", FAS_TIMES[ f, gst] , "si", stands_intervals[st2])
            fit = not any([not intervals_not_intersects(interval, curint) for interval in stands_intervals[st2]])
            if fit :
                good_stands[st2] = max([-100] + [ interval[1] for interval in stands_intervals[st2] if interval[1] < curint[0]])
                if wide :
                    stands_intervals[st2].append((curint[0], curint[1], f, "w"))
                else :
                    stands_intervals[st2].append((curint[0], curint[1], f))
                return st2
            else :
                intersections[st2] = [interval for interval in stands_intervals[st2] if not  intervals_not_intersects(interval, curint)]
    if good_stands : 
        best_st2 = max([ (v,k) for (k,v) in good_stands.items()])[1]
        if wide :
            stands_intervals[best_st2].append((curint[0], curint[1], f, "w"))
        else :
            stands_intervals[best_st2].append((curint[0], curint[1], f))
        return st2


    print("Cannot find stand for", timetablerow)
    print("flight", f, "wide", wide)
    print("group stand", gst, stands, "wide stands", wide_stands)
    print("interval", curint)
    print("intersections", intersections)
    print("busy", busy)
    for t in range(cur_interval[0], cur_interval[1] +1) :
        print("     busy", t, value(model.stand_quantity[ gst, t ]), "wide busy", value(model.wide_stand_quantity[ gst, t ]))
    for stand in aircraft_stands.itertuples() :
        st2 = stand.Aircraft_Stand
        if stand.Aircraft_Stand == gst and (not wide or st2 in wide_stands) :
            print("stand", st2, stands_intervals[st2])
    exit()

def gen_wide_column(aircraft_stands, asgd, GSTANDS):
    wide=[]
    wide2=[]
    asi={}
    ws1={1:0,2:0,3:0, 4:0, 5:0}
    ws2={1:0,2:0,3:0, 4:0, 5:0}
    terms = {}
    for i in range(aircraft_stands.shape[0]) :
        wide.append(False)
        wide2.append(False)
    stands=[]
    for gst in GSTANDS: 
        for stand in aircraft_stands.itertuples() :
            st = stand.Index
            st = stand.Index
            if stand.Aircraft_Stand == gst :
                stands.append(st)
                asi[st]= gst
                terms[st] = stand.Terminal
    print("stands", sorted(stands))
    print(asi)
    for cst in sorted(stands) :
        if cst==0 or pd.isna(terms[cst-1]) or pd.isna(terms[cst]):#or asgd[cst-1].Terminal :
            wide[cst] = True
            wide2[cst] = True
        elif not wide[(cst-1)] :
            t= int(terms[cst])
            wide[cst]= True
            wide2[cst]= False
            gst=asi[st]
            print(t, cst, asgd[gst])
            ws1[t] += asgd[gst]["t%d"%t]
        else :
            wide[cst]= False
            wide2[cst]= True
            t= int(terms[cst])
            gst=asi[st]
            print(t, cst, asgd[gst])
            ws2[t] += asgd[gst]["t%d" %t]

    for stand in aircraft_stands.itertuples() :
        if not stand.Terminal or pd.isna(stand.Terminal):
            wide[stand.Index] = True
    #print(wide)
    print("ws1", ws1)
    print("ws2", ws2)
    return wide2


def calc_FST(timetable, asg, aircraft_classes,handling_time, handling_rates,  args):
        F_AS_KEYS=[]
        FAS_COST = {}
        FAS_COST1 = {}
        FAS_COST2 = {}
        FAS_COST3 = {}
        FAS_COST4 = {}
        FAS_TIMES = {}
        FAS_TIMES2 = {}
        WIDE={}
        if True:
            for row in timetable.itertuples() :
                print(row, row.flight_AD, row.tt_index)
                flight_fit = False
                ac = get_aircraft_class(row.flight_AC_PAX_capacity_total, aircraft_classes)
                WIDE[row.tt_index] = ac =='Wide_Body'
                if args.case <3  :#  or args.baseline:
                    dt = datetime.datetime.strptime(row.flight_datetime, '%Y-%m-%d %H:%M:%S')
                else :
                    dt = datetime.datetime.strptime(row.flight_datetime, '%d.%m.%Y %H:%M')
                flight_time = dt.hour*60 + dt.minute

                for row2 in asg.itertuples() :
                    fit = True
                    cost = 0
                    cost1 = 0
                    cost2 = 0
                    cost3 = 0
                    cost4 = 0
                    #key = (row.tt_index, row2.group_index)
                    key = (row.tt_index, row2.Aircraft_Stand)
                    start_time = 0
                    use_time = 0
                    if fit and row.Aircraft_Stand and not pd.isna(row.Aircraft_Stand):
                        #print(row)
                        #print(row.Aircraft_Stand)
                        #print(row2.Index)
                        if True:
                            #fit = row.Aircraft_Stand  == row2.group_index
                            fit = row.Aircraft_Stand  == row2.Aircraft_Stand  
                    if fit: 
                        print("row2", row2.Aircraft_Stand)

                    if fit :
                        fit_teletrap = row2.Terminal and not pd.isna(row2.Terminal)
                        fit_teletrap = fit_teletrap and  row2.Terminal == row.flight_terminal_N

                        if args.baseline : 
                            if row.flight_AD=='A':
                                fit_teletrap = fit_teletrap and row.flight_ID == row2.JetBridge_on_Arrival
                            else :
                                fit_teletrap = fit_teletrap and row.flight_ID == row2.JetBridge_on_Departure
                        if  fit_teletrap : 
                            #print("if1", fit, row.flight_ID)

                            if not args.baseline : 
                                if row.flight_AD=='A':
                                    fit_teletrap = fit_teletrap and row.flight_ID == row2.JetBridge_on_Arrival
                                else :
                                    fit_teletrap = fit_teletrap and row.flight_ID == row2.JetBridge_on_Departure
                                fit = fit and fit_teletrap
                            

                            #print("if2", fit, fit_teletrap)
                            if fit and  row2.Terminal != row.flight_terminal_N: 
                                print("wrong terminal")


                            #print("if3", fit)
                            if not args.baseline and WIDE[row.tt_index] :
                                if fit and not row2.Wide :
                                    print("removed fit because of wide", row.tt_index, row2.Index)
                                fit = fit and row2.Wide
                            #print("if4", fit)
                            if args.wide_algo == "only_wide" and row2.Terminal and row2.Wide and not WIDE[row.tt_index] :
                                print("only wide")
                                fit = False
                            #print("if5", fit)
                            if fit :

                                use_time =handling_time[ac]['JetBridge_Handling_Time']
                                #print(row2)
                                cost1 = handling_time[ac]['JetBridge_Handling_Time'] * handling_rates['JetBridge_Aircraft_Stand_Cost_per_Minute']['Value'] + 0.00
                                #print("cost1", cost1)
                                cost += cost1
                        else :
                            #print("else buses")
                            if row2.Terminal and not pd.isna(row2.Terminal):
                                cost1 = handling_time[ac]['Away_Handling_Time'] * handling_rates['JetBridge_Aircraft_Stand_Cost_per_Minute']['Value'] + 0.00
                            else :
                                cost3 = handling_time[ac]['Away_Handling_Time'] * handling_rates['Away_Aircraft_Stand_Cost_per_Minute']['Value']
                            use_time =handling_time[ac]['Away_Handling_Time']
                            n_buses = math.ceil(row.flight_PAX/80.0)
                            bus_minutes =0 
                            if row.flight_terminal_N == 1:
                                bus_minutes = row2.t1
                            elif row.flight_terminal_N == 2:
                                bus_minutes = row2.t2
                            elif row.flight_terminal_N == 3:
                                bus_minutes = row2.t3
                            elif row.flight_terminal_N == 4:
                                bus_minutes = row2.t4
                            elif row.flight_terminal_N == 5:
                                bus_minutes = row2.t5
                            else :
                                print("didt found terminal", row.flight_terminal)
                                exit()
                            cost4 = n_buses * bus_minutes * handling_rates['Bus_Cost_per_Minute']['Value']
                            #print("cost3 cost4", cost3, cost4)
                            cost += cost1 + cost3 + cost4
                            
                            if args.cut_long_minutes and  bus_minutes >=args.cut_long_minutes:
                                print("remove long minutes bus")
                                fit = False

                    #print("fit4", fit)
                    if fit : 
                        if row.flight_AD=='A':
                            start_time = flight_time + row2.Taxiing_Time
                        elif row.flight_AD=='D':
                            start_time = flight_time - row2.Taxiing_Time - use_time



                        cost2 = row2.Taxiing_Time * handling_rates['Aircraft_Taxiing_Cost_per_Minute']['Value']
                        cost += cost2
                        #print("cost2", cost2)
                        F_AS_KEYS.append (key)
                        FAS_COST[key] = int(cost)
                        FAS_COST1[key] = int(cost1)
                        FAS_COST2[key] = int(cost2)
                        FAS_COST3[key] = int(cost3)
                        FAS_COST4[key] = int(cost4)
                        print("add key", key, "costs", cost1, cost2, cost3, cost4) # , row ,row2)

                        FAS_TIMES[key] =(int( math.floor(start_time/args.round_minutes)), int( math.floor((start_time + use_time)/args.round_minutes)) )
                        FAS_TIMES2[key] =(int( math.floor(start_time)), int( math.floor((start_time + use_time))) )
                        print ("fas times", FAS_TIMES[key])
                        flight_fit = True
                if not flight_fit:
                    print("Error, totaly remoed flight", row)
                    exit()

        print("finish FST")
        return (F_AS_KEYS, FAS_COST, FAS_COST1, FAS_COST2, FAS_COST3, FAS_COST4, FAS_TIMES, FAS_TIMES2, WIDE)
def main():

    parser = make_parser()
    args = parser.parse_args()
    start= time.time()
    
    if args.case <3 :
        timetable_main = pd.read_csv("data/Timetable_Public.csv", sep=",").reset_index()
    else :
        if args.baseline : 

            timetable_main = pd.read_csv("case3/Timetable_baseline.csv", sep=",").reset_index()
        else : 
            timetable_main = pd.read_csv("case3/Timetable_Private.csv", sep=",").reset_index()
    timetable_main = timetable_main.rename(columns={'flight_terminal_#': 'flight_terminal_N'})

    tt_index = list(range(timetable_main.shape[0]))
    timetable_main = timetable_main.assign(tt_index = tt_index) #.set_index('tt_index')
    print("tmc", timetable_main.columns)
    result_st_column =  timetable_main.Aircraft_Stand 
    result_cost1_column = np.zeros(timetable_main.shape[0])
    result_cost2_column = np.zeros(timetable_main.shape[0])
    result_cost3_column = np.zeros(timetable_main.shape[0])
    result_cost4_column = np.zeros(timetable_main.shape[0])
    result_cost_column = np.zeros(timetable_main.shape[0])
    result_interval1_column = np.zeros(timetable_main.shape[0])
    result_interval2_column = np.zeros(timetable_main.shape[0])

    
    #print(timetable)
    handling_time = pd.read_csv("data/Handling_Time_Public.csv", sep=",")
    #print(handling_time)
    handling_time= handling_time.set_index('Aircraft_Class').T.to_dict()
    if args.case == 1 :
        handling_rates = pd.read_csv("data/Handling_Rates_Public.csv", sep=",")
    elif args.case == 3: 
        handling_rates = pd.read_csv("case3/Handling_Rates_Private.csv", sep=",")
    else : 
        handling_rates = pd.read_csv("data/Handling_Rates_SVO_Private.csv", sep=",")
    handling_rates= handling_rates.set_index('Name').T.to_dict()
    #print(handling_time, handling_rates)

    aircraft_classes= pd.read_csv("data/AirCraftClasses_Public.csv", sep=",")
    #aircraft_classes= pd.read_csv("data/AirCraftClasses_Private.csv", sep=",")
    #print(aircraft_classes)
    if args.case == 1 :
        aircraft_stands= pd.read_csv("data/Aircraft_Stands_Public.csv", sep=";")
    elif args.case == 3: 
        aircraft_stands= pd.read_csv("case3/Aircraft_Stands_Private.csv", sep=",")
    else :
        aircraft_stands= pd.read_csv("data/Aircraft_Stands_Private.csv", sep=",")
        aircraft_stands_old= aircraft_stands

    aircraft_stands = aircraft_stands.rename(columns={'1': 't1'})
    aircraft_stands = aircraft_stands.rename(columns={'2': 't2'})
    aircraft_stands = aircraft_stands.rename(columns={'3': 't3'})
    aircraft_stands = aircraft_stands.rename(columns={'4': 't4'})
    aircraft_stands = aircraft_stands.rename(columns={'5': 't5'})

    #aircraft_stands = aircraft_stands[0:200]

    if False :
        maxBusTime = 15
        maxBusTime2 = 60
        aircraft_stands.t1 = aircraft_stands.t1.where(aircraft_stands.t1 <=maxBusTime, maxBusTime2)
        aircraft_stands.t2 = aircraft_stands.t2.where(aircraft_stands.t2 <=maxBusTime, maxBusTime2)
        aircraft_stands.t3 = aircraft_stands.t3.where(aircraft_stands.t3 <=maxBusTime, maxBusTime2)
        aircraft_stands.t4 = aircraft_stands.t4.where(aircraft_stands.t4 <=maxBusTime, maxBusTime2)
        aircraft_stands.t5 = aircraft_stands.t5.where(aircraft_stands.t5 <=maxBusTime, maxBusTime2)
    #if args.baseline : ### fix Aircraft_Stand
    #    print(aircraft_stands.shape)
    #    aircraft_stands.Aircraft_Stand = list(range(aircraft_stands.shape[0]))
    #print(aircraft_stands.Aircraft_Stand)

    #print(get_aircraft_class( 250, aircraft_classes))
    #print(timetable.size)
    #print(aircraft_stands.size)


    expr = 0
    asf = list(aircraft_stands.columns[1:])
    

    if False : 
        if args.case ==1 :
            aircraft_stands['group_index_old'] = aircraft_stands.groupby(by=asf, dropna = False).ngroup()
        else :
            aircraft_stands['group_index_old'] = aircraft_stands.Aircraft_Stand
        

        #print(asg)
        #print(aircraft_stands.group_index_old)
        GSTANDS = list(set(aircraft_stands.group_index_old))
        wide_column = gen_wide_column(aircraft_stands, GSTANDS)
        STANDS = list(range(aircraft_stands.shape[0]))
        #print(STANDS)
        print(GSTANDS)
        aircraft_stands= aircraft_stands.assign(Wide = wide_column)

        asf = list(aircraft_stands.columns[1:])
        aircraft_stands['group_index'] = aircraft_stands.groupby(by=asf, dropna = False).ngroup()

        asg= aircraft_stands.groupby(by=list(asf +['group_index']), dropna=False).count().add_suffix('_Count').reset_index()
        asgd = asg.set_index('group_index').T.to_dict()
    

    ### no groups
    GSTANDS = list(set(aircraft_stands.Aircraft_Stand))
    STANDS = list(range(aircraft_stands.shape[0]))
    asg = aircraft_stands
    asgd = asg.set_index('Aircraft_Stand').T.to_dict()
    wide_column = gen_wide_column(aircraft_stands, asgd, GSTANDS)
    aircraft_stands= aircraft_stands.assign(Wide = wide_column)

    asg = aircraft_stands
    asgd = asg.set_index('Aircraft_Stand').T.to_dict()

    GSTANDS = list(set(aircraft_stands.Aircraft_Stand))
    #print("asgd", asgd)
    #print("asg", asg)
    #asg.to_csv("asg.csv", sep=";")
    #exit()
    #aircraft_stands.to_csv("as.csv", sep=";")
    print(aircraft_stands.columns)
    
    last_moment_column = np.zeros(timetable_main.shape[0])
    (F_AS_KEYS, FAS_COST, FAS_COST1, FAS_COST2, FAS_COST3, FAS_COST4, FAS_TIMES, FAS_TIMES2, WIDE) = calc_FST(timetable_main, asg, aircraft_classes,handling_time, handling_rates,  args)

    if args.backward: 
        for (f,st) in FAS_TIMES:
            last_moment_column[f] = max( last_moment_column[f], - FAS_TIMES[f, st][0])
    else :

        for (f,st) in FAS_TIMES:
            last_moment_column[f] = max( last_moment_column[f], FAS_TIMES[f, st][1])

    timetable_main = timetable_main.assign(last_moment = last_moment_column).sort_values(by= 'last_moment')
    #timetable_main.to_csv("tt_sorted.csv", sep=";")
    #print("savd sorted")
    firststep =0
    if args.wide_algo == "wide_first":
        firststep = -1
    steps= args.steps
    if args.by_terminal :
        steps = args.by_terminal*5

    for step in range(firststep, steps):
    #for step in range(1):

        last_step = step == args.steps -1
        l= timetable_main.shape[0]
        ttn=[1,2,3,5]
        if args.by_terminal :
            tstep=step//args.by_terminal
            istep = step%args.by_terminal
            if tstep < 4 : 

                print(timetable_main.shape, tstep, istep)

                timetable = timetable_main[timetable_main['flight_terminal_N'] == ttn[tstep]] 
                l= timetable.shape[0]
                ttstart = int (istep * l/args.by_terminal)
                ttstart= 0 
                ttfinish = min( int ((istep+1) * (l)/args.by_terminal), l)

                timetable = timetable.sort_values(by='flight_datetime')[ttstart:ttfinish].reset_index()
                print(timetable_main.shape)
                print(timetable.shape)
            else :
                timetable = timetable_main

                l= timetable.shape[0]
                #ttstart = int (istep * l/args.by_terminal)
                ttstart= 0 
                ttfinish = min( int ((istep+1) * (l)/args.by_terminal), l)

                timetable = timetable.sort_values(by='flight_datetime')[ttstart:ttfinish].reset_index()
            print("tt shape", step, tstep, istep, timetable.shape)
        else  :
            if step >=0 :
                ttstart = int (step * (l-args.initial_size)/args.steps)
                ttstart= 0 
                ttfinish = min( args.initial_size + int ((step+1) * (l- args.initial_size)/args.steps), l)

                print("tt start finish", ttstart, ttfinish, l)
                
                #timetable = timetable_main.sort_values(by='flight_datetime')[int (step * l/args.steps):int( (step+1)* l /args.steps)].reset_index()
                timetable = timetable_main.sort_values(by='flight_datetime')[ttstart:ttfinish].reset_index()
            else : 
                timetable = timetable_main[timetable_main['flight_AC_PAX_capacity_total'] > 220] 

        print("data loaded")

        (F_AS_KEYS, FAS_COST, FAS_COST1, FAS_COST2, FAS_COST3, FAS_COST4, FAS_TIMES, FAS_TIMES2, WIDE) = calc_FST(timetable, asg, aircraft_classes, handling_time, handling_rates, args)
        print("len F_AS_KEYS", len (F_AS_KEYS))
        a=list (set([t[0] for t in FAS_TIMES.values()] + [t[1] for t in FAS_TIMES.values()] ))

        tmin = min([t[0] for t in FAS_TIMES.values()]) - 20
        tmax = max([t[1] for t in FAS_TIMES.values()]) +1
        #print(a)
        print(len(a))
        #print("FAS_TIMES len", len(list(set(FAS_TIMES.values()[0]).union(set(FAS_TIMES.values()[1])) )) )
        print("tmin tmax", tmin, tmax)
        #exit()

        GSTANDS = list(set([k[1] for k in F_AS_KEYS]))
        TIMES = [x for x in range(tmin, tmax+1)]
        #print("GSTANDS", GSTANDS)
        #print("TIMES", TIMES)
        print("len FAS KEYS", len(F_AS_KEYS))
        #exit()

        model = ConcreteModel()
        model.flight2stand = Var(F_AS_KEYS, domain=Boolean)
        print("made flightstand")

        model.Obj = Objective(expr = sum(FAS_COST[key]* model.flight2stand[key] for key in F_AS_KEYS))


        model.cons1 = ConstraintList()
        model.cons2 = ConstraintList()
        model.cons3 = ConstraintList()
        model.cons4 = ConstraintList()
        expr = {}
        for ik, (f, gst) in enumerate(F_AS_KEYS) :
            #if ik%10000 == 0 :
            #    print("ik", ik, f, gst, time.time() - start)
            if f not in expr: 
                expr[f] = 0
            expr[f] +=model.flight2stand[ f, gst]
        for f in expr: 
            #print(len(expr[f]), expr[f])
            model.cons1.add (expr[f] == 1)


        if False : ## without groups
            for (i1, (f1, st1)) in enumerate(F_AS_KEYS) :
                #if i1%100 == 0 :
                #    print("without groups", i1)
                for (i2, (f2, st2)) in enumerate(F_AS_KEYS) :
                    if i2> i1 :
                        intersect = True
                        if FAS_TIMES[f1, st1][0] > FAS_TIMES[f2, st2][1]: 
                            intersect = False
                        if FAS_TIMES[f1, st1][1] < FAS_TIMES[f2, st2][0]: 
                            intersect = False
                        if intersect :
                            model.cons3.add(model.flight2stand[ f1, st1] + model.flight2stand[ f2, st2] <=1)






        if True: ### old quantity for groupped
            model.stand_quantity = Var (GSTANDS, TIMES, domain=NonNegativeReals)
            model.wide_stand_quantity = Var (GSTANDS, TIMES, domain=NonNegativeReals)
            quantity_updates={}
            wide_quantity_updates={}
            for st in GSTANDS:
                for t in TIMES :
                    quantity_updates[st, t] = 0
                    wide_quantity_updates[st, t] = 0
            for ik, key in enumerate(F_AS_KEYS) :
                if ik%1000 == 0 :
                    print("ik", ik, key)

                quantity_updates[key[1], FAS_TIMES[key][0]] += model.flight2stand[key]
                quantity_updates[key[1], FAS_TIMES[key][1] +1] -= model.flight2stand[key]
                if WIDE[key[0]] :
                    wide_quantity_updates[key[1], FAS_TIMES[key][0]] += model.flight2stand[key]
                    wide_quantity_updates[key[1], FAS_TIMES[key][1]+1] -= model.flight2stand[key]

            for (st, t) in quantity_updates :
                if t == tmin : 
                    model.cons2.add (  model.stand_quantity[ st, t] == 0) 
                    model.cons2.add (  model.wide_stand_quantity[ st, t] == 0) 
                else :
                    model.cons2.add (  model.stand_quantity[ st, t] == model.stand_quantity[st, t -1] + quantity_updates[ st, t]) 
                    model.cons2.add (  model.wide_stand_quantity[ st, t] == model.wide_stand_quantity[st, t -1] + wide_quantity_updates[ st, t]) 
                if not ( st in GSTANDS and  t in TIMES):
                        print( st in GSTANDS, t in TIMES)
                        print(st, GSTANDS)
                        print(t, TIMES)
                model.cons3.add ( model.stand_quantity[ st, t] <= 1 ) #asgd[st]['Aircraft_Stand_Count'])
                model.cons4.add ( model.wide_stand_quantity[ st, t] <= 1) # math.ceil(asgd[st]['Aircraft_Stand_Count']/2))
        print("model generated", step, time.time() - start)
        model.write('test.lp')
        solver = SolverFactory(args.solver)
        solver.options['MipGap'] = 0.001
        if args.solver == "cbc" :
            solver.options['barr'] = ''
        
        results = solver.solve(model, tee=True)
        print("model solved")
        if len(results.solution) > 0:
            model.solutions.load_from(results)
        condition = results.solver.termination_condition
        if condition == TerminationCondition.infeasible:
            print("infeasible")
            exit()

        stands_intervals={}


        for st in STANDS :
            stands_intervals[st] = []
        flight2stand= {}

        for k in F_AS_KEYS:
            if value(model.flight2stand[k]) >0 :
                (f, gst) = k 
                flight2stand[f] = gst

        ### ?????? ???????? ?????????? ?????? ?????????????????? ??????????????????????. ???????????? ?????? ??????
        #FLIGHTS = sorted([row.tt_index for row in timetable.itertuples()], key = lambda f: FAS_TIMES[ f, flight2stand[f] ])
        FLIGHTS = [row.tt_index for row in timetable_main.itertuples() ]
        print("FLIGHTS main", len(FLIGHTS), FLIGHTS)
        f2row = { row.tt_index: row for row in timetable_main.itertuples()}
        print("f2row main", f2row)

        FLIGHTS = [row.tt_index for row in timetable.itertuples()]
        print("FLIGHTS", len(FLIGHTS), FLIGHTS)
        f2row = { row.tt_index: row for row in timetable.itertuples()}
        print("f2row", f2row)

        if False: 
            for fi, f in enumerate(FLIGHTS):
                row = f2row[f]
                gst = flight2stand[f]
                result_cost1_column[f] = FAS_COST1[ f,gst] 
                result_cost2_column[f] = FAS_COST2[ f,gst] 
                result_cost3_column[f] = FAS_COST3[ f,gst] 
                result_cost4_column[f] = FAS_COST4[ f,gst] 
                cur_interval = FAS_TIMES[f, gst]
                busy1 = value(model.wide_stand_quantity[ gst, cur_interval[0] ])
                busy2 = value(model.wide_stand_quantity[ gst, cur_interval[1] ])
                busy3 = math.ceil(asgd[gst]['Aircraft_Stand_Count']/2) 
                busy4 = asgd[gst]['Aircraft_Stand_Count']

                if True or WIDE[f] :
                    st2 = find_stand(cur_interval , aircraft_stands, f, gst, stands_intervals, row, wide=WIDE[f], busy=(busy1, busy2, busy3, busy4))
                    result_st_column[f] = st2


        if True: 
            for f in FLIGHTS :
                row = f2row[f]
                gst = flight2stand[f]
                print("f gst", f, gst)
                result_cost1_column[f] = FAS_COST1[ f,gst] 
                result_cost2_column[f] = FAS_COST2[ f,gst] 
                result_cost3_column[f] = FAS_COST3[ f,gst] 
                result_cost4_column[f] = FAS_COST4[ f,gst] 
                result_cost_column[f] = FAS_COST[ f,gst] 
                result_interval1_column[f] = FAS_TIMES[ f,gst] [0]
                result_interval2_column[f] = FAS_TIMES[ f,gst] [1]
                cur_interval = FAS_TIMES[f, gst]
                busy1 = value(model.stand_quantity[ gst, cur_interval[0] ])
                busy2 = value(model.stand_quantity[ gst, cur_interval[1] ])
                busy3 = 1 #asgd[gst]['Aircraft_Stand_Count']

                if True or not WIDE[f] :
                    st2 = find_stand(cur_interval, aircraft_stands, f, gst, stands_intervals, row, wide=WIDE[f], busy=(busy1, busy2, busy3), model= model) #WIDE[f])
                    #st2 = asgd[gst]['group_index_old']
                    st2 = gst
                    if last_step :
                        result_st_column[f] = st2 
                    elif not args.by_terminal and FAS_TIMES[ f, gst][1] < tmax - args.soft_period:
                        result_st_column[f] = st2 
                    elif args.by_terminal :
                        tstep=step//args.by_terminal
                        istep = step%args.by_terminal
                        if tstep <4 and istep == args.by_terminal -1  and not pd.isna(asgd[st2]["Terminal"]) :
                        #if istep == args.by_terminal -1  or (FAS_TIMES[ f, gst][1] < tmax - args.soft_period and not pd.isna(asgd[st2]["Terminal"]) ):
                            result_st_column[f] = st2 

                        elif tstep < 4 and istep < args.by_terminal -1  or (FAS_TIMES[ f, gst][1] < tmax - args.soft_period and not pd.isna(asgd[st2]["Terminal"]) ):
                            result_st_column[f] = st2 
                        elif tstep == 4 and istep < args.by_terminal -1  or (FAS_TIMES[ f, gst][1] < tmax - args.soft_period ):
                            result_st_column[f] = st2 


                timetable_main.Aircraft_Stand = result_st_column


                print(len(result_st_column))
    #print(timetable.shape[0])
    #print(timetable.Aircraft_Stand)
    timetable_main.Aircraft_Stand = result_st_column
    timetable_main= timetable_main.assign(Cost1 = result_cost1_column).assign(Cost2 = result_cost2_column).assign(Cost3 = result_cost3_column).assign(Cost4 = result_cost4_column).assign(Cost = result_cost_column).assign(Start = result_interval1_column).assign(Finish = result_interval2_column).assign(Aircraft_stands = result_st_column)
    #print(timetable.columns)
    timetable_main = timetable_main.astype({"Cost1": int, "Cost2": int, "Cost3" : int, "Cost4" : int,"Cost": int, "Start": int, "Finish": int})
    timetable_main = timetable_main.rename(columns={'flight_terminal_N': 'flight_terminal_#'})
    total_cost = int(sum(result_cost_column))
    fname= "result-%d.csv" % total_cost
    timetable_main.to_csv(fname, sep=",")

    f=open("stand_usage_report.txt","w")
    for st in STANDS:
        f.write("Stand %s :"% (st))
        for interval in sorted(stands_intervals[st]):
            f.write("    %d %d (%d)" % interval[0:3])
        f.write("\n")
    f.close()

    print("results checked and saved", time.time()- start)
    print("total cost", sum(result_cost_column))
    print("file name ", fname)
    print("args", args)
                        



main()
