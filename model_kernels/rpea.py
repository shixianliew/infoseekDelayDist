# attempted implementation of the iigaya et al (2016) model.

import numpy as np
import math
import pandas as pd
#from numpy.random import choice #pretty much R's sample function
from numpy.random import binomial #pretty much R's sample function
import scipy.optimize as op
import scipy.stats as ss
import matplotlib.pyplot as plt
import random
import utilities as utl

#Define some useful descriptive properties of the model.
def getDetails():
    return({'name':'m1',
            'authname':'Iigaya',
            'parmnames':['gamma','nu','eta','boost','alpha','beta']
    })

def parmxform(parms,direction=1):
    #Iigaya model log transforms all parms except alpha
    alphaIndex = 4
    notAlpha = np.array(np.zeros(len(parms)) + 1,dtype='bool')
    notAlpha[alphaIndex] = False
    isAlpha = np.logical_not(notAlpha)
    #Ensure parms is a numpy array
    parms = np.array(parms)
    #Convert
    parms[notAlpha] = utl.log_scale(parms[notAlpha],direction=direction)
    parms[isAlpha] = utl.logit_scale(parms[isAlpha],direction=direction)
    
    return(parms)

# time discounted reward
def discountedReward(reward, delay, gamma):
    return(reward * math.exp(-gamma * delay))


# time-discounted anticipatory reward integrated over time
def anticipatoryReward( reward, delay, gamma, nu):
    #Note that nu here is lambda in the grant
    if nu == gamma:
        return(0.)
    return((reward / (nu - gamma)) * (math.exp(-gamma * delay) - math.exp(-nu * delay)))

#Streamlined function
# time-discounted anticipatory reward integrated over time
def anticipatoryRewardSlim(reward,gamma, eGamDel, nu, eNuDel):
    #Note that nu here is lambda in the grant
    if nu == gamma:
        return(0.)
    
    return((reward / (nu - gamma)) * (eGamDel - eNuDel)) #( (eGamDel - eNuDel) / (nu - gamma) )
    #return( (eGamDel - math.exp(-nu * delay)) / (nu - gamma) )

# function computing time discounted and anticipatory
# inclusive value of the reward predicting cue is a
# function of the subjective parameters, the objective
# reward, and the time delay between the cue and the
# reward
def cueValue(reward,gamma,eGamDel,eta,nu,eNuDel):
    Q_reward = reward * eGamDel #discountedReward(reward, delay, gamma)
    Q_anticipation = anticipatoryRewardSlim(reward,gamma, eGamDel, nu,eNuDel)#anticipatoryReward(reward, delay, gamma, nu)    
    Q = (eta * Q_anticipation + Q_reward) 
    return(Q)


# reward prediction error
def RPE(reward, delay, gamma, eta0, nu, q, c=1):#(Va,Vr,eta0,q,c):#(Va,e0VaVr,cVa,invNq):#(reward, gamma, eGamDel,eta0, nu, eNuDel, q, c=1):
    Va = anticipatoryReward(reward, delay, gamma, nu)
    Vr = discountedReward(reward, delay, gamma)
    delta = (1-q) * (eta0 * Va + Vr) / (1- (1-q) * c * Va)


    #The problem here is that when the denominator is vvvsmall, delta quickly gets huge.
    # Va = anticipatoryRewardSlim(reward,gamma,eGamDel,nu, eNuDel)
    # Vr = reward * eGamDel #discountedReward(reward, delay, gamma)
    # q1 = (1.-q)
    # return((q1) * (eta0 * Va + Vr) / (1.- (q1) * c * Va))

    #invNq = 1./(1.-q) #inverse negative q
    
    #return((e0VaVr) / (invNq - cVa))    
    #return((eta0 * Va + Vr) / (1./(c*(1.-q)) - Va)) #include the c boost from updateQ
    return(delta)



def updateQOld(Qx, alpha, reward, delay, gamma, eGamDel,eta0, nu, eNuDel,q=.5,c=1):
    # eta = eta0 + c * abs(RPE(reward, delay, gamma, eGamDel,eta0, nu, q=.5))
    # V = cueValue(reward, delay, gamma, eGamDel,eta, nu)

    Qa = anticipatoryReward(reward,delay,gamma,nu)#(reward,gamma,eGamDel,nu, eNuDel)
    Qr = reward * eGamDel
    
    #eta = eta0 + abs(RPE(Qa,Qr,eta0,q=.5))#eta0 + c * abs(RPE(reward, gamma, eGamDel,eta0, nu, eNuDel, q=.5))
    eta = eta0 + c * abs(RPE(reward, delay,gamma,eta0, nu, q=.5, c=c))
    V =  eta * Qa + Qr #cueValue(reward, gamma, eGamDel,eta, nu, eNuDel)
    Qx = Qx + alpha * (V - Qx)
    return(Qx)

def updateQ(Qx, V, alpha):
    # eta = eta0 + c * abs(RPE(reward, delay, gamma, eGamDel,eta0, nu, q=.5))
    # V = cueValue(reward, delay, gamma, eGamDel,eta, nu)


    #eta = eta0 + abs(RPE(Va,Vr,eta0,q=.5))#eta0 + c * abs(RPE(reward, gamma, eGamDel,eta0, nu, eNuDel, q=.5))
    #eta = eta0 + c * abs(RPE)#eta0 + c * abs(RPE(reward, gamma, eGamDel,eta0, nu, eNuDel, q=.5))
    #V =  eta * Va + Vr #cueValue(reward, gamma, eGamDel,eta, nu, eNuDel)
    #Qx = Qx + alpha * (V - Qx)
    return(Qx + alpha * (V - Qx))


# probability of choosing informative option
def choice(Qinfo,Qno,beta=1.0):#(Qinfo, Qno, sigma=1):
    #dQ = Qno - Qinfo #Swapped order of Qinfo and Qno to reduce number of operations (see the removal of negative in front of dQ in final equation)
    #pi = 1/(1 + np.exp(-dQ/sigma))
    
    try:
        pi = 1./(1. + math.exp(beta * (Qno-Qinfo))) #Removed sigma to speed up code a little
    except OverflowError:
        pi = 0.
    
    return(pi)

def initialState():
    states = dict(
        Qinfo = 0.,
        Qno = 0.,
        choice = [],
        m_choice = 0.,
        #QinfoL = [], #Temp list to track hist of Qinfo
        #QnoL = [], #Temp list to track hist of Qnon
        )
    return(states)

    

def oneTrial(states,learner,task,trial):
    Qinfo = states['Qinfo']
    Qno = states['Qno']
    
    #states['QinfoL'] += [Qinfo] #Temp list to track hist of Qinfo
    #states['QnoL'] += [Qno] #Temp list to track hist of Qno

    onechoice = choice(Qinfo, Qno,learner['beta'])
    getchoice = random.random()#random.random() is faster than random.uniform(0,1)
    getreward = random.random()#random.uniform(0,1)
    task['pYA'] = 1- task['pXA']
    if getchoice <= onechoice: # choose informative
        if getreward <= task['pXA']: # will a reward appear?
            # states['Qinfo']  = Qinfo + learner['alpha'] * (learner['V_IA'] - Qinfo) #updateQ(Qinfo, learner['QaIRA'], learner['QrIRA'], learner['eta'], learner['alpha'], c = learner['boost'])
            states['Qinfo'] =  learner['alphaN'] * Qinfo + learner['V_IA']
        else:
            states['Qinfo'] = learner['alphaN'] * Qinfo + learner['V_IB'] #Qinfo + learner['alpha'] * (learner['V_IB'] - Qinfo)
    else: # choose uninformative
        if getreward <= task['pYA']: # will a reward appear?            
            states['Qno'] = learner['alphaN'] * Qno + learner['V_NA'] #Qno + learner['alpha'] * (learner['V_NA'] - Qno)
        else:
            states['Qno'] = learner['alphaN'] * Qno + learner['V_NB'] #Qno + learner['alpha'] * (learner['V_NB'] - Qno)
    states['choice'][trial] = onechoice #states['choice'].append(onechoice)
    #states['m_choice'] = states['m_choice'] + onechoice/task['ntrials'] #Accumulate the average choice prob
    return(states)

def oneTrialOld(states,learner,task):
    Qinfo = states['Qinfo']
    Qno = states['Qno']
    
    onechoice = choice(Qinfo, Qno)
    getchoice = random.uniform(0,1)
    getreward = random.uniform(0,1)
    
    if getchoice <= onechoice: # choose informative
        if getreward <= task['pXA']: # will a reward appear?
            states['Qinfo'] = updateQOld(Qinfo, learner['alpha'], task['RA'], task['delay'], learner['gamma'],learner['eGamDel'], learner['eta'], learner['nu'],learner['eNuDel'], c = learner['boost'])
        else:
            states['Qinfo'] = updateQOld(Qinfo, learner['alpha'], task['RB'], task['delay'], learner['gamma'],learner['eGamDel'], learner['eta'], learner['nu'],learner['eNuDel'], c = learner['boost'])
    else: # choose uninformative
        #eta and nu are = 0
        if getreward <= task['pYA']: # will a reward appear?
            states['Qno'] = updateQOld(Qno, learner['alpha'], task['RA'], task['delay'], learner['gamma'], learner['eGamDel'], 0, 0, 1, c = learner['boost'])
        else:
            states['Qno'] = updateQOld(Qno, learner['alpha'], task['RB'], task['delay'], learner['gamma'], learner['eGamDel'], 0, 0, 1, c = learner['boost'])
    states['choice'] += [onechoice]
    return(states)


def oneTrialwithChoice(states,learner,task):
    Qinfo = states['Qinfo']
    Qno = states['Qno']
    
    onechoiceProb = choice(Qinfo, Qno) #choiceprob
    tr = len(states['choice']) #current trial
    states['choice'] += [onechoiceProb]

    #     getchoice = random.uniform(0,1)
    #     getreward = random.uniform(0,1)

    onechoice = states['choicemade'][tr]==0#getchoice <= onechoiceProb: # choose informative
    if onechoice: # choose informative
        onereward = states['outcome'] == 0 #getreward <= task['pXA'] # will a reward appear?
        if onereward: # will a reward appear?
            states['Qinfo'] = updateQ(Qinfo, learner['alpha'], task['RA'], task['delay'], learner['gamma'], learner['eta'], learner['nu'], c = learner['boost'])
        else:
            states['Qinfo'] = updateQ(Qinfo, learner['alpha'], task['RB'], task['delay'], learner['gamma'], learner['eta'], learner['nu'], c = learner['boost'])
    else: # choose uninformative
        onereward = states['outcome'] == 0 # getreward <= task['pYA'] # will a reward appear?
        if onereward: # will a reward appear?
            states['Qno'] = updateQ(Qno, learner['alpha'], task['RA'], task['delay'], learner['gamma'], 0, 0, c = learner['boost'])
        else:
            states['Qno'] = updateQ(Qno, learner['alpha'], task['RB'], task['delay'], learner['gamma'], 0, 0, c = learner['boost'])

    return(states)

def minthis(freeparms,data,task,fixVal=None,ivs=['delay'],minfun='negll',randseed=None,returnPreds = False):
    #Script to run some function to minimise
    #Free parms are in the order:
    #[gamma,nu,eta,boost,alpha,beta]
    #Reverse-transform parms
    freeparms = parmxform(freeparms,direction=-1)
    
    #Insert freeparm values into vector of all parms
    if fixVal==None:
        fixVal = [np.nan] * len(freeparms)
    fixVal = np.array(fixVal)
    freeIdx = np.isnan(fixVal)
    fixVal[freeIdx] = freeparms
    parms = fixVal

    # unpack learner parameters
    learner = dict(
        gamma = parms[0], #1# intertemporal discount parameter
        nu = parms[1],#.5, # anticipation gain parameter
        eta = parms[2],#3.,
        boost = parms[3],#3.,
        alpha = parms[4],#.3 # learning rate
    )
    #Also check for beta parm (placed in conditional for legacy reasons, some old code won't supply 6th parm)
    if len(parms)>5:
           learner['beta'] = parms[5]
    else:
        learner['beta'] = 1.0

    if not randseed is None:
        random.seed(randseed)

    # ivSet = iv + 'Set'
    # preds = np.zeros(len(task[ivSet]))

    #Allow for multiple ivs -- have iv be a list of strings
    ivSet = []
    for iv in ivs:
        ivSet += [iv + 'Set']
    #delays = task['delays']
    nLevels = len(task[ivSet[0]])
    preds = np.zeros(nLevels)
    
    #for li,level in enumerate(task[ivSet]):
    for li in range(nLevels):
        pi = [0]*task['nrep'] #Preallocate
        
#        task[iv] = level
        for iv in ivs:
            task[iv] = task[iv + 'Set'][li] 
                
        #Precalculate some expensive constants
        learner['eNuDel'] = math.exp(-learner['nu'] * task['delay'])
        learner['eGamDel'] = math.exp(-learner['gamma'] * task['delay'])

        # Values of informative choices
        QaIA = anticipatoryRewardSlim(task['RA'],learner['gamma'],learner['eGamDel'],learner['nu'], learner['eNuDel']) #Anticipatory value of a win
        QrIA = task['RA'] * learner['eGamDel'] #Reward value of a win
        QaIB = anticipatoryRewardSlim(task['RB'],learner['gamma'],learner['eGamDel'],learner['nu'], learner['eNuDel']) #Anticipatory value of a loss
        QrIB = task['RB'] * learner['eGamDel'] #Reward value of a loss

        #For noninformative cues the diff is that nu is set to 0 and so eNuDel is 1
        QaNA = anticipatoryRewardSlim(task['RA'],learner['gamma'],learner['eGamDel'],0, 1) #Anticipatory value of a win
        QrNA = QrIA #Reward value of a win
        QaNB = anticipatoryRewardSlim(task['RB'],learner['gamma'],learner['eGamDel'],0, 1) #Anticipatory value of a loss
        QrNB = QrIB  #Reward value of a loss

        #Different possible RPE values
        q = .5
        rpeIA = RPE(task['RA'], task['delay'], learner['gamma'], learner['eta'], learner['nu'], q, learner['boost'])# RPE(QaIA,QrIA,learner['eta'],q,learner['boost'])
        rpeIB = RPE(task['RB'], task['delay'], learner['gamma'], learner['eta'], learner['nu'], q, learner['boost'])
        rpeNA = RPE(task['RA'], task['delay'], learner['gamma'], 0, 0, q, learner['boost'])
        rpeNB = RPE(task['RB'], task['delay'], learner['gamma'], 0, 0, q, learner['boost'])
        #Different possible eta values
        etaIA = learner['eta'] + learner['boost'] * abs(rpeIA)
        etaIB = learner['eta'] + learner['boost'] * abs(rpeIB)
        etaNA = learner['boost'] * abs(rpeNA)
        etaNB = learner['boost'] * abs(rpeNB)
        #Finally the different cue values
        learner['alphaN'] = 1-learner['alpha']
        learner['V_IA'] =  (etaIA * QaIA + QrIA) * learner ['alpha'] #cueValue(reward, gamma, eGamDel,eta, nu, eNuDel)
        learner['V_IB'] =  (etaIB * QaIB + QrIB) * learner ['alpha'] #cueValue(reward, gamma, eGamDel,eta, nu, eNuDel)
        #Shouldnt these be zero?? No anticipatory signal right?
        learner['V_NA'] =  0#(etaNA * QaNA + QrNA) * learner ['alpha'] #cueValue(reward, gamma, eGamDel,eta, nu, eNuDel)
        learner['V_NB'] =  0#(etaNB * QaNB + QrNB) * learner ['alpha'] #cueValue(reward, gamma, eGamDel,eta, nu, eNuDel)
        
        #For debugging 021220, pls remember to remove ##x
        #statei = [] ##
        for r in range(task['nrep']):
            state = initialState()
            state['choice'] = [0]*task['ntrials']
            #statei += [[]] ##
            for i in range(task['ntrials']):
                state = oneTrial(state, learner, task, i)
                #statei[r] += [{'Qinfo':state['Qinfo'],'Qno':state['Qno'],'choice':state['choice'][i]}]##
            #pi += [np.mean(state['choice'])]
            pi[r] = state['choice'] ##m_choice for mean choice accumulated in one_trial

        preds[li] = np.mean(pi)
    if minfun == 'rmsd':
        out = np.sqrt(np.mean((preds-np.array(data))**2))
    elif minfun == 'negll':        
        out = sum(utl.binoNegLL(preds,data['count_inform'],data['count_total']))
    elif minfun == None:
        out = None
        
    if not returnPreds:
        return(out)
    else:
        return(out,preds)#,pi,learner,{'etaIAIB':[etaIA,etaIB],'etaNA':etaNA,'rpeIAIB':[rpeIA,rpeIB],'rpeNA':rpeNA},task,statei)

# def binoNegLL(prob_info,freq_info,freq_total):
#     #Get loglikelihoods based on Bennett 2016 -- binomial rate parameter
#     #Ignore constant coefficient
#     prob_info = np.array(prob_info)
#     freq_info = np.array(freq_info)
#     freq_total = np.array(freq_total)
#     L = prob_info ** freq_info * (1-prob_info)**(freq_total-freq_info)
#     LL = -np.log(L)
#     return(LL)

def tbtnegll(freeparms, data, task):
    #Transform parms
    freeparms = [math.exp(fp) for fp in freeparms]
    learner = dict(
        gamma = freeparms[0], #1# intertemporal discount parameter
        nu = freeparms[1],#.5, # anticipation gain parameter
        eta = freeparms[2],#3.,
        boost = freeparms[3],#3.,
        alpha = freeparms[4]#.3 # learning rate
    )
    nloglike = 0
    for choiceDict in data:
        task['delay'] = choiceDict['delay']
        choices = choiceDict['choices']
        counts = choiceDict['count']
        outcomes = choiceDict['outcomes']
        state = initialState()
        state['choicemade'] = choices
        state['outcome'] = outcomes
        nlls = []
        for choice in choices:
            state = oneTrialwithChoice(state,learner,task)
            if choice == 0: #Choose informtive
                like = state['choice'][-1]
            elif choice == 1: #Choose noninformative
                like = 1-state['choice'][-1]
            nll = -np.log(like)
            nlls += [nll]
        nloglike += np.sum(nlls) * counts
    return(nloglike)
