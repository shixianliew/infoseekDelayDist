
# attempted implementation of the Bennett 2016 uncertainty penalty model
# with MDP structure adapted for secrets task. For structure more appropriate
# for the Bennett card revealing task, check out the BennettModel.ipynb jupyter notebook.
import numpy as np
import itertools
#import modelBennettkernel as mBenn
from scipy import sparse
import pandas as pd
from scipy.stats import binom
import utilities as utl
import pickle
import itertools
import os
import multiprocessing
import time
import random

# Apologies for this very non-object oriented way of doing things...
def getDetails():
    return({'name':'mBenn',
            'authname':'Bennett',
            'parmnames':['gamma','k','beta']
    })
def parmxform(parms,direction=1):
    #Bennett model log transforms all parms
    parms = utl.log_scale(parms,direction=direction)    
    return(parms)


# #free parms
# observer = dict(
#     k = 1,
#     gamma = 1,
#     beta = .15,
# )
# #fixed parms
# task = dict(
#     delay = 4, #Two wait nodes or steps
#     pXA = .5, #pWin
#     RA = 20.,#reward for winning
#     RB = 0.#reward for losing
# )

def HSecrets(p,steps2win,winIdx,losIdx):
    #provided in vector form
    nRows = p.shape[0]
    out = np.zeros(nRows)
    for s in range(nRows):
        pWin = pWLSecrets(p,steps2win,s,winIdx)
        pLose = pWLSecrets(p,steps2win,s,losIdx)
        if not 0 in [pWin,pLose]:
            H = -pWin * np.log2(pWin) - (pLose * np.log2(pLose))
            out[s] = H
    return out

def pWLSecrets(p,steps2target,currIdx,targetIdx):
    #Assume no recursion & no shortcuts to winning
    #This takes super long with large matrices though. Even when sparse.
    steps = steps2target[currIdx]
    out = p[currIdx,targetIdx]
    pBase = p
    for s in range(steps-1):
        p = np.dot(p,pBase)
        out += p[currIdx,targetIdx]
    return out

def HSecretsFast(pWins):
    #provided in vector form
    nRows = len(pWins)
    out = np.zeros(nRows)
    for s in range(nRows):
        pWin = pWins[s]
        pLose = 1-pWin#pWLSecrets(p,steps2win,s,losIdx)
        if not 0 in [pWin,pLose]:
            H = -pWin * np.log2(pWin) - (pLose * np.log2(pLose))
            out[s] = H
    return out


def RPEfun(Q, s, reward, gamma):
    rpe = reward + gamma * max(Q[s+1]) - Q[s]
    return rpe

def updateQ(Q,s,reward,gamma,alpha):
    Qout = Q
    rpe = RPEfun(Q, s, reward, gamma)
    Qout[s] += alpha * rpe

def makeV(observer,task):#pA,rewards,waits,k,gamma):     
    pA = task['pXA']
    rewards = [task['RA'],task['RB']]
    waits = int(task['delay'])
    k = observer['k']
    gamma = observer['gamma']
    #States:
    # Inform, Noninform, CueWin, CueLose, CueAmb, Win, Lose
    #Inform
    #Noninform
    #CueWin
    #CueLose
    #CueAmb
    #Win
    #Lose
    stateNames = ['inform','noninform','cueWin','cueLose','cueAmb']#,'win','lose']
    #Insert waits, one for each cue
    cues = ['Win','Lose','Amb']
    waitStates = {}
    for wait in range(waits):
        for cue in cues:
            waitState = 'wait'+cue+str(wait+1)
            stateNames += [waitState]
            if not cue in waitStates.keys():
                waitStates[cue] = []
            waitStates[cue] += [waitState]
    #Append win and lose states
    stateNames += ['win'] 
    stateNames += ['lose'] 
    # print(stateNames)
    winIdx = stateNames.index('win')
    losIdx = stateNames.index('lose')

    nStates = len(stateNames)

    #Generate transition probs
    pTrans = np.zeros((nStates,nStates))
    pTrans[0,2] = pA    #inform > cueWin
    pTrans[0,3] = 1.-pA #inform > cueLose
    pTrans[1,4] = 1.    #noninform > cueAmb
    steps2win = np.zeros(nStates,dtype=int) #vector of number of steps to take to win (assuming no recursion)
    steps2win[0] = 2+waits #inform
    steps2win[1] = 2+waits #noninform
    steps2win[2] = 1+waits #cueWin
    steps2win[3] = 1+waits #cueLose
    steps2win[4] = 1+waits #cueAmb
    pWins = np.zeros(nStates) #probability of each state eventually winning
    pWins[0] = pA #inform
    pWins[1] = pA #noninform
    pWins[2] = 1. #cueWin
    pWins[3] = 0. #cueLose
    pWins[4] = pA #cueAmb
    cueIdcs = [2,3,4] #Indices of cue states in stateNames
    winIdcs = [1.,0.,.5]
    for ci,cue in enumerate(cues):
        prevIdcs = cueIdcs
        for wait in range(waits):
            waitState = waitStates[cue][wait]
            prevIdx = prevIdcs[ci]#cueIdx+(wait*(waits+1)) #index of previous waitstate (or cue state)
            waitIdx = stateNames.index(waitState)
            pTrans[prevIdx,waitIdx] = 1.    #cueState > waitStateWaitIdx
            #Update prevIdcs
            prevIdcs[ci] += len(cueIdcs)
            steps2win[prevIdcs[ci]] = waits-wait
            pWins[prevIdcs[ci]] = winIdcs[ci]
            
    pTrans[prevIdcs[0],winIdx] = 1.    #cueWin > win
    pTrans[prevIdcs[1],losIdx] = 1.    #cueLose > lose
    pTrans[prevIdcs[2],winIdx] = pA    #cueAmb > win
    pTrans[prevIdcs[2],losIdx] = 1.-pA #cueAmb > lose

    Rvec = np.zeros(nStates) #Reward vector
    Rvec[winIdx] = rewards[0]
    Rvec[losIdx] = rewards[1]

    V = np.zeros(nStates) #V vector
    #Convert pTrans to sparse
#     if task['use_sparse']:
#         pTrans = sparse.csr_matrix(pTrans)
    for i in range(nStates):
        currIdx = nStates - i - 1
#         Vcurr = sum(pTrans[currIdx,:] * (Rvec*np.exp(-gamma*waits) + V * np.exp(-k*HSecrets(pTrans,steps2win,winIdx,losIdx))))
        Vcurr = sum(pTrans[currIdx,:] * (Rvec*np.exp(-gamma*waits) + V * np.exp(-k*HSecretsFast(pWins))))
        if not np.isnan(Vcurr):
            V[currIdx] = Vcurr

    return V

def makeVma(observer,task):#multialternative version #pA,rewards,waits,k,gamma):    
    pA = task['pXA']
    rewards = [task['RA'],task['RB']]
    waits = int(task['delay'])
    k = observer['k']
    gamma = observer['gamma']
    #States:
    # Inform, Midinform, Noninform, CueWin, CueLose, CueAmb, Win, Lose
    #Inform
    #Midinform
    #Noninform
    #CueWin
    #CueLose
    #CueAmb
    #Win
    #Lose
    stateNames = ['inform','midinform','noninform','cueWin','cueLose','cueAmb']#,'win','lose']
    #Insert waits, one for each cue
    cues = ['Win','Lose','Amb']
    waitStates = {}
    for wait in range(waits):
        for cue in cues:
            waitState = 'wait'+cue+str(wait+1)
            stateNames += [waitState]
            if not cue in waitStates.keys():
                waitStates[cue] = []
            waitStates[cue] += [waitState]
    #Append win and lose states
    stateNames += ['win'] 
    stateNames += ['lose'] 
    # print(stateNames)
    winIdx = stateNames.index('win')
    losIdx = stateNames.index('lose')

    nStates = len(stateNames)

    #Generate transition probs
    pTrans = np.zeros((nStates,nStates))
    #Can I not automate this? 191020
    pTrans[0,3] = pA    #inform > cueWin
    pTrans[0,4] = 1.-pA #inform > cueLose
    pTrans[1,3] = pA    #midinform > cueLose
    pTrans[1,4] = pA    #midinform > cueLose
    pTrans[1,5] = 1.    #noninform > cueAmb
    steps2win = np.zeros(nStates,dtype=int) #vector of number of steps to take to win (assuming no recursion)
    steps2win[0] = 2+waits #inform
    steps2win[1] = 2+waits #noninform
    steps2win[2] = 1+waits #cueWin
    steps2win[3] = 1+waits #cueLose
    steps2win[4] = 1+waits #cueAmb
    pWins = np.zeros(nStates) #probability of each state eventually winning
    pWins[0] = pA #inform
    pWins[1] = pA #noninform
    pWins[2] = 1. #cueWin
    pWins[3] = 0. #cueLose
    pWins[4] = pA #cueAmb
    cueIdcs = [2,3,4] #Indices of cue states in stateNames
    winIdcs = [1.,0.,.5]
    for ci,cue in enumerate(cues):
        prevIdcs = cueIdcs
        for wait in range(waits):
            waitState = waitStates[cue][wait]
            prevIdx = prevIdcs[ci]#cueIdx+(wait*(waits+1)) #index of previous waitstate (or cue state)
            waitIdx = stateNames.index(waitState)
            pTrans[prevIdx,waitIdx] = 1.    #cueState > waitStateWaitIdx
            #Update prevIdcs
            prevIdcs[ci] += len(cueIdcs)
            steps2win[prevIdcs[ci]] = waits-wait
            pWins[prevIdcs[ci]] = winIdcs[ci]
            
    pTrans[prevIdcs[0],winIdx] = 1.    #cueWin > win
    pTrans[prevIdcs[1],losIdx] = 1.    #cueLose > lose
    pTrans[prevIdcs[2],winIdx] = pA    #cueAmb > win
    pTrans[prevIdcs[2],losIdx] = 1.-pA #cueAmb > lose

    Rvec = np.zeros(nStates) #Reward vector
    Rvec[winIdx] = rewards[0]
    Rvec[losIdx] = rewards[1]

    V = np.zeros(nStates) #V vector
    #Convert pTrans to sparse
#     if task['use_sparse']:
#         pTrans = sparse.csr_matrix(pTrans)
    for i in range(nStates):
        currIdx = nStates - i - 1
#         Vcurr = sum(pTrans[currIdx,:] * (Rvec*np.exp(-gamma*waits) + V * np.exp(-k*HSecrets(pTrans,steps2win,winIdx,losIdx))))
        Vcurr = sum(pTrans[currIdx,:] * (Rvec*np.exp(-gamma*waits) + V * np.exp(-k*HSecretsFast(pWins))))
        if not np.isnan(Vcurr):
            V[currIdx] = Vcurr

    return V

#Take Q(a) to be the V at second layer (i.e., the inform/noninform)
def choice(Qi,Qn,beta):
    #Luce choice rule
    p_inform = np.exp(beta*Qi) / (np.exp(beta*Qi) + np.exp(beta*Qn))
    return p_inform


def minthis(freeparms,data,task,fixVal,ivs=['delay'],minfun='negll',randseed=None,returnPreds = False):#(freeparms,data,task):
    ##iv is the name of the variable to be varied
    #Reverse-transform parms
    freeparms = parmxform(freeparms,direction=-1)#utl.log_scale(freeparms,direction = -1)

    #Seed RNG if it's not None #Shouldn't matter for Bennet model though
    if not randseed is None:
        random.seed(randseed)
        
    #Insert freeparm values into vector of all parms
    if fixVal==None:
        fixVal = [np.nan] * len(freeparms)
    fixVal = np.array(fixVal)
    freeIdx = np.isnan(fixVal)
    fixVal[freeIdx] = freeparms
    parms = fixVal

    # unpack observer parameters
    observer = dict(
        gamma = parms[0], #1# intertemporal discount parameter
        k = parms[1],#.5, # uncertainty penalty
        beta = parms[2],#3., #determinism in luce
    )
    #Allow for multiple ivs -- have iv be a list of strings
    ivSet = []
    for iv in ivs:
        ivSet += [iv + 'Set']
    #delays = task['delays']
    nLevels = len(task[ivSet[0]])
    preds = np.zeros(nLevels)
    #for li,level in enumerate(task[ivSet]):
    for li in range(nLevels):
        #pi = []
        for iv in ivs:
            task[iv] = task[iv + 'Set'][li] 
        # if iv=='RA': #Special case for rewards as iv
        #     task['RB'] = task['RBSet'][li]
        #state = m1.initialState()
        Vd = makeV(observer,task)
        
        preds[li] = choice(Vd[0],Vd[1],observer['beta'])      
    #out = np.sqrt(np.mean((preds-np.array(data))**2))
    if minfun == 'rmsd':
        out = np.sqrt(np.mean((preds-np.array(data))**2))
    elif minfun == 'negll':        
        out = sum(utl.binoNegLL(preds,data['count_inform'],data['count_total']))
    elif minfun == None:
        out = None
        
    if not returnPreds:
        return(out)
    else:
        return(out,preds)



