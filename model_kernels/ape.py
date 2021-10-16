# attempted implementation of the zhu et al (2017) model.
import numpy as np
import random
import utilities as utl

# Apologies for this very non-object oriented way of doing things...
def getDetails():
    return({'name':'mZhu',
            'authname':'Zhu',
            'parmnames':['gamma','w0','w1','beta']
    })

def parmxform(parms,direction=1):
    #Zhu model log transforms only gamma and beta parms
    parms = np.array(parms)
    parms[0] = utl.logit_scale(parms[0], min=0, max=1, direction=direction)#utl.log_scale(parms[0],direction=direction)
    #XIAN! This is too hacky, please find a way to deal with it??
    if len(parms)>3:
        parms[3] = utl.log_scale(parms[3],direction=direction)    
    return(parms)

#For simplified model, use equations 7 to 12. Treat them as the values of each action
# because we can treat Q(S*|uncued) as zero (because the authors say so)
def Qbar(w,gamma,delay,r):
        return (w[0]-w[1]) * (gamma**delay) * abs(r[0]+r[1])/ 2

def choice(Qi,Qn,beta):
    #Luce choice rule
    #p_inform = np.exp(beta*Qi) / (np.exp(beta*Qi) + np.exp(beta*Qn))
    p_inform = 1./(1 + np.exp(beta*(Qn-Qi)))
    return p_inform


def minthis(freeparms,data,task,fixVal,ivs=['delay'],minfun='negll',randseed=None,returnPreds = False):
    #Script to run some function to minimise
    #Free parms are in the order:
    #[gamma,w0,w1,beta]
    #Reverse-transform parms
    freeparms = parmxform(freeparms,direction=-1)
    #freeparms = utl.log_scale(freeparms,direction = -1)

    #Insert freeparm values into vector of all parms
    if fixVal==None:
        fixVal = [np.nan] * len(freeparms)
    fixVal = np.array(fixVal)
    freeIdx = np.isnan(fixVal)
    fixVal[freeIdx] = freeparms
    parms = fixVal

    # unpack learner parameters
    learner = dict(
        gamma = parms[0], # intertemporal discount parameter
        w0 = parms[1], #weight of better outcome
        w1 = parms[2], #weight of worse cue
        beta = parms[3], #Luce determinism
    )

    #Seed RNG if it's not None
    if not randseed is None:
        random.seed(randseed)

    # #Identify the IV
    # ivSet = iv + 'Set'
        
    # #Preallocate vector of predictions    
    # preds = np.zeros(len(task[ivSet]))#matrix(0,1,length(delays))

    #Allow for multiple ivs -- have iv be a list of strings
    ivSet = []
    for iv in ivs:
        ivSet += [iv + 'Set']
    #delays = task['delays']
    nLevels = len(task[ivSet[0]])
    preds = np.zeros(nLevels)

    #for li,level in enumerate(task[ivSet]):
    for li in range(nLevels):
        for iv in ivs:
            task[iv] = task[iv + 'Set'][li] 
            #task[iv] = level
        #Get Qbar for the better outcome
        w = [learner['w0'],learner['w1']]
        rs = [task['RA'],task['RB']]
        Qb = Qbar(w,learner['gamma'],task['delay'],rs)
        #Compute probability of choice
        preds[li] = choice(Qb,0,learner['beta'])
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
