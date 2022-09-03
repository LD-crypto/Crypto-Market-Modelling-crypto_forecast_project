import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity

def getTrainTimes(t1,testTimes):
    '''
    Given testTimes, find the times of the training observations.
    —t1.index: Time when the observation started.
    —t1.value: Time when the observation ended.
    —testTimes: Times of testing observations.
    '''
    trn=t1.copy(deep=True)
    for i,j in testTimes.iteritems():
        df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
        df1=trn[(i<=trn)&(trn<=j)].index # train ends within test
        df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test
        trn=trn.drop(df0.union(df1).union(df2))
    return trn

def getEmbargoTimes(times,pctEmbargo):
    # Get embargo time for each bar
    step=int(times.shape[0]*pctEmbargo)
    if step==0:
        mbrg=pd.Series(times,index=times)
    else:
        mbrg=pd.Series(times[step:],index=times[:-step])
        mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
    return mbrg

from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observation overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o trainng samples in between
    '''
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shufﬂe=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo
    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train (with embargo)
                train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None, pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    #from src.models import PurgedKFold
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo)
        # purged
    score=[]
    for train,test in cvGen.split(X=X):
        ﬁt=clf.ﬁt(X=X.iloc[train,:],y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring=='neg_log_loss':
            prob=ﬁt.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test],prob, sample_weight=sample_weight.iloc[test].values,labels=clf.classes_)
        else:
            pred=ﬁt.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test],pred,sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)

# ------------Matrix denoising + detoning----------
def mpPDF(var,q,pts):
    # Marcenko-Pastur pdf
    # q=T/N, T is features and N is number of data
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf,index=eVal)
    return pdf

def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec

def fitKDE(obs,bWidth=.25,kernel='gaussian',x=None):
    # Fit kernelto a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:obs=obs.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).ﬁt(obs)
    if x is None: x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1: x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.ﬂatten())
    return pdf

def errPDFs(var,eVal,q,bWidth,pts=1000):
    # Fit error
    pdf0=mpPDF(var,q,pts) # theoretical pdf
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal,q,bWidth):
    from scipy.optimize import minimize
    # find max random eVal by fitting Marcenko's dist
    out=minimize(lambda *x:errPDFs(*x),.5,args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
    if out['success']: var=out['x'][0]
    else: var=1
    eMax=var*(1+ (1./q)**.5)**2
    return eMax, var

def corr2cov(corr,std):
    cov=corr*np.outer(std,std)
    return cov

def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(cov))
    corr=cov/np.outer(std,std)
    corr[corr<-1],corr[corr>1]=-1,1 # numerical error
    return corr

def denoisedCorr(eVal,eVec,nFacts):
    '''
    The constant residual method to emove noise fromm corr by fixing random eigenvalues.
    '''
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T)
    corr1=cov2corr(corr1)
    return corr1

def getDenoisedCorr(corr0,q,bWidth):
    '''
    This functions returns a denoised correlation matrix using the constant residual eigenvalue method.
    :param corr0: empirical correlation matrix
    :param q: T/N where T is features and N is number of data
    :param bWidth:
    :return: denoised correlation matrix
    '''
    eVal0,eVec0=getPCA(corr0)
    eMax0,var0=findMaxEval(np.diag(eVal0),q,bWidth)
    nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1=denoisedCorr(eVal0,eVec0,nFacts0)
    return corr1

def getCorrMetric(corr0, method='default', rnd=True):
    # returns a distance metric matrix of the correlation matrix
    d0 = corr0.copy(deep=True)
    if method == 'abs':
        d0 = (1-np.abs(d0))**.5
    else:
        d0 = (1/2*(1-d0))**.5
    if rnd:
        d0 = round(d0,5)
    return d0

# -------- Clustering ---------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def clusterKMeansBase(corr0,minNumClusters=2,maxNumClusters=10,n_init=10):
    x,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series() # observations matrix
    for init in range(n_init):
        for i in range(minNumClusters,maxNumClusters+1):
            kmeans_=KMeans(n_clusters=i,n_jobs=1,n_init=1)
            kmeans_=kmeans_.fit(x)
            silh_=silhouette_samples(x,kmeans_.labels_)
            stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh,kmeans=silh_,kmeans_
    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx] # reorder rows

    corr1=corr1.iloc[:,newIdx] # reorder columns
    clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_) } # cluster members
    silh=pd.Series(silh,index=x.index)
    return corr1,clstrs,silh

def makeNewOutputs(corr0,clstrs,clstrs2):
    clstrsNew={}
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])
    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx, newIdx]
    x = ((1 - corr0.ﬁllna(0)) / 2.) ** .5
    kmeans_labels = np.zeros(len(x.columns))
    for i in clstrsNew.keys(): idxs = [x.index.get_loc(k) for k in clstrsNew[i]]
    kmeans_labels[idxs] = i
    silhNew = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    return corrNew, clstrsNew, silhNew

def clusterKMeansTop(corr0,maxNumClusters=None,n_init=10):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1] - 1
    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters= min(maxNumClusters, corr0.shape[1] - 1), n_init = n_init)
    clusterTstats = {i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean=sum(clusterTstats.values()) / len(clusterTstats)
    redoClusters=[i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]
    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])
        corr2, clstrs2, silh2 = clusterKMeansTop(corrTmp,maxNumClusters=min(maxNumClusters,corrTmp.shape[1] - 1), n_init = n_init)
        # make new outputs, if necessary
        corrNew,clstrsNew,silhNew=makeNewOutputs(corr0,{i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2)
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean<=tStatMean:
            return corr1,clstrs,silh
        else:
            return corrNew,clstrsNew,silhNew

# ---- SEQUENTIAL BOOTSTRAP --------
def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1,i]=1.
    return indM

def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    avgU=u[u>0].mean() # average uniqueness
    return avgU

def seqBootstrap(indM,sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength=indM.shape[1]
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series()
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]
        prob=avgU/avgU.sum() # draw prob
        phi +=[np.random.choice(indM.columns,p=prob)]
    return phi

# ------ BET SIZING ------------
def getSignal(events,stepSize,prob,pred,numClasses,numThreads,**kargs):
    '''
    This functions translates class probabilties to bet sizes.
    :param events:
    :param stepSize: Size of discretization of the bet sizes
    :param prob: predicted probabiltiy of classes
    :param pred: the predicted class
    :param numClasses:
    :param numThreads:
    :param kargs:
    :return: the betsize
    '''
    from scipy.stats import norm
    if prob.shape[0]==0:
        return pd.Series()
    # 1) generate signals from multinomial classification
    signal0 = (prob -1./numClasses)/(prob*(1.-prob))**.5 # t-values of OvR
    signal0 = pred*(2*norm.cdf(signal0)-1) # signal =side*size
    if 'side' in events:
        signal0*=events.loc[signal0.index,'side'] # meta-labeling
    # 2) compute average signal among those concurrently open
    df0=signal0.to_frame('signal').join(events[['t1']],how='left')
    df0=avgActiveSignals(df0,numThreads)
    signal1=discreteSignal(signal0=df0,stepSize=stepSize)
    return signal1

def avgActiveSignals(signals,numThreads):
    # compute the average signal among those active
    # 1) time points where signals change (either one starts or one ends)
    tPnts=set(signals['t1'].dropna().values)
    tPnts=tPnts.union(signals.index.values)
    tPnts=list(tPnts);tPnts.sort()
    out=mpAvgActiveSignals(signals,tPnts)
    return out

def mpAvgActiveSignals(signals,molecule):
    '''
    At time loc, average signal among those still active.
    Signal is active is active if:
        a) issued before or at loc AND
        b) loc before signal's endtime, or endtime is still unknown (NaT)
    '''
    out=pd.Series()
    for loc in molecule:
        df0=(signals.index.values<=loc)&((loc<signals['t1'])|pd.isnull(signals['t1']))
        act=signals[df0].index
        if len(act)>0:
            out[loc]=signals.loc[act,'signal'].mean()
        else:
            out[loc]=0 # no signals active at this time
    return out

def discreteSignal(signal0,stepSize):
    # discretize signal
    signal1=(signal0/stepSize).round()*stepSize
    signal1[signal1>1]=1 # cap
    signal1[signal1<-1] # floor
    return signal1
