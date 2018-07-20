# -*- coding: utf-8 -*-

###############################################################################
### The objective of this code is to calculate the                   ##########
### prediction and compare results.                                  ##########
###############################################################################

# Import packages:
import numpy as np
import pandas as pd
import copy
import math
import multiprocessing
from numba import jit

@jit
def probabModel(slNdMrgnTempNp,sazoIndexVec):

    w1 = sazoIndexVec[0]
    w2 = sazoIndexVec[1]
    w3 = sazoIndexVec[2]

    minDWeek = slNdMrgnTempNp[:,1].min()+6
    maxDWeek = slNdMrgnTempNp[:,1].max()
    m = maxDWeek-minDWeek+1
    n = slNdMrgnTempNp.shape[0]
    count = 0

    # 3.8.1 - Probability of selling each quantity per week:
    listSales = np.zeros(shape=(int(m),3))
    for ii in range(0,int(m)):
        listSales[ii,1] = 1l
        v1 = minDWeek-6+ii
        v2 = minDWeek+ii
        for jj in range(0,n):
            if((slNdMrgnTempNp[jj,1] >= v1)&(slNdMrgnTempNp[jj,1] <= v2)):
                # Col 2: QtdeProcDes | Col 3: QtdeProc:
                listSales[ii,0] = listSales[ii,0] + slNdMrgnTempNp[jj,2]
                # Col 9: flag:
                listSales[ii,2] = listSales[ii,2] + slNdMrgnTempNp[jj,9]
        listSales[ii,0] = round(listSales[ii,0])

    #slNdMrgnTemp222.to_csv('C:\\Users\\lucas.sala\\Desktop\\slNdMrgnTemp222.csv',sep=';',index=False)
    #listSales = pd.DataFrame(listSales)
    #listSales.to_csv('C:\\Users\\lucas.sala\\Desktop\\listSales.csv',sep=';',index=False)

    # 3.8.2 - Filter Ruptura:
    index = listSales[:,2] < 3
    if any(index):
        listSales2 = listSales[index]
    else:
        listSales2 = listSales[[0,1],:]
        listSales2[:,0] = 0
        listSales2[:,1] = 1
        listSales2[:,2] = 0
    qttyPoints = listSales2.shape[0]
    
    #qttyPoints = listSales2.shape[0]
    uniqVec = np.unique(listSales2[:,0])
    m = uniqVec.shape[0]
    n = listSales2.shape[0]
    tableSales = np.zeros(shape = (m,3))
    for ii in range(m):
        tableSales[ii,0] = uniqVec[ii]
        for jj in range(n):
            if(uniqVec[ii] == listSales2[jj,0]):
                tableSales[ii,1] = tableSales[ii,1] + 1
                tableSales[ii,2] = tableSales[ii,2] + (1/float(n))

    lenTab = tableSales.shape[0]
    listProbTemp = np.zeros(shape=(lenTab**2,2))
    countDif = 0
    for ii in range(lenTab):
        vn = tableSales[ii,0]
        probn = tableSales[ii,2]
        for jj in range(lenTab):
            vn2 = round((w2*vn)+(w1*tableSales[jj,0]))
            probn2 = probn*tableSales[jj,2]
            flag = 1
            ll = 0
            while (flag == 1):
                if (ll == countDif):
                    listProbTemp[ll,0] = vn2
                    listProbTemp[ll,1] = probn2
                    flag = 0
                    countDif += 1
                elif (abs(listProbTemp[ll,0]-vn2)<0.0001):
                    listProbTemp[ll,1] = listProbTemp[ll,1] + probn2
                    flag = 0
                ll += 1
    listProbTemp = listProbTemp[range(0,countDif),:]
    n = listProbTemp.shape[0]
    listProb = np.zeros(shape=((lenTab*n),2))
    countDif = 0
    for ii in range(lenTab):
        vn = tableSales[ii,0]
        probn = tableSales[ii,2]
        for jj in range(n):
            vn2 = round((w3*vn)+listProbTemp[jj,0])
            probn2 = probn*listProbTemp[jj,1]
            flag = 1
            ll = 0
            while (flag == 1):
                if (ll == countDif):
                    listProb[ll,0] = vn2
                    listProb[ll,1] = probn2
                    flag = 0
                    countDif += 1
                elif (abs(listProb[ll,0]-vn2)<0.0001):
                    listProb[ll,1] = listProb[ll,1] + probn2
                    flag = 0
                ll += 1
    listProb = listProb[range(0,countDif),:]
    tempVec = listProb[:,0].argsort()
    finalProb = listProb[tempVec,:]
    finalProbTemp = np.zeros(shape = (20,2))
    for ii in range(0,finalProbTemp.shape[0]):
        acumP = (ii+1)*0.05
        count = 0
        acumS = finalProb[count,1]
        while acumS < (acumP-0.000000001):
            count += 1
            acumS += finalProb[count,1]
        finalProbTemp[ii,0] = acumP
        finalProbTemp[ii,1] = finalProb[count,0]
    return([finalProbTemp,qttyPoints])

def ConsolidateSKUinfo(modelAdvisia,modelMagazine,uniqWeekCodes,SKU,priceSKU,costSKU):
    # 4- Consolidate information per Sku:
    #
    # modelMagazine[i][j]
    # i - Results for Safra i
    # j:
    # 0- meanSales ("Media Seca")
    # 1- sdSales (standard deviation from meanSales)
    # 2- VMD60
    # 3- slNdMrgnTemp2 - Sales per date from safra j-1 and j-2
    # 4- realSales - Sales from safra i
    # 5- sales2pastMo - All sales
    # 6- SKU_PandC - Price and cost per month
    # 7- currentRup - "Ruptura" per validation week
    # 8- VMDEdu - Planning team calculation
    # 9- firstSale - First Week with Sales:
    # 10- qttyPoints- number of points for analysis
    # 11- maxFuture - Maximum sales based on stock period
    # 12- countSW - Number of previous weeks with stockout

    # 4.1 (Test 2) - Consolidation Advisia Model:
    teste = np.vstack(modelAdvisia)

    # Pre allocate results table:
    toExportTable = np.zeros(shape = (teste.shape[0],
                                     (teste.shape[1]+27)))

    # 4.1.1 - Prediction Sales:
    for j in xrange(0,teste.shape[1]):
        toExportTable[range(0,teste.shape[0]),j] = teste[:,j]

    # 4.1.2 - Create columns:
    cteCol = teste.shape[1]-1 # +len(modelMagazine)

    # c1-VMDAd | c2-VMDEdu | c3-Vendas | c4-maxFuture | c5-Initial
    # c6-QtdePontos | c7-ValidationRupture | c8-flag | c9-flag1
    # c10-QtdeFinalAd1 | c11-QtdeFinalAd2 | c12-QtdeFinalMl
    # c13-ErroAd | c14-AbsErroAd | c15-ErroMl | c16-AbsErroMl
    # c17-Cluster | c18-Preco | c19-custo | c20-VendasPerdidasAd
    # c21-EstoqueExtraAd | c22-VendasPerdidasMl | c23-EstoqueExtraMl
    # c24-PerdasTotaisAd | c25-PerdasTotaisMl | c26-CustoAd | c27-CustoMl
    # c28-RuptValid
    listColumns = [(cteCol+1),(cteCol+2),(cteCol+3),(cteCol+4),(cteCol+5),
                   (cteCol+6),(cteCol+7),(cteCol+8),(cteCol+9),(cteCol+10),
                   (cteCol+11),(cteCol+12),(cteCol+13),(cteCol+14),(cteCol+15),
                   (cteCol+16),(cteCol+17),(cteCol+18),(cteCol+19),(cteCol+20),
                   (cteCol+21),(cteCol+22),(cteCol+23),(cteCol+24),(cteCol+25),
                   (cteCol+26),(cteCol+27)]
    listColNames = ['VMDAd','VMDEdu','Vendas','maxFuture','Initial','QtdePontos',
                    'ValidationRupture','flag','flag1','QtdeFinalAd1','QtdeFinalAd2',
                    'QtdeFinalMl','ErroFinal','AbsErroFinal','ErroFinalMl','AbsErroFinalMl',
                    'Cluster','Preco','custo','VendasPerdidasAd','EstoqueExtraAd',
                    'VendasPerdidasMl','EstoqueExtraMl','PerdasTotaisAd','PerdasTotaisMl',
                    'CustoAd','CustoMl']

    # c1-VMDAd:
    for j in xrange(0,toExportTable.shape[0]):
        vmdAd = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][0][0,5]
        if math.isnan(vmdAd):
            vmdAd = 0
        toExportTable[j,listColumns[0]] = vmdAd

    # c2-VMDEdu:
    for j in xrange(0,toExportTable.shape[0]):
        toExportTable[j,listColumns[1]] = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][8][0,0]

    # c3-Vendas:
    for j in xrange(0,toExportTable.shape[0]):
        toExportTable[j,listColumns[2]] = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][4][0,0]

    # c4-maxFuture:
    for j in xrange(0,toExportTable.shape[0]):
        toExportTable[j,listColumns[3]] = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][11]

    # c5-Initial:
    for j in xrange(0,toExportTable.shape[0]):
        toExportTable[j,listColumns[4]] = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][9]

    # c6-QtdePontos:
    for j in xrange(0,toExportTable.shape[0]):
        toExportTable[j,listColumns[5]] = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][10] # Qtde Ptos:

    # c7-ValidationRupture:
    for j in xrange(0,toExportTable.shape[0]):  # 29
        toExportTable[j,listColumns[6]] = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][7][0,0]
                                           
    # c8-flag | c9-flag1:
    for j in xrange(0,toExportTable.shape[0]):
        if (modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][10] > 30):
            toExportTable[j,listColumns[7]] = 1 # flag
        else:
            toExportTable[j,listColumns[7]] = 0 # flag
        
        # Number of previous weeks with stockout:
        toExportTable[j,listColumns[8]] = modelMagazine[int(toExportTable[j,1] - uniqWeekCodes[0])][12]

    # c10-QtdeFinalAd1 | c11-QtdeFinalAd2 | c12-QtdeFinalMl
    # c13-ErroAd | c14-AbsErroAd | c15-ErroMl | c16-AbsErroMl
    for j in xrange(0,toExportTable.shape[0]):
        if ((toExportTable[j,listColumns[8]] <= 4)&(toExportTable[j,listColumns[7]] == 1)):
            toExportTable[j,listColumns[9]] = toExportTable[j,8] # QtdeFinalAd1 = Mod9
        elif ((toExportTable[j,listColumns[8]] <= 6)&(toExportTable[j,listColumns[7]] == 1)):
            toExportTable[j,listColumns[9]] = toExportTable[j,7] # QtdeFinalAd1 = Mod13
        elif (toExportTable[j,listColumns[8]] <= 8):
            toExportTable[j,listColumns[9]] = toExportTable[j,6] # QtdeFinalAd1 = Mod17
        elif (toExportTable[j,listColumns[8]] <= 10):
            toExportTable[j,listColumns[9]] = toExportTable[j,5] # QtdeFinalAd1 = Mod21
        elif (toExportTable[j,listColumns[8]] <= 13):
            toExportTable[j,listColumns[9]] = toExportTable[j,4] # QtdeFinalAd1 = Mod26
        else:
            toExportTable[j,listColumns[9]] = toExportTable[j,3] # QtdeFinalAd1 = Mod52
        # This limitation can be both current stock/future stock and VMD:
        # The reason for limiting is:
        # When we optimize our coverage, if we priorize products which will 
        # not be sold then we will underestimate products that definetly will.
        # Although when suggesting a purchase, the numbers shall be based on
        # the real sales.
        if (toExportTable[j,listColumns[1]] > 0):
            limitUp = toExportTable[j,listColumns[3]] #,(21*1.5*toExportTable[j,listColumns[1]])) # future stock or VMD
        else:
            limitUp = toExportTable[j,listColumns[3]] # future stock or VMD
        limitDw = 0 # VMD
        if ((toExportTable[j,listColumns[8]] <= 4)&(toExportTable[j,listColumns[7]] == 1)):
            toExportTable[j,listColumns[10]] = max(limitDw,min(limitUp,toExportTable[j,8])) # QtdeFinalAd2 = min(Mod9,limitation)
        elif ((toExportTable[j,listColumns[8]] <= 6)&(toExportTable[j,listColumns[7]] == 1)):
            toExportTable[j,listColumns[10]] = max(limitDw,min(limitUp,toExportTable[j,7])) # QtdeFinalAd2 = min(Mod13,limitation)
        elif (toExportTable[j,listColumns[8]] <= 8):
            toExportTable[j,listColumns[10]] = max(limitDw,min(limitUp,toExportTable[j,6])) # QtdeFinalAd2 = min(Mod17,limitation)
        elif (toExportTable[j,listColumns[8]] <= 10):
            toExportTable[j,listColumns[10]] = max(limitDw,min(limitUp,toExportTable[j,5])) # QtdeFinalAd2 = min(Mod21,limitation)
        elif (toExportTable[j,listColumns[8]] <= 13):
            toExportTable[j,listColumns[10]] = max(limitDw,min(limitUp,toExportTable[j,4])) # QtdeFinalAd2 = min(Mod26,limitation)
        else:
            toExportTable[j,listColumns[10]] = max(limitDw,min(limitUp,toExportTable[j,3])) # QtdeFinalAd2 = min(Mod52,limitation)
        # Qtde Final Ml
        if (toExportTable[j,listColumns[1]] < 0):
            toExportTable[j,listColumns[11]] = 0
        elif (toExportTable[j,listColumns[1]] >= 0):
            toExportTable[j,listColumns[11]] = 21*toExportTable[j,listColumns[1]]
        # Erro ad / Abs error ad:
        toExportTable[j,listColumns[12]] = toExportTable[j,listColumns[10]]-toExportTable[j,listColumns[2]]
        toExportTable[j,listColumns[13]] = abs(toExportTable[j,listColumns[12]])
        # Erro ml / Abs error ml:
        toExportTable[j,listColumns[14]] = toExportTable[j,listColumns[11]]-toExportTable[j,listColumns[2]]
        toExportTable[j,listColumns[15]] = abs(toExportTable[j,listColumns[14]])
        
    # Cluster, Preco e custo(based on last 10 weeks):
    meanSalesSKU = 0
    countSKU = 0
    index = priceSKU['IBMCode'] == SKU
    priceSKUTemp = priceSKU.loc[index,'FakeProc'].max()
    index = costSKU['IBMCode'] == SKU
    costSKUTemp = costSKU.loc[index,'costUniq'].max()
    for j in range((len(modelMagazine)-7),len(modelMagazine)):
        if(math.isnan(modelMagazine[j][0][0,5])):
            meanSalesSKU = meanSalesSKU
        else:
            meanSalesSKU = meanSalesSKU + modelMagazine[j][0][0,5]
        countSKU = countSKU + 1

    meanSalesSKU = meanSalesSKU / float(countSKU)

    # Int VMD			Int Price			Cluster
    if (toExportTable[j,listColumns[4]] <= 182):

        tempCluster = 0

        # 0                           c0                  c1
        if (meanSalesSKU == 0):
            tempCluster = 1

        # 0	   0.1		0	320		     c1b                 c2
        if ((meanSalesSKU>0)&(meanSalesSKU<=0.1)&(priceSKUTemp>0)&(priceSKUTemp<=320)):
            tempCluster = 2
    
        # 0	   0.1		320	800		   c1m                c3
        if ((meanSalesSKU>0)&(meanSalesSKU<=0.1)&(priceSKUTemp>320)&(priceSKUTemp<=800)):
            tempCluster = 3
    
        # 0	   0.1		800	6000		   c1a                c4
        if ((meanSalesSKU>0)&(meanSalesSKU<=0.1)&(priceSKUTemp>800)&(priceSKUTemp<=6000)):
            tempCluster = 4

        # 0.1	0.3333	0	140		     c2b                 c5
        if ((meanSalesSKU>0.1)&(meanSalesSKU<=0.3333)&(priceSKUTemp>0)&(priceSKUTemp<=140)):
            tempCluster = 5

        # 0.1	0.3333	140	400		   c2m                c6
        if ((meanSalesSKU>0.1)&(meanSalesSKU<=0.3333)&(priceSKUTemp>140)&(priceSKUTemp<=400)):
            tempCluster = 6

        # 0.1	0.3333	400	6000		   c2a                c7
        if ((meanSalesSKU>0.1)&(meanSalesSKU<=0.3333)&(priceSKUTemp>400)&(priceSKUTemp<=6000)):
            tempCluster = 7

        # 0.3333	0.523809524		     0	150		 c3b     c8
        if ((meanSalesSKU>0.3333)&(meanSalesSKU<=0.523809524)&(priceSKUTemp>0)&(priceSKUTemp<=150)):
            tempCluster = 8

        # 0.3333	0.523809524		     150	6000	 c3a     c9
        if ((meanSalesSKU>0.3333)&(meanSalesSKU<=0.523809524)&(priceSKUTemp>150)&(priceSKUTemp<=6000)):
            tempCluster = 9

        # 0.523809524	0.761904762		0	130		 c4b     c10
        if ((meanSalesSKU>0.523809524)&(meanSalesSKU<=0.9)&(priceSKUTemp>0)&(priceSKUTemp<=130)):
            tempCluster = 10

        # 0.523809524	0.761904762		130	6000	 c4a     c11
        if ((meanSalesSKU>0.523809524)&(meanSalesSKU<=0.9)&(priceSKUTemp>130)&(priceSKUTemp<=6000)):
            tempCluster = 11
    
        # 0.761904762	0.952380952		0	6000		 c5    	 c12						
        if ((meanSalesSKU>0.9)&(meanSalesSKU<=1.5)&(priceSKUTemp>0)&(priceSKUTemp<=6000)):
            tempCluster = 12
    
        # 0.952380952	1.285714286		0	6000		 c6      c13
        if ((meanSalesSKU>1.5)&(meanSalesSKU<=2.000)&(priceSKUTemp>0)&(priceSKUTemp<=6000)):  #1.285714286
            tempCluster = 13
    
        # 1.285714286	100		0	6000		         c7      c14
        if ((meanSalesSKU>2.000)&(meanSalesSKU<=100)&(priceSKUTemp>0)&(priceSKUTemp<=6000)):
            tempCluster = 14
    else:
        tempCluster = -1

    for j in range(0,toExportTable.shape[0]):
        toExportTable[j,listColumns[16]] = tempCluster  # cluster
        toExportTable[j,listColumns[17]] = priceSKUTemp # price
        toExportTable[j,listColumns[18]] = costSKUTemp  # cost

    # Vendas Perdidas Ad
    # Estoque Extra Ad
    # Vendas Perdidas Ml
    # Estoque Extra Ml
    # Perdas Totais Ad
    # Perdas Totais Ml
    # Custo Ad
    # Custo Ml
    for j in range(0,toExportTable.shape[0]):
        if (toExportTable[j,listColumns[12]]<0):
            toExportTable[j,listColumns[19]] = abs(toExportTable[j,listColumns[12]]*toExportTable[j,listColumns[17]]) # Vendas Perdidas Ad
        else:
            toExportTable[j,listColumns[19]] = 0
        if (toExportTable[j,listColumns[12]]>0):
            toExportTable[j,listColumns[20]] = abs(toExportTable[j,listColumns[12]]*toExportTable[j,listColumns[18]]) # Estoque Extra Ad
        else:
            toExportTable[j,listColumns[20]] = 0
        if (toExportTable[j,listColumns[14]]<0):
            toExportTable[j,listColumns[21]] = abs(toExportTable[j,listColumns[14]]*toExportTable[j,listColumns[17]]) # Vendas Perdidas Ml
        else:
            toExportTable[j,listColumns[21]] = 0
        if (toExportTable[j,listColumns[14]]>0):
            toExportTable[j,listColumns[22]] = abs(toExportTable[j,listColumns[14]]*toExportTable[j,listColumns[18]]) # Estoque Extra Ml
        else:
            toExportTable[j,listColumns[22]] = 0
        
        toExportTable[j,listColumns[23]] = toExportTable[j,listColumns[19]]+toExportTable[j,listColumns[20]]
        toExportTable[j,listColumns[24]] = toExportTable[j,listColumns[21]]+toExportTable[j,listColumns[22]]
        toExportTable[j,listColumns[25]] = toExportTable[j,listColumns[10]]*toExportTable[j,listColumns[18]]
        toExportTable[j,listColumns[26]] = toExportTable[j,listColumns[11]]*toExportTable[j,listColumns[18]]

    toExportTable = pd.DataFrame(toExportTable)
    #print(toExportTable.shape)
    return(toExportTable)

def ProjectDemanda(listInput):
    # 0 - SKU
    # 1 - finalBaseP
    # 2 - estoque
    # 3 - sazoBase2
    # 4 - deParaTable
    # 5 - priceSKU
    # 6 - costSKU
    SKU = listInput[0]
    print(SKU)
    finalBaseP = listInput[1]
    estoque = listInput[2]
    sazoBase2 = listInput[3]
    deParaTable = listInput[4]
    priceSKU = listInput[5]
    costSKU = listInput[6]

    # Define sazonal base:
    sazoBase = finalBaseP.groupby(['category',
                                   'catSazo',
                                   'SemanaCode']).agg({'SazoIndexUsed':'mean'}).reset_index()
    uniqWeekCodes = [180,181,182,183,184,185,186,187,188]
    qttyWeeksAnaly = [52,26,21,17,13,9]

    # 1- Create temporary base for this SKU
    # 1.1 - finalBaseP:
    index = finalBaseP['IBMCode'] == SKU
    tempBase = finalBaseP[index]
    SKUcategory = tempBase['category'].unique().min()
    SKUsubcategory = tempBase['catSazo'].unique().min()
    #tempBase.to_csv('C:\\Users\\lucas.sala\\Dropbox (ADVISIA)\\201708 Magazine Luiza - Demand Pred\\tempBase.csv',index=False)

    # 1.3 - Estoque:
    index = estoque['IBMCode'] == SKU
    tempEstoque = estoque[index]
    tempEstoque2 = tempEstoque[['CaptureDate', 'Futu']]
    tempEstoque = tempEstoque[['CaptureDate', 'Total']]

    tempEstoque3 = pd.merge(tempEstoque2, deParaTable, how='left', on=['CaptureDate'])
    tempEstoque4 = tempEstoque3.groupby('SemanaCode').agg({'Futu':'max'}).reset_index()

    tempEstoque3 = pd.merge(tempEstoque, deParaTable, how='left', on=['CaptureDate'])
    tempEstoque5 = tempEstoque3.groupby('SemanaCode').agg({'Total':'max'}).reset_index()
    
    # 2- Product Info:
    tempB1 = copy.deepcopy(tempBase)
    tempB2 = tempB1.groupby('SemanaCode').agg({'QtdeProc':'sum',
                                               'FakeProc':'sum',
                                               'CostProc':'sum'}).reset_index()
    tempB2['price'] = tempB2['FakeProc']/tempB2['QtdeProc']
    tempB2['cost'] = tempB2['CostProc']/tempB2['QtdeProc']

    # 2.1 - Updated cost and price:
    SKU_PandC = tempB2[['SemanaCode',
                        'price',
                        'cost']]
    # SKU_PandC = 0

    # 2.2 - Margin and sales over time:
    slNdMrgn = tempBase.groupby(['SemanaCode',
                                 'DataCode']).agg({'QtdeProc':'sum',
                                                   'MargemRod':'mean',
                                                   'PedidosUni':'sum',
                                                   'QtdeProcDes':'sum',
                                                   'SazoIndexUsed':'mean'}).reset_index()
    slNdMrgn['QtdePerOrder'] = slNdMrgn['QtdeProc']/slNdMrgn['PedidosUni']

    # 2.3 - Remove strange values:
    index = slNdMrgn['QtdeProc'] != 0
    slNdMrgn = slNdMrgn[index]

    # 3- Create model "VMD", "Media Seca" and "Advisia" for last months:
    modelMagazine = []
    modelAdvisia = []
    #maxSales = 0

    # Modeling:
    for j in range(0,len(uniqWeekCodes)):
        
        # Pre allocate Model Magazine:
        meanSales = np.zeros(shape=(1,len(qttyWeeksAnaly)))
        sdSales = np.zeros(shape=(1,len(qttyWeeksAnaly)))
        VMD60 = np.zeros(shape=(1,len(qttyWeeksAnaly)))
        sales2pastMo = np.zeros(shape=(3,len(qttyWeeksAnaly)))
        realSales = np.zeros(shape=(6,len(qttyWeeksAnaly)))
        currentRup = np.zeros(shape=(1,len(qttyWeeksAnaly)))
        VMDEdu = np.zeros(shape=(1,len(qttyWeeksAnaly)))

        firstSale = slNdMrgn['SemanaCode'].min()
        firstSaleD = slNdMrgn['DataCode'].min()

        # Get sazonal index:
        index1 = sazoBase['category'] == SKUcategory
        index2 = sazoBase['catSazo'] == SKUsubcategory
        index3 = sazoBase['SemanaCode'] == uniqWeekCodes[j]
        w1 = sazoBase.loc[((index1)&(index2)&(index3)),'SazoIndexUsed'].max()

        index1 = sazoBase['category'] == SKUcategory
        index2 = sazoBase['catSazo'] == SKUsubcategory
        index3 = sazoBase['SemanaCode'] == (uniqWeekCodes[j]+1)
        w2 = sazoBase.loc[((index1)&(index2)&(index3)),'SazoIndexUsed'].max()

        index1 = sazoBase['category'] == SKUcategory
        index2 = sazoBase['catSazo'] == SKUsubcategory
        index3 = sazoBase['SemanaCode'] == (uniqWeekCodes[j]+2)
        w3 = sazoBase.loc[((index1)&(index2)&(index3)),'SazoIndexUsed'].max()

        if math.isnan(w1):
            w1 = 1
        if math.isnan(w2):
            w2 = 1
        if math.isnan(w3):
            w3 = 1
        
        sazoIndexVec = np.array([w1,w2,w3])
        
        # Pre allocate Model Advisia:
        modelAdvisiaTemp = []

        # Get future stock:
        maxFuture = 99999999
        index1 = tempEstoque3['Total'].diff() > 1
        index2 = tempEstoque3['SemanaCode'] < uniqWeekCodes[j]
        safraValue = tempEstoque3.loc[((index1)&(index2)),'SemanaCode'].max()
        if(math.isnan(safraValue)):
            safraValue = uniqWeekCodes[j]-3
        index1 = tempEstoque4['SemanaCode'].isin(range(int(safraValue),uniqWeekCodes[j]))
        index2 = tempEstoque5['SemanaCode'] == uniqWeekCodes[j]
        if ((any(index1) == False)&(any(index2) == False)):
            maxFuture = 0
        elif ((any(index1) == True)&(any(index2) == True)):
            maxFuture = tempEstoque4.loc[index1,'Futu'].max() + tempEstoque5.loc[index2,'Total'].max()
        elif (any(index2) == True):
            maxFuture = tempEstoque5.loc[index2,'Total'].max()
        elif (any(index1) == True):
            maxFuture = tempEstoque4.loc[index1,'Futu'].max()
        else:
            maxFuture = tempEstoque4.loc[index1,'Futu'].max() + tempEstoque5.loc[index2,'Total'].max()

        if(math.isnan(maxFuture)):
            maxFuture = 99999999

        # Get past Stockout:
        index1 = tempEstoque5['SemanaCode'] <= (uniqWeekCodes[j]-1)
        index2 = tempEstoque5['SemanaCode'] >= (uniqWeekCodes[j]-26)
        index3 = tempEstoque5['Total'] > 1
        SKUtempEstoque5 = tempEstoque5[((index1)&(index2)&(index3))]
        flag = 1
        countSW = 0
        for i in range(1,27):
            index = SKUtempEstoque5['SemanaCode'] == (uniqWeekCodes[j]-i)
            if ((flag == 1)&(any(index))):
                flag = 0
            elif (flag == 1):
                countSW += 1

        # Qtty of Months Model:
        for l in range(0,len(qttyWeeksAnaly)):
            
            #timeStart = time.time()
            
            # 3.1 - Filter base:
            # 3.1.1 - Filter for months prediction:
            index1 = slNdMrgn['SemanaCode'] <= (uniqWeekCodes[j]-1)
            index2 = slNdMrgn['SemanaCode'] >= (uniqWeekCodes[j]-qttyWeeksAnaly[l])
            index3 = slNdMrgn['SemanaCode'] >= uniqWeekCodes[j]
            index4 = slNdMrgn['SemanaCode'] <= uniqWeekCodes[j]+2

            slNdMrgnTemp10 = slNdMrgn[(index1&index2)]
            slNdMrgnTemp30 = slNdMrgn[(index3&index4)]

            # 3.1.2 - Filter for sales Outliers:
            index1 = slNdMrgnTemp10['MargemRod']>=-0.3
            slNdMrgnTemp11 = slNdMrgnTemp10[index1]

            # 3.1.3 - Filter for mean sales per order:
            index1 = slNdMrgnTemp11['QtdePerOrder']<=5
            slNdMrgnTemp12 = slNdMrgnTemp11[index1]
            
            sales2pastMo[0,l] = slNdMrgnTemp10['QtdeProc'].sum()

            # 3.2 - Add dates with no sales:
            index1 = deParaTable['SemanaCode'] <= (uniqWeekCodes[j]-1)
            index2 = deParaTable['SemanaCode'] >= (uniqWeekCodes[j]-qttyWeeksAnaly[l])
            index3 = deParaTable['SemanaCode'] >= uniqWeekCodes[j]
            index4 = deParaTable['SemanaCode'] <= uniqWeekCodes[j]+2

            deParaTableTemp = deParaTable[(index1&index2)]
            slNdMrgnTemp22 = pd.merge(deParaTableTemp, slNdMrgnTemp12, how='left', on=['SemanaCode', 'DataCode'])
            
            deParaTableTemp = deParaTable[(index3&index4)]
            slNdMrgnTemp40 = pd.merge(deParaTableTemp, slNdMrgnTemp30, how='left', on=['SemanaCode', 'DataCode'])
            
            # 3.3 - Add "estoque" dates:
            slNdMrgnTemp222 = pd.merge(slNdMrgnTemp22, tempEstoque, how='left', on=['CaptureDate'])

            slNdMrgnTemp440 = pd.merge(slNdMrgnTemp40, tempEstoque, how='left', on=['CaptureDate'])

            slNdMrgnTemp222 = slNdMrgnTemp222.drop(['CaptureDate'], axis=1)

            slNdMrgnTemp440 = slNdMrgnTemp440.drop(['CaptureDate'], axis=1)

            # 3.4 - Remove NA's from merging:
            index = slNdMrgnTemp222.isnull()
            slNdMrgnTemp222[index] = 0

            index = slNdMrgnTemp440.isnull()
            slNdMrgnTemp440[index] = 0

            # 3.5 - Add flag for "RUPTURA"
            slNdMrgnTemp222['flag'] = 0
            index1 = slNdMrgnTemp222['Total'] <= 0.001
            index2 = slNdMrgnTemp222['QtdeProc'] == 0.00
            slNdMrgnTemp222.loc[((index1)&(index2)),'flag'] = 1

            # 3.6 - VMD:
            tempEdu = -1

            maxDay = slNdMrgnTemp222['DataCode'].max()

            index1 = slNdMrgnTemp222['DataCode'] >= max((maxDay-59),firstSaleD)
            index2 = slNdMrgnTemp222['flag'] == 0
            
            meanSales[0,l] = slNdMrgnTemp222.loc[((index1)&(index2)),'QtdeProc'].mean()
            realSales[0,l] = slNdMrgnTemp440['QtdeProc'].sum()
            realSales[3,l] = slNdMrgnTemp440['QtdeProcDes'].sum()
            VMDEdu[0,l] = tempEdu
                  
            # 3.7 - Get rupture:
            currentRup[0,l] = sum(slNdMrgnTemp440['Total'] < 0.001)
            
            # 3.8 - Advisia model:
            
            # 3.8.0 - Transform into numpy ndarray:
            slNdMrgnTempNp = slNdMrgnTemp222.values
            
            # 3.8.1 - Caclulate probabilistic model:
            qttyPoints = 0
            listOutput = probabModel(slNdMrgnTempNp,sazoIndexVec)
            finalProbTemp = listOutput[0]
            qttyPoints = listOutput[1]
            
            # 3.8.2 - Save Model:
            modelAdvisiaTemp.append(finalProbTemp)
            #print(str(time.time()-timeStart))

        # 3.7 - Transform results into matrix:
        modelAdvisiaJ = np.zeros(shape = (modelAdvisiaTemp[0].shape[0],(len(modelAdvisiaTemp)+3)))
        modelAdvisiaJ[:,0] = SKU
        modelAdvisiaJ[:,1] = uniqWeekCodes[j]
        modelAdvisiaJ[:,2] = modelAdvisiaTemp[0][:,0]
        for l in range(0,len(modelAdvisiaTemp)):
            modelAdvisiaJ[:,(l+3)] = modelAdvisiaTemp[l][:,1]

        modelAdvisia.append(modelAdvisiaJ)

        modelMagazineTemp = [meanSales,         # 0  - Mean Sales
                             sdSales,           # 1  - Std Deviation from sales
                             VMD60,             # 2  - VMD60 considering outliers
                             slNdMrgnTemp222,   # 3  - Base used for modeling (Treat mrgm, outliersand missing data)
                             realSales,         # 4  - Real sales on validation
                             sales2pastMo,      # 5  - sales last 2 months
                             SKU_PandC,         # 6  - SKU pricing history
                             currentRup,        # 7  - Days in stockout validation
                             VMDEdu,            # 8  - VMD from Eduardo
                             firstSale,         # 9  - First date with sales
                             qttyPoints,        # 10 - Quantity of points
                             maxFuture,         # 11 - Maximum sales based on stock period
                             countSW]           # 12 - Number of previous weeks with stockout
        modelMagazine.append(modelMagazineTemp)
    
    toExportTable = ConsolidateSKUinfo(modelAdvisia,modelMagazine,uniqWeekCodes,SKU,priceSKU,costSKU)

    # 4.2 Return:
    return(toExportTable.values)

# Set working directory:
def main():

    #pessoa = 'paperspace'
    pessoa = 'lucas.sala'
    root = 'C:\\Users\\'+pessoa+'\\Dropbox (ADVISIA)\\201708 Magazine Luiza - Demand Pred\\02 Data Gathering\\Bases\\'

    # Using "finalBaseP" created by "SellingAnalysis.R":
    finalBaseP = pd.read_csv(root+'Output\\finalBaseP.csv')
    estoque = pd.read_csv(root+'Output\\estoqueAnaDIB2.csv')
    sazoBase2 = pd.read_csv(root+'Output\\sazoSemanaCode.csv',sep=";")
    priceSKU = pd.read_csv(root+'Output\\priceSKUV2.csv',sep=";")
    costSKU = pd.read_csv(root+'Output\\costSKUV2.csv',sep=";")
	
    deParaTable = finalBaseP.groupby(['SemanaCode',
                                      'DataCode']).agg({'CaptureDate':'first'}).reset_index()
    deParaTable = deParaTable[['SemanaCode','DataCode','CaptureDate']]

    # Get only SKUs specified:
    uniqIBMVec = estoque['IBMCode'].unique()

    index = finalBaseP['IBMCode'].isin(uniqIBMVec)
    finalBaseP = finalBaseP[index]
    index = estoque['IBMCode'].isin(uniqIBMVec)
    estoque = estoque[index]

    # Filter incomplete SemanaCode:
    #index = finalBaseP['SemanaCode'] != 154
    #finalBaseP = finalBaseP[index]

    index = finalBaseP['Eq2017'].isin([1,37,47])
    offDates = finalBaseP.loc[index,:] # remove off dates!
    offDates = offDates[['CaptureDate','Year','MonthCode',
                         'SemanaCode','DataCode','Semana','Eq2017']]
    offDatesVec = offDates['CaptureDate'].unique()

    index = index == False
    finalBaseP = finalBaseP[index]
    
    index = estoque['CaptureDate'].isin(offDatesVec)
    index = index == False
    estoque = estoque[index] # remove off dates!

    # Small test:
    print(((((finalBaseP['QtdeProc']/finalBaseP['SazoIndexUsed'])-finalBaseP['QtdeProcDes']).abs()<0.0001).shape ==
    (((finalBaseP['QtdeProc']/finalBaseP['SazoIndexUsed'])-finalBaseP['QtdeProcDes']).abs()<0.0001).sum())[0])

    # Create model for each SKU:
    uniqProd = pd.unique(finalBaseP['IBMCode'])

    # Great SKU Optimization:
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(ProjectDemanda,[(SKU,finalBaseP,estoque,sazoBase2,deParaTable,priceSKU,costSKU) for SKU in uniqProd])
    pool.close()
    pool.join()
    #results = []
    #count = 1
    #for SKU in uniqProd:
    #    timeStart = time.time()
    #    results.append(ProjectDemanda(SKU))
    #    print(str(time.time()-timeStart))
    #    print(count)
    #    count += 1

    # 4.2.1 - Consolidate great list:
    teste = np.vstack(results)
    teste = pd.DataFrame(teste)
    teste.columns = ['SKU','Safra','%','Mod52','Mod26','Mod21','Mod17','Mod13','Mod9',
                     'VMDAd','VMDEdu','Vendas','maxFuture','Initial','QtdePontos',
                     'ValidationRupture','flag','flag1','QtdeFinalAd1','QtdeFinalAd2',
                     'QtdeFinalMl','ErroFinal','AbsErroFinal','ErroFinalMl','AbsErroFinalMl',
                     'Cluster','Preco','custo','VendasPerdidasAd','EstoqueExtraAd',
                     'VendasPerdidasMl','EstoqueExtraMl','PerdasTotaisAd','PerdasTotaisMl',
                     'CustoAd','CustoMl']

    # 5- Dynamic Clustering:
    teste.to_csv('C:\\Users\\'+pessoa+'\\Dropbox (ADVISIA)\\201708 Magazine Luiza - Demand Pred\\05 Results\\results20180718 EP.csv',sep=';',index=False)

if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    main()
