import random
import numpy as np
from sympy import Matrix
import copy
from termcolor import colored

NO_NODES=100
NO_GW=5
NO_SIM_ROUNDS=5000
GF = 128
COVERAGE='RAND'   # 'RAND'  or "FULL" or "EQUAL"
ENCODING= 'DFL'      # 'RLC' or 'DFL'
TRAFFIC_PROP=0.01    # A probability of node transmitting every round
NO_GW_COVERAGE=50    # Number of GW reachable by every node (only EQUAL)

# Class to represent data packet
class Packet:   
    def __init__(self, sourceID, data):
        self.sourceID = sourceID
        self.data=data

class MACModel:
    def __init__(self, coverageMode='FULL'):
        self.coverage =[[0]*NO_NODES for _ in range(NO_GW)]
        self.weights=[0]*NO_NODES
       
       # All nodes reach all gateways
        if coverageMode == 'FULL':
            self.coverage =[[1]*NO_NODES for _ in range(NO_GW)]
            self.weights=[NO_GW]*NO_NODES

        # Nodes RANDally distributed under gateways
        elif coverageMode == 'RAND':
            for i in range(NO_NODES):
                ng = random.randint(1,NO_GW)
                reachbleGW=random.sample(range(0, NO_GW), ng)
                for j in reachbleGW:
                    self.coverage[j][i]=1
                    self.weights[i]+=1
        
        # Nodes RANDally distributed under gateways  
        elif coverageMode == 'EQUAL':
            for i in range(NO_NODES):
                reachbleGW=random.sample(range(0, NO_GW), NO_GW_COVERAGE)
                for j in reachbleGW:
                    self.coverage[j][i]=1
                    self.weights[i]+=1

        for i in range(NO_NODES):
            self.weights[i]= abs(NO_NODES-NO_GW)+self.weights[i]
        

    # returns TRUE if node reaches gw,otherwise False.
    def isGWReachable(self, node, gw):
        #print("node=", node, " gw=", gw, "=",self.coverage[gw][node])
        return self.coverage[gw][node]

# Class to store simulation stats
class SimulationStat:
    def __init__(self):
        # Store stats for single run
        self.rawPacketsCount=0
        self.encodedPacketsCount=0
        self.sucessDecodingCount=0

        # Store total stats for all runs
        self.rawPacketsTotal=0
        self.encodedPacketsTotal=0
        self.sucessDecodingTotal=0
        self.unsucessDecodingTotal=0
   
        #Store individual stats for all runs
        self.rawPackets=list()
        self.encodedPackets=list()
        self.decodingResult=list()

        #run stats 
        self.runStats = np.empty((0,8),int)
       

# Class to represent LoRa End Node
class LoraNode:
    def __init__(self, id):
        self.id=id

    def send(self):
        data = random.randint(1,100)
        return Packet(self.id,data)

# Class to represent LoRa Gateway
class LoraGW:
    def __init__(self, id):
        self.id=id
        self.receivedPkts = []

    def receive(self, p):
        self.receivedPkts.append(p)
        simStat.rawPacketsTotal+=1
        simStat.rawPacketsCount+=1

    def sendRaw(self):
        return self.receivedPkts

    def sendEncoded(self,  EG, encoding='RLC'):
        tempPkts = self.receivedPkts.copy()
        E = []
        G = []
        # Using RLC
        if encoding=='RLC':
            while len(tempPkts) !=0:
                e = 0
                g = [0]*NO_NODES

                for _ in range(NO_GW):
                    p = tempPkts.pop()
                    r = random.randint(1,GF-1)
                    e += p.data *r
                    g[p.sourceID]=r
                    if len(tempPkts)==0:
                        break
                simStat.encodedPacketsTotal+=1
                simStat.encodedPacketsCount+=1
                E.append(e)
                G.append(g)
        #Use Louai-
        elif encoding== 'DFL':
            pkts = [0]*NO_NODES
            while len(tempPkts) >0:
                p = tempPkts.pop()
                pkts[p.sourceID]=p.data
            
            EG2 =[]
            for i in range(NO_NODES):
                if EG[i] !=0 and pkts[i]!=0:
                    EG2.append(1)
                else:
                    EG2.append(0)

            toEncodePktCount= EG2.count(1)
            
            for _ in range(toEncodePktCount):
                g=[]
                e=0
                for i in range(NO_NODES):
                    
                    if EG2[i] ==1:
                        r = random.randint(1,GF-1)
                        e+= pkts[i]* r
                        g.append(r)
                    else:
                        g.append(0)

                G.append(g)
                E.append(e)
                simStat.encodedPacketsTotal+=1
                simStat.encodedPacketsCount+=1
            #print(G)
            #print(E)
        return E,G

# Class to represent Network Server
class NetworkServer:
    def __init__(self):
        self.rawPkts = []
        self.encodedPkts = []
        self.encodingCoefficients = []
        self.G = [[0]*NO_NODES for _ in range(NO_NODES)]

    def generateEncoding(self):
        #generate the encoding matrix
        self.G= copy.deepcopy(macModel.coverage)

        for i in range(NO_NODES):
            ones=[]
            for j in range(NO_GW):
                if self.G[j][i]==1:
                    ones.append(j)
            if len(ones) >1:
                c =random.choice(ones)
                ones.remove(c)
                for n in ones:
                   self.G[n][i]=0


    def receiveEncodedPkts(self, E,G):
        self.encodedPkts.extend(E)
        self.encodingCoefficients.extend(G)

    def receiveRawPkts(self, p):
        for pkt in p:
            if pkt not in self.rawPkts:
                self.rawPkts.append(pkt)

    def returnRawPackets(self):
        data = [None]*NO_NODES
        for p in self.rawPkts:
            data[p.sourceID]=p.data
        return(data)

    def confirmDecodedData(self, decodedData):
        data = [None]*NO_NODES
        for p in self.rawPkts:
            data[p.sourceID]=p.data        
        return decodedData == data
            
    def preprocessRREF(self, A):
        M = Matrix(A) 
        return(M.rref())

    def decode(self):
                
        index=0            
        A = np.array(self.encodingCoefficients, dtype=np.int64)
        B = np.array(self.encodedPkts, dtype=np.int64)
        #print(A)
        #print(B)
        map = [None]*NO_NODES
        # remove columns with all zeros
        result = np.all((A == 0), axis=0)
        toBeDeleted=[]
        # #print("Stage 1")
        #print(result)
        for col in range(NO_NODES):
            #print("col=",col)
            if result[col]:
                toBeDeleted.append(col)
            else:
                 map[col]=index
                 index =index+1
        # print("Stage 2")
        A=np.delete(A,toBeDeleted,axis=1)
        # #print("Stage 3")
        # Augment B to A
        #aug = np.append(A,B.reshape(len(B),1),axis=1)
        #print(aug)
        #print("Stage 4")
        # Reduce Augment matrix to REF
        #print(aug.shape)
        #red_aug = self.preprocessRREF(aug)[0]
        #print(red_aug.shape)
        #print("Stage 5")
        #A2 = np.array(aug.tolist(),dtype=np.int64)
        #print(A2)
        #A2 = np.array(red_aug.tolist(),dtype=np.int64)
        #print("Stage 6")
        # remove rows with all zeros
        # result2 = np.all((A2 == 0), axis=1)
        # toBeDeleted2=[]
        # for row in range(A2.shape[0]):
        #     if result2[row]:
        #         toBeDeleted2.append(row)
        # A2=np.delete(A2,toBeDeleted2,axis=0)
        # print(A2.shape)

        #print("Stage 7")
        #Split Augmented matrix into A and B
        #B2 = A2[:,-1]
        #B2 = np.ravel(B2)
        #A2 = np.delete(A2,-1,axis=1)
        
        try:
            #print(np.linalg.matrix_rank(A2))
            X = np.linalg.solve(A,B)
        except:
            X=np.array([])
            print('unsolvable')

        X=np.round(X).astype('int32').tolist()

        j=0
        for i in map:
             if i == None:
                 X.insert(j,None)
             j+=1

        self.encodingCoefficients.clear()
        self.encodedPkts.clear()
        return(X)


#Run Simulation
for NO_NODES in np.arange(100, 1010, 100).tolist()     :                       # Packets Vs. Nodes
    NO_GW= int(NO_NODES *0.05)
    print('Nodes=',NO_NODES, '  NO_GW=',NO_GW)
#for NO_GW_COVERAGE in range(1,NO_GW+2, 2):                      # Packets vs Connectivity
#for TRAFFIC_PROP in np.round(np.arange(0.01, 1, 0.01),2):      # Packets vs TProp


    simStat = SimulationStat()



    #Create Network
    nodes = []
    gateways = []
    netServer = NetworkServer()


    #Generate Coverage Model
    macModel = MACModel(COVERAGE)
    #print(macModel.coverage)

    # Create Nodes
    for i in range(NO_NODES):
        nodes.append(LoraNode(i))

    # Create GWs
    for i in range(NO_GW):
        gateways.append(LoraGW(i))

    sim=0
    while sim < NO_SIM_ROUNDS:
    #for sim in range(NO_SIM_ROUNDS):
        # rest stats
        simStat.rawPacketsCount=0
        simStat.encodedPacketsCount=0
        simStat.sucessDecoding=0

        #if sim%10 == 0:
        #    print('round#',sim)
        #randomly select N nodes to transmit.
        #n = random.randint(1,NO_NODES)
        #print('No of senders=',n)
        #transNodes = random.sample(range(0, NO_NODES), n)
        # Transmit
        #for i in transNodes:
        transmit_nodes=0
        for node in nodes: #Sample nodes according to taffic probability
            if(np.random.choice(np.arange(0,2), p=[1-TRAFFIC_PROP,TRAFFIC_PROP])):
                transmit_nodes+=1
                #print('node ',node.id, " sent")
                pkt = node.send()
                for g in gateways:
                    if macModel.isGWReachable(node.id,g.id):
                        #print("gateway ", g.id, ' got it')
                        g.receive(pkt)
        #TODO: How to Enocde? if binary may not be enough.

        if transmit_nodes == 0:
            continue
        else:
            sim+=1

        #Encode and forward
        netServer.generateEncoding()

        for g in gateways:
            netServer.receiveRawPkts(g.sendRaw())
            netServer.receiveEncodedPkts(*g.sendEncoded(netServer.G[g.id],ENCODING))
            g.receivedPkts.clear() # empty buffers
        
        X = netServer.decode()
        if netServer.confirmDecodedData(X):
            simStat.sucessDecodingTotal+=1
            simStat.sucessDecoding=1
        else:
            simStat.unsucessDecodingTotal+=1
            # print(colored('Unsucessful','red'))
            # print(macModel.coverage)
            # print(netServer.G)
            # quit()

        netServer.rawPkts.clear()
        simStat.rawPackets.append(simStat.rawPacketsCount)
        simStat.encodedPackets.append(simStat.encodedPacketsCount)
        simStat.decodingResult.append(simStat.sucessDecoding)

        simStat.runStats = np.append(simStat.runStats,np.array([[NO_GW,NO_NODES,\
            simStat.rawPacketsCount,\
            simStat.encodedPacketsCount,\
            simStat.sucessDecoding,\
            TRAFFIC_PROP,\
            COVERAGE,\
            NO_GW_COVERAGE]]),axis=0)

    with open("runStats-NCF.csv", "at") as f:
        np.savetxt(f, simStat.runStats, fmt='%s',delimiter=',')



print('raw=',simStat.rawPacketsTotal)
print('encoded=',simStat.encodedPacketsTotal)
print('suc=',simStat.sucessDecodingTotal)
print('uns=',simStat.unsucessDecodingTotal)

