import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.35  # the width of the bars
#plt.style.use(['science'])
header_list = ['m', 'n', 'raw','encoded','success','tprop','connectivity','fw_gw']

df = pd.read_csv('runStats-NCF.csv',names=header_list)
df_fig1= df[(df['tprop']==0.5)& (df['connectivity']=='RAND')& (df['success']==1)]

network_sizes = [100,300,500,700,1000]
raw_avg = list()
raw_err = list()
enc_avg = list()
enc_err = list()

for i in network_sizes:
    dfTemp= df_fig1[df_fig1['n']==i]
    raw_avg.append(dfTemp['raw'].mean())
    raw_err.append(dfTemp['raw'].std())
    enc_avg.append(dfTemp['encoded'].mean())
    enc_err.append(dfTemp['encoded'].std())

r1 = np.arange(len(network_sizes))
r2 = [x + barWidth for x in r1]


fig_packets_vs_size=plt.figure(1)
plt.rc('font', family='serif')
plt.bar(r1, raw_avg, yerr=raw_err,color='gray', hatch="//",width=barWidth, edgecolor='white', label='LoRaWAN')
plt.bar(r2, enc_avg, yerr=enc_err,color='red', hatch="", width=barWidth, edgecolor='white', label='NCF')


plt.title('Forwarded Packets vs. Network Size')
plt.xlabel('Netwrok Size ($n$)', fontweight='bold')
plt.ylabel('No. of Packets per Generation', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(network_sizes))], network_sizes)
 
# Create legend & Show graphic
plt.legend(prop={'size': 15})

raw = list()
encoded = list()
raw_err = list()
encoded_err = list()
tprop = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

df_fig2= df[(df['n']==100)& (df['m']==5)& (df['connectivity']=='RAND')& (df['success']==1)]

for tp in tprop:
    dfTemp= df_fig2[df_fig2['tprop']==tp]
    raw.append(dfTemp['raw'].mean())
    raw_err.append(dfTemp['raw'].std())

    encoded.append(dfTemp['encoded'].mean())
    encoded_err.append(dfTemp['encoded'].std())

   
fig_packets_vs_tprop=plt.figure(2)
plt.rc('font', family='serif')

plt.plot(tprop,raw,'--',label='LoRaWAN',color='black')
#plt.errorbar(tprop,raw,yerr=raw_err,color='black')
plt.plot(tprop,encoded,'-+',label='NCF', color='red')
plt.title("Forwarded Packets vs. Transmission Probability")
plt.xlabel('Transmission Probability ($p_t$)', fontweight='bold')
plt.ylabel('No. of Packets per Generation', fontweight='bold')
 
# Create legend & Show graphic
plt.legend()

df_fig3= df[(df['tprop']==0.01)& (df['connectivity']=='RAND')& (df['success']==1)]

raw = []
encoded = []
sizes = np.arange(100, 1010, 100).tolist()

raw=list()
encoded=list()
for sz in sizes:
    df_sz= df_fig3[df_fig3['n']==sz]
    raw.append(df_sz['raw'].mean())
    encoded.append(df_sz['encoded'].mean())

fig_packets_vs_size_duty=plt.figure(3)
plt.rc('font', family='serif')

plt.plot(sizes,raw,'--',label='LoRaWAN',color='black')
plt.plot(sizes,encoded,'-+', label='NCF', color='red')

plt.title("Forwarded Packets vs. Network Size - Low Duty Cycle")
plt.xlabel('Network Size ($n$)', fontweight='bold')
plt.ylabel('No. of Packets per Generation', fontweight='bold')
 
# Create legend & Show graphic
plt.legend()

raw = list()
encoded = list()
fw_gw= list(range(5, 55, 5))

df_fig4= df[(df['n']==1000)&(df['tprop']==0.5)& (df['connectivity']=='EQUAL')& (df['success']==1)]

for gw in fw_gw:
    df_gw= df_fig4[df['fw_gw']==gw]
    raw.append(df_gw['raw'].mean())
    encoded.append(df_gw['encoded'].mean())

fig_packets_vs_connectivity=plt.figure(4)

plt.rc('font', family='serif')
plt.plot(fw_gw,raw, '-',label='LoRaWAN', color='black')
plt.plot(fw_gw,encoded,'-+', label='NCF', color='red' )
plt.title("Forwarded Packets vs. Connectivity ")
plt.xlabel('Connectivity Factor ($w$)', fontweight='bold')
plt.ylabel('No. of Packets per Generation', fontweight='bold')
 
plt.legend()


plt.show()
