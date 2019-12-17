import numpy as np
from numpy import genfromtxt

print(111)
Sigma_Matlab = genfromtxt('C:/Users/Lanston/Desktop/vs/Sigma_from_Matlab.csv', delimiter=',')

print(222)
# inverse calculated by Matlab
invSigma_Matlab = genfromtxt('C:/Users/Lanston/Desktop/vs/inv_Sigma_from_Matlab.csv', delimiter=',')

print(333)
# inverse calculated by Mathematica based on the Sigma_Matlab
invSigma_Mathematica = genfromtxt('C:/Users/Lanston/Desktop/vs/inv_Sigma_from_Mathematica.csv', delimiter=',')

print(444)
# inverse calculated by Numpy based on the Sigma_Matlab
invSigma_Numpy = np.linalg.inv(Sigma_Matlab)

print(555)

print("\n")
print("invSigma_Matlab")
print(invSigma_Matlab[99:105,99:105])

print("\n")
print("invSigma_Mathematica")
print(invSigma_Mathematica[99:105,99:105])

print("\n")
print("invSigma_Numpy")
print(invSigma_Numpy[99:105,99:105])

# Conclusion: invSigma_Mathematica and invSigma_Numpy are more numerically stable than invSigma_Matlab

# save the inverse result of Numpy
# np.savetxt('C:/Users/Lanston/Desktop/vs/inv_Sigma_from_Numpy.csv', invSigma_Numpy, delimiter=',')

# check discrepancies between Mathematica and Numpy, which should be small
mean=np.mean(abs(invSigma_Mathematica - invSigma_Numpy))
mean_percent=np.mean(abs((invSigma_Mathematica - invSigma_Numpy)/invSigma_Numpy))

print(mean)
print(mean_percent)

# =============================================================================
# # Result is as below:
# 
# 111
# 222
# 333
# 444
# 555
# 
# 
# invSigma_Matlab
# [[ 1.0337e+16 -1.2370e+15  3.2908e+15 -3.9745e+15 -7.8996e+14  1.4687e+16]
#  [ 2.2252e+15 -3.3553e+13  1.0002e+14 -5.5002e+13 -6.9257e+13  1.8160e+15]
#  [-3.0064e+15 -1.4786e+14 -2.3670e+14 -3.1408e+14  1.4574e+14 -3.1822e+15]
#  [ 9.8375e+15 -1.2174e+15  2.7639e+15 -3.8727e+15 -9.7735e+14  1.3561e+16]
#  [-1.5132e+15  1.3535e+14 -5.6759e+14  4.1619e+14 -1.6743e+14 -2.5041e+15]
#  [ 1.0673e+16 -6.3501e+14  3.1237e+15 -3.8945e+15 -8.7971e+14  1.4094e+16]]
# 
# 
# invSigma_Mathematica
# [[ 32927.7780845    5044.80130375 -10826.53586437 -11409.42593295
#   -13291.08339083  16502.97116196]
#  [  5044.80144684  -2994.24887904   7589.53133686  16496.62275451
#   -11280.50589794 -14060.40800959]
#  [-10826.53574045   7589.53139514  16329.27807163   9200.20981913
#     1391.45428838 -21213.50226234]
#  [-11409.42586744  16496.62277106   9200.20981193   6680.48592293
#     5322.91248378 -18212.51599248]
#  [-13291.08339195 -11280.50587053   1391.45431113   5322.91252175
#     2278.37167424   -729.70283222]
#  [ 16502.97092759 -14060.40812005 -21213.50228088 -18212.51599022
#     -729.70278882  23000.70054773]]
# 
# 
# invSigma_Numpy
# [[ 32927.77791411   5044.80137162 -10826.53574156 -11409.42581635
#   -13291.08335009  16502.97092921]
#  [  5044.80127419  -2994.24893335   7589.53138303  16496.62278704
#   -11280.50585556 -14060.40810778]
#  [-10826.53584502   7589.53131515  16329.27806751   9200.2098091
#     1391.45430137 -21213.50227408]
#  [-11409.42586212  16496.62273613   9200.20974761   6680.48587706
#     5322.91250082 -18212.51589697]
#  [-13291.08341213 -11280.50590705   1391.45430291   5322.91251071
#     2278.37166376   -729.70280656]
#  [ 16502.97112127 -14060.40802401 -21213.50232186 -18212.5159996
#     -729.70282549  23000.70064189]]
# 0.0001561503647614306
# 1.044086112713871e-07
# =============================================================================
