import sys
from pylab import * 
import numpy as np

c0  = [0.652038718491, 0.665904403312, 0.666705429585, 0.673930993119, 0.678480915649]
c1 = [0.649899205305, 0.663314062943, 0.663178448261, 0.666452859737, 0.665664495277]
c2 = [0.654783333056, 0.671640204588, 0.670931476267, 0.669954850641, 0.671067090948]
c3 = [0.653149625977, 0.664821818303, 0.664611232361, 0.665313962981, 0.661984905786]
gridd =  [1,2,3,4,5]

nameString = 'UVD model AUCs'
fig = figure(1)
subplot1 = fig.add_subplot(111)
subplot1.plot(gridd, c0, 'b-', label='Normal')
subplot1.plot(gridd, c1, 'r-', label='Coef=1.1')
subplot1.plot(gridd, c2, 'g-', label='Coef=1.2')
subplot1.plot(gridd, c3, 'm-', label='Coef=1.3')
subplot1.set_xlim(0.5, 5.5)
subplot1.grid(True)
xlabel('Setup number')
ylabel('AUC')
title(nameString)
legend(loc=2)
savefig(nameString + '.png')
show()
close()

