from Gravfield import Gravfield
import math
import numpy as np
import time
start_time=time.time()

"this class hold all planet initial info including name mass position velocity acceleration"
#DATA TAKEN FROM https://ssd.jpl.nasa.gov/horizons.cgi#results 2018-NOV-21 00:00:00.0000 TDB

#SUN DATA:
sunpos=np.array([-6.545790*(10**4),1.098244*(10**6),-9.787565*(10**3)]) 
sunvel=np.array([-1.330409*(10**(-2)),3.950885*(10**(-3)),3.332900*(10**(-4))])
Gravfield.addplanet(Gravfield,"Sun",1988500*(10**24),sunpos,sunvel)
#MERCURY DATA:
merpos=np.array([4.367991*(10**7),2.323932*(10**7),-2.213683*(10**6)])
mervel=np.array([-3.151102*(10**1),4.561638*(10**1),6.617025*(10**0)])
Gravfield.addplanet(Gravfield,"Mercury",3.302*(10**23),merpos,mervel)
#VENUS DATA:
venpos=np.array([3.032726*(10**7),1.045425*(10**8),-3.443039*(10**5)])
venvel=np.array([-3.373087*(10**1),9.708600*(10**0),2.079253*(10**0)])
Gravfield.addplanet(Gravfield,"Venus",48.685*(10**23),venpos,venvel)
#EARTH DATA:
earpos=np.array([7.747157*(10**7),1.269188*(10**8),-1.531971*(10**4)])
earvel=np.array([-2.585205*(10**1),1.550995*(10**1),-3.854601*(10**-4)])
Gravfield.addplanet(Gravfield,"Earth",5.972*(10**24),earpos,earvel)
#MARS DATA:
marpos=np.array([2.015822*(10**8),6.335599*(10**7),-3.653223*(10**6)])
marvel=np.array([-6.235924*(10**0),2.522533*(10**1),6.815145*(10**-1)])
Gravfield.addplanet(Gravfield,"Mars",6.417*(10**23),marpos,marvel)
#JUPITER DATA:
juppos=np.array([-3.607318*(10**8),-7.152400*(10**8),1.103552*(10**7)])
jupvel=np.array([1.150949*(10**1),-5.260275*(10**0),-2.356136*(10**-1)])
Gravfield.addplanet(Gravfield,"Jupiter",1898*(10**24),juppos,jupvel)
#SATURN DATA:
satpos=np.array([2.612890*(10**8),-1.481187*(10**9),1.535317*(10**7)])
satvel=np.array([8.980822*(10**0),1.646269*(10**0),-3.855444*(10**-1)])
Gravfield.addplanet(Gravfield,"Saturn",5.6834*(10**26),satpos,satvel)
#URANUS DATA:
urapos=np.array([2.557999*(10**9),1.513870*(10**9),-2.751670*(10**7)])
uravel=np.array([-3.518567*(10**0),5.543201*(10**0),6.624814*(10**-2)])
Gravfield.addplanet(Gravfield,"Uranus",86.813*(10**24),urapos,uravel)
#NEPTUNE DATA:
neppos=np.array([4.330866*(10**9),-1.137919*(10**9),-7.637590*(10**7)])
nepvel=np.array([1.345189*(10**0),5.288525*(10**0),-1.405763*(10**-1)])
Gravfield.addplanet(Gravfield,"Neptune",102.41*(10**24),neppos,nepvel)
#PLUTO DATA:
plupos=np.array([1.759027*(10**9),-4.721131*(10**9),-3.623700*(10**6)])
pluvel=np.array([5.225199*(10**0),7.518022*(10**-1),-1.608957*(10**0)])
Gravfield.addplanet(Gravfield,"Pluto",1.31*(10**22),plupos,pluvel)

#check if inputs are correct
print("Names:\n", Gravfield.testnames(Gravfield))
print("Masses:\n", Gravfield.testmasses(Gravfield))
print("Positions:\n",Gravfield.testpositions(Gravfield))
print("Velocities:\n",Gravfield.testvelocities(Gravfield))
print("Accelerations:\n",Gravfield.testaccelerations(Gravfield))
#run code
#no. of seconds in year= 60*60*24*365=31536000
totaltime=31536000
timestep=10000
Gravfield.verletIterate(Gravfield,totaltime,timestep)

print("This program took", time.time()-start_time, "from start to exit")