import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import time as mytime
from mpl_toolkits.mplot3d import Axes3D


class Gravfield:
    """
    Class to model an N-body gravitational system. It will make use of numpy arrays to
    store the names, masses, positions, velocities and accelerations of the bodies we are
    simulating for. The coordinate system origin is at the centre of mass of the entire system.

    Units:
    Masses - kilograms
    Positions - kilometres
    Velocities - kilometres per second
    Accelerations - kilometres per second per second
    """
    names=np.array([])
    masses=np.array([])
    positions=np.array([])
    velocities=np.array([])
    accelerations=np.array([])
    start_time=mytime.time()


    def __init__(self,name,mass,position,velocity):
        
        if not isinstance(name,(str)):
            raise ValueError("name is not a string!")

        if not isinstance (mass, (int, float)):
            raise ValueError("mass is not an int or a float!")

        if not isinstance(position, list):
            if not isinstance(position, tuple):
                raise ValueError("position is not a list or a tuple!")
        if not len(position)==3:
            raise ValueError("position does not contain exactly 3 values!")

        if not isinstance(velocity, list):
            if not isinstance(velocity, tuple):
                raise ValueError("velocity is not a list or a tuple!")
        if not len(velocity)==3:
            raise ValueError("velocity does not contain exactly 3 values!")              
        self.G=6.67*(10**-11)

    def __repr__(self):
        #(f'{self.__class__.__name__}('
           #f'{self.names!r}, {self.masses!r}, {self.positions!r}, {self.velocities!r}, {self.accelerations!r})')
        return 'Body: %10s, Mass: %.5e, Position: %s, Velocity: %s, Acceleration: %s'%(np.array_repr(self.names),np.array_repr(self.masses),np.array_repr(self.positions), np.array_repr(self.velocities),np.array_repr(self.accelerations))  
            
            
       #    Body: %10s, Mass: , Position: , Velocity: , Acceleration: '%(self.names,self.masses,self.positions,self.velocities,self.accelerations)

    def addplanet(self, name, mass, initialPosition, initialVelocity):
        """
        Adds a new body to the system. The formatting of the input must be correct. Make sure name entry is a string and all others are floats.
        name: name assigned to the body being added
        mass: mass of body in kilograms
        initialPosition: initial positional coordinates of the body in kilometres (coordinate origin is the centre of mass of the entire system). Should be an array of length 3
        initialVelocity: initial velocity components of the body in kilometres per second (coordinate origin is the centre of mass of the entire system). Should be an array of length 3
        """

        if not self.names.size==0:
            self.names=np.vstack((self.names,name))
        if self.masses.size==0:
            self.names=np.append(self.names,name)
        if not self.masses.size==0:
            self.masses=np.vstack((self.masses,mass))
        if self.masses.size==0:
            self.masses=np.append(self.masses,mass)
        if not self.positions.size==0:
            self.positions=np.vstack((self.positions,initialPosition))
        if self.positions.size==0:
            self.positions=np.append(self.positions,initialPosition)
        if not self.velocities.size==0:
            self.velocities=np.vstack((self.velocities,initialVelocity))
        if self.velocities.size==0:
            self.velocities=np.append(self.velocities,initialVelocity)
        if not self.accelerations.size==0:
            self.accelerations=np.vstack((self.accelerations,[0,0,0])) 
        if self.accelerations.size==0:
            self.accelerations=np.append(self.accelerations,[0,0,0])
        return


    def testnames(self):
        """Outputs 1xn array of names. For testing purposes to check if names are inputted correctly"""
        return self.names

    def testmasses(self):
        """Outputs 1xn array of masses. For testing purposes to check if masses are inputted correctly"""
        return self.masses

    def testpositions(self):
        """Outputs 3xn array of positions. For testing purposes to check if positions are inputted correctly"""
        return self.positions

    def testvelocities(self):
        """Outputs 3xn array of velocities. For testing purposes to check if velocities are inputted correctly"""
        return self.velocities

    def testaccelerations(self):
        """Outputs 3xn array of positions. For testing purposes to check if accelerations are in correct format.
        Note: if used before any iteration the accelerations will all be initialised to [0,0,0]"""
        return self.accelerations    

    def calculateTotalMomentum(self):
        """Calculates the total momentum of the system at the current point in time of the simulation using p=mv."""
        totalmomentum=np.zeros((3))
        #self.masses=masses()
        for i in range(0,self.masses.size,1):
            for j in range(0,3,1):
                #print("i:",i)
                #print("j:",j)
                #print("name:",self.names[i])
                #print("mass:",self.masses[i])
                #print("velocity:",self.velocities[i][j])
                #print("totalmomentum:",totalmomentum)
                #print("totalmomentum[j]",totalmomentum[j])
                newmomentum=totalmomentum[j]+self.masses[i]*self.velocities[i][j]
                #print("newmomentum:",newmomentum)
                #totalmomentum[j]=newmomentum        
                np.put(totalmomentum,j,newmomentum)    
                #print("totalmomentum:",totalmomentum)   
                #print("magnitude of total momentum:",np.linalg.norm(totalmomentum))
        return totalmomentum

    def calculateTotalAngularMomentum(self):
        """Calculates the total angular momentum of the system"""
        totalangularmomentum=np.zeros((3))
        #self.masses=masses
        for i in range(0,self.masses.size,1):
            #print("i:",i)
            #print("name:",self.names[i])
            #print("position[i]:",self.positions[i])
            position=self.positions[i]
            #print("position:",position)
            #print("velocities[i]:",self.velocities[i])
            velocity=self.velocities[i]
            #print("velocity:",velocity)
            momentum=self.masses[i]*velocity
            #print("momentum:", momentum)
            #for some reason np.cross doesnt work? i suspect an overflow error of some kind
            #print("cross product",np.cross(position,momentum))
            #totalangularmomentum=totalangularmomentum+np.cross(position,momentum)
            newangularmomentumicomponent=position[1]*momentum[2]-position[2]*momentum[1]
            newangularmomentumjcomponent=position[2]*momentum[0]-position[0]*momentum[2]
            newangularmomentumkcomponent=position[0]*momentum[1]-position[1]*momentum[0]
            newtotalangularmomentumicomponent=totalangularmomentum[0]+newangularmomentumicomponent
            newtotalangularmomentumjcomponent=totalangularmomentum[1]+newangularmomentumjcomponent
            newtotalangularmomentumkcomponent=totalangularmomentum[2]+newangularmomentumkcomponent
            np.put(totalangularmomentum,0,newtotalangularmomentumicomponent)
            np.put(totalangularmomentum,1,newtotalangularmomentumjcomponent)
            np.put(totalangularmomentum,2,newtotalangularmomentumkcomponent)
            #print("total angular momentum:",totalangularmomentum)
        return totalangularmomentum

    def eulerIterate(self, time, timestep):
        """
        Using the Euler method; iterates all bodies for a given amount of time and given timestep.
        time: Total amount of time the simulation is run for.
        timestep: Size of the timestep used in the algorithms. Caution: A large time to timestep ratio may take a long time!
        """
        totalT=0.0        
        wtime=np.array([])        
        momentumarray=np.array([])
        momentummagarray=np.array([])
        angmomentumarray=np.array([])
        angmomentmagarray=np.array([])
        xpos=np.array([])
        xvel=np.array([])
        ypos=np.array([])
        yvel=np.array([])
        zpos=np.array([])
        zvel=np.array([])
        newxpos=np.empty([self.masses.size])
        newxvel=np.empty([self.masses.size])
        newypos=np.empty([self.masses.size])
        newyvel=np.empty([self.masses.size])
        newzpos=np.empty([self.masses.size])
        newzvel=np.empty([self.masses.size])
        data=[]
        while(totalT<time):
            for i in range(0,self.masses.size,1):
                for j in range(0,3,1):
                    self.positions[i][j]=self.eulerPosition(Gravfield,self.positions[i][j],self.velocities[i][j],timestep)
                    self.velocities[i][j]=self.eulerVelocity(Gravfield,self.velocities[i][j],self.accelerations[i][j],timestep)
                    
                    if j==0:
                        newxpos[i]=self.positions[i][0]
                        newxvel[i]=self.velocities[i][0]
                    if j==1:
                        newypos[i]=self.positions[i][1]
                        newyvel[i]=self.velocities[i][1]
                    if j==2:    
                        newzpos[i]=self.positions[i][2]
                        newzvel[i]=self.velocities[i][2]
            self.accelerations=self.newAccelerations(Gravfield,self.masses,self.positions)
            totalT=totalT+timestep            
            print("Time:",totalT,)
            #print("Names:\n",self.names)
            #print("masses:\n",self.masses)
            #print("positions:\n",self.positions)
            #print("velocities:\n",self.velocities)
            #print("accelerations:\n",self.accelerations)
            #print("Momentum:",self.calculateTotalMomentum(Gravfield))
            #print("Angular momentum:",self.calculateTotalAngularMomentum(Gravfield))
            #print("Time: %6.3f,xposition: %s,yposition: %s,zposition: %s,xvelocity: %s,yvelocity: %s,zvelocity: %s"%(totalT,newxpos,newypos,newzpos,newxvel,newyvel,newzvel))
            item=[totalT]
            data.append(item)            
            wtime=np.append(wtime,totalT)  
            
            if not momentumarray.size==0:
                momentumarray=np.vstack((momentumarray,self.calculateTotalMomentum(Gravfield)))  
            if momentumarray.size==0:
                momentumarray=np.append(momentumarray,self.calculateTotalMomentum(Gravfield))
            momentummagarray=np.append(momentummagarray,np.linalg.norm(self.calculateTotalMomentum(Gravfield)))            
            if not angmomentumarray.size==0:
                angmomentumarray=np.vstack((angmomentumarray,self.calculateTotalAngularMomentum(Gravfield)))
            if angmomentumarray.size==0:
                angmomentumarray=np.append(angmomentumarray,self.calculateTotalAngularMomentum(Gravfield))    
            angmomentmagarray=np.append(angmomentmagarray,np.linalg.norm(self.calculateTotalAngularMomentum(Gravfield)))
            if not xpos.size==0:
                xpos=np.vstack((xpos,newxpos))
            if xpos.size==0:
                xpos=np.append(xpos,newxpos)            
            if not xvel.size==0:
                xvel=np.vstack((xvel,newxvel))
            if xvel.size==0:
                xvel=np.append(xvel,newxvel)            
            if not ypos.size==0:
                ypos=np.vstack((ypos,newypos))
            if ypos.size==0:
                ypos=np.append(ypos,newypos)            
            if not yvel.size==0:
                yvel=np.vstack((yvel,newyvel))
            if yvel.size==0:
                yvel=np.append(yvel,newyvel)
            if not zpos.size==0:
                zpos=np.vstack((zpos,newzpos))
            if zpos.size==0:
                zpos=np.append(zpos,newzpos)
            if not zvel.size==0:
                zvel=np.vstack((zvel,newzvel))
            if zvel.size==0:
                zvel=np.append(zvel,newzvel)
        
        #print(data)
        print("Final time:",max(wtime))        
        #print(momentumarray)
        #print("xpos:\n",xpos)
        #print("xvel:\n",xvel)
        #print("ypos:\n",ypos)
        #print("yvel:\n",yvel)
        #print("zpos:\n",zpos)
        #print("zvel:\n",zvel)

        fig=plt.figure()  
        plt.subplot()
        ax=fig.gca(projection='3d')    
        for i in range(0,self.masses.size,1):
            labelfori=self.names[i]
            #print("i:",i)
            #print("name:",self.names[i])
            #print("xpos:\n",xpos[:,i])
            #print("xpos[:,i] length:",xpos[:,i].size)
            ax.plot(xpos[:,i],ypos[:,i],zpos[:,i],',-',label=str(labelfori))
        ax.set_title("Paths of Bodies")
        #ax.xlim(xpos.min(),xpos.max())
        #ax.ylim(ypos.min(),ypos.max())
        #ax.zlim(zpos.min(),zpos.max())
        ax.set_xlabel("x-direction")
        ax.set_ylabel("y-direction")
        ax.set_zlabel("z-direction")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()
        #make aspect ratio equal, as far as i can tell this is not built into matplotlib so i found this fix w
        # which builds a box around the data  from: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        max_range = np.array([xpos.max()-xpos.min(),ypos.max()-ypos.min(),zpos.max()-zpos.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xpos.max()+xpos.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ypos.max()+ypos.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zpos.max()+zpos.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        plt.show()

        plt.subplot()
        plt.plot(wtime,momentumarray[:,0], ',-r',label='Total Momentum in x-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,1], ',-g',label='Total Momentum in y-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,2], ',-b',label='Total Momentum in z-Direction of System',linewidth=1)
        plt.title("Change in Momentum in over time")        
        plt.subplot
        plt.plot(wtime,momentummagarray,',-k',label="Magnitude of momentum")
        plt.xlabel("time (s)")
        plt.ylabel("Total Momentum (kgm/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,0],',-r',label='Total Angular Momentum in x-Direction of of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,1],',-g',label='Total Angular Momentum in y-Direction of of System',linewidth=1)        
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,2],',-b',label='Total Angular Momentum in z-Direction of of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentmagarray,',-k',label='Total Magnitude Angular Momentum of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        print("This method simulation took", mytime.time()-self.start_time, "seconds from start to exit")
        plt.show() 

        return


    def eulercromerIterate(self, time, timestep):
        """
        Using the Euler-Cromer method; iterates all bodies for a given amount of time and given timestep.
        time: Total amount of time the simulation is run for.
        timestep: Size of the timestep used in the algorithms. Caution: A large time to timestep ratio may take a long time!
        """
        totalT=0.0        
        wtime=np.array([])        
        momentumarray=np.array([])
        momentummagarray=np.array([])
        angmomentumarray=np.array([])
        angmomentmagarray=np.array([])
        xpos=np.array([])
        xvel=np.array([])
        ypos=np.array([])
        yvel=np.array([])
        zpos=np.array([])
        zvel=np.array([])
        newxpos=np.empty([self.masses.size])
        newxvel=np.empty([self.masses.size])
        newypos=np.empty([self.masses.size])
        newyvel=np.empty([self.masses.size])
        newzpos=np.empty([self.masses.size])
        newzvel=np.empty([self.masses.size])
        data=[]
        while(totalT<time):
            for i in range(0,self.masses.size,1):
                for j in range(0,3,1):
                    self.velocities[i][j]=self.eulerVelocity(Gravfield,self.velocities[i][j],self.accelerations[i][j],timestep)
                    self.positions[i][j]=self.eulerPosition(Gravfield,self.positions[i][j],self.velocities[i][j],timestep)
                    
                    if j==0:
                        newxvel[i]=self.velocities[i][0]
                        newxpos[i]=self.positions[i][0]
                    if j==1:
                        newyvel[i]=self.velocities[i][1]
                        newypos[i]=self.positions[i][1]
                    if j==2:                       
                        newzvel[i]=self.positions[i][2]
                        newzpos[i]=self.positions[i][2]
            self.accelerations=self.newAccelerations(Gravfield,self.masses,self.positions)
            totalT=totalT+timestep            
            print("Time:",totalT,)
            #print("Names:\n",self.names)
            #print("masses:\n",self.masses)
            #print("positions:\n",self.positions)
            #print("velocities:\n",self.velocities)
            #print("accelerations:\n",self.accelerations)
            #print("Momentum:",self.calculateTotalMomentum(Gravfield))
            #print("Angular momentum:",self.calculateTotalAngularMomentum(Gravfield))
            #print("Time: %6.3f,xposition: %s,yposition: %s,zposition: %s,xvelocity: %s,yvelocity: %s,zvelocity: %s"%(totalT,newxpos,newypos,newzpos,newxvel,newyvel,newzvel))
            item=[totalT]
            data.append(item)            
            wtime=np.append(wtime,totalT)  
            
            if not momentumarray.size==0:
                momentumarray=np.vstack((momentumarray,self.calculateTotalMomentum(Gravfield)))  
            if momentumarray.size==0:
                momentumarray=np.append(momentumarray,self.calculateTotalMomentum(Gravfield))
            momentummagarray=np.append(momentummagarray,np.linalg.norm(self.calculateTotalMomentum(Gravfield)))            
            if not angmomentumarray.size==0:
                angmomentumarray=np.vstack((angmomentumarray,self.calculateTotalAngularMomentum(Gravfield)))
            if angmomentumarray.size==0:
                angmomentumarray=np.append(angmomentumarray,self.calculateTotalAngularMomentum(Gravfield))    
            angmomentmagarray=np.append(angmomentmagarray,np.linalg.norm(self.calculateTotalAngularMomentum(Gravfield)))
            if not xpos.size==0:
                xpos=np.vstack((xpos,newxpos))
            if xpos.size==0:
                xpos=np.append(xpos,newxpos)            
            if not xvel.size==0:
                xvel=np.vstack((xvel,newxvel))
            if xvel.size==0:
                xvel=np.append(xvel,newxvel)            
            if not ypos.size==0:
                ypos=np.vstack((ypos,newypos))
            if ypos.size==0:
                ypos=np.append(ypos,newypos)            
            if not yvel.size==0:
                yvel=np.vstack((yvel,newyvel))
            if yvel.size==0:
                yvel=np.append(yvel,newyvel)
            if not zpos.size==0:
                zpos=np.vstack((zpos,newzpos))
            if zpos.size==0:
                zpos=np.append(zpos,newzpos)
            if not zvel.size==0:
                zvel=np.vstack((zvel,newzvel))
            if zvel.size==0:
                zvel=np.append(zvel,newzvel)
        
        #print(data)
        print("Final time:",max(wtime))        
        #print(momentumarray)
        #print("xpos:\n",xpos)
        #print("xvel:\n",xvel)
        #print("ypos:\n",ypos)
        #print("yvel:\n",yvel)
        #print("zpos:\n",zpos)
        #print("zvel:\n",zvel)

        fig=plt.figure()  
        plt.subplot()
        ax=fig.gca(projection='3d')    
        for i in range(0,self.masses.size,1):
            labelfori=self.names[i]
            #print("i:",i)
            #print("name:",self.names[i])
            #print("xpos:\n",xpos[:,i])
            #print("xpos[:,i] length:",xpos[:,i].size)
            ax.plot(xpos[:,i],ypos[:,i],zpos[:,i],',-',label=str(labelfori))
        ax.set_title("Paths of Bodies")
        #ax.xlim(xpos.min(),xpos.max())
        #ax.ylim(ypos.min(),ypos.max())
        #ax.zlim(zpos.min(),zpos.max())
        ax.set_xlabel("x-direction")
        ax.set_ylabel("y-direction")
        ax.set_zlabel("z-direction")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()
        #make aspect ratio equal, as far as i can tell this is not built into matplotlib so i found this fix w
        # which builds a box around the data  from: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        max_range = np.array([xpos.max()-xpos.min(),ypos.max()-ypos.min(),zpos.max()-zpos.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xpos.max()+xpos.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ypos.max()+ypos.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zpos.max()+zpos.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        plt.show()

        plt.subplot()
        plt.plot(wtime,momentumarray[:,0], ',-r',label='Total Momentum in x-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,1], ',-g',label='Total Momentum in y-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,2], ',-b',label='Total Momentum in z-Direction of System',linewidth=1)
        plt.title("Change in Momentum in over time")        
        plt.subplot
        plt.plot(wtime,momentummagarray,',-k',label="Magnitude of momentum")
        plt.xlabel("time (s)")
        plt.ylabel("Total Momentum (kgm/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,0],',-r',label='Total Angular Momentum in x-Direction of of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,1],',-g',label='Total Angular Momentum in y-Direction of of System',linewidth=1)        
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,2],',-b',label='Total Angular Momentum in z-Direction of of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentmagarray,',-k',label='Total Magnitude Angular Momentum of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        print("This method simulation took", mytime.time()-self.start_time, "seconds from start to exit")
        plt.show() 
        return


    def eulerrichardsonIterate(self, time, timestep):
        """
        Using the Euler-Richardson method; iterates all bodies for a given amount of time and given timestep.
        time: Total amount of time the simulation is run for.
        timestep: Size of the timestep used in the algorithms. Caution: A large time to timestep ratio may take a long time!
        """
        totalT=0.0        
        wtime=np.array([])        
        momentumarray=np.array([])
        momentummagarray=np.array([])
        angmomentumarray=np.array([])
        angmomentmagarray=np.array([])
        xpos=np.array([])
        xvel=np.array([])
        ypos=np.array([])
        yvel=np.array([])
        zpos=np.array([])
        zvel=np.array([])
        newxpos=np.empty([self.masses.size])
        newxvel=np.empty([self.masses.size])
        newypos=np.empty([self.masses.size])
        newyvel=np.empty([self.masses.size])
        newzpos=np.empty([self.masses.size])
        newzvel=np.empty([self.masses.size])
        newpositions=np.empty([self.masses.size,3])
        data=[]
        while(totalT<time):
            for i in range(0,self.masses.size,1):
                for j in range(0,3,1):
                    newpositions[i][j]=self.eulerrichardsonPosition(Gravfield,self.positions[i][j],self.velocities[i][j],self.accelerations[i][j],timestep)
                    midposition=self.positions[i][j]+0.5*self.velocities[i][j]*timestep
                    midpositionsarray=copy.deepcopy(self.positions)
                    midpositionsarray[i][j]=midposition
                    newmidaccelerations=self.newAccelerations(Gravfield,self.masses,midpositionsarray)
                    midacceleration=newmidaccelerations[i][j]
                    self.velocities[i][j]=self.eulerrichardsonVelocity(Gravfield,self.velocities[i][j],midacceleration,timestep)
                    self.positions[i][j]=newpositions[i][j]
                    if j==0:
                        newxpos[i]=self.positions[i][0]
                        newxvel[i]=self.velocities[i][0]
                    if j==1:
                        newypos[i]=self.positions[i][1]
                        newyvel[i]=self.velocities[i][1]
                    if j==2:    
                        newzpos[i]=self.positions[i][2]
                        newzvel[i]=self.positions[i][2]
            self.accelerations=self.newAccelerations(Gravfield,self.masses,self.positions)
            totalT=totalT+timestep            
            print("Time:",totalT,)
            #print("Names:\n",self.names)
            #print("masses:\n",self.masses)
            #print("positions:\n",self.positions)
            #print("velocities:\n",self.velocities)
            #print("accelerations:\n",self.accelerations)
            #print("Momentum:",self.calculateTotalMomentum(Gravfield))
            #print("Angular momentum:",self.calculateTotalAngularMomentum(Gravfield))
            #print("Time: %6.3f,xposition: %s,yposition: %s,zposition: %s,xvelocity: %s,yvelocity: %s,zvelocity: %s"%(totalT,newxpos,newypos,newzpos,newxvel,newyvel,newzvel))
            item=[totalT]
            data.append(item)            
            wtime=np.append(wtime,totalT)  
            
            if not momentumarray.size==0:
                momentumarray=np.vstack((momentumarray,self.calculateTotalMomentum(Gravfield)))  
            if momentumarray.size==0:
                momentumarray=np.append(momentumarray,self.calculateTotalMomentum(Gravfield))
            momentummagarray=np.append(momentummagarray,np.linalg.norm(self.calculateTotalMomentum(Gravfield)))            
            if not angmomentumarray.size==0:
                angmomentumarray=np.vstack((angmomentumarray,self.calculateTotalAngularMomentum(Gravfield)))
            if angmomentumarray.size==0:
                angmomentumarray=np.append(angmomentumarray,self.calculateTotalAngularMomentum(Gravfield))    
            angmomentmagarray=np.append(angmomentmagarray,np.linalg.norm(self.calculateTotalAngularMomentum(Gravfield)))
            if not xpos.size==0:
                xpos=np.vstack((xpos,newxpos))
            if xpos.size==0:
                xpos=np.append(xpos,newxpos)            
            if not xvel.size==0:
                xvel=np.vstack((xvel,newxvel))
            if xvel.size==0:
                xvel=np.append(xvel,newxvel)            
            if not ypos.size==0:
                ypos=np.vstack((ypos,newypos))
            if ypos.size==0:
                ypos=np.append(ypos,newypos)            
            if not yvel.size==0:
                yvel=np.vstack((yvel,newyvel))
            if yvel.size==0:
                yvel=np.append(yvel,newyvel)
            if not zpos.size==0:
                zpos=np.vstack((zpos,newzpos))
            if zpos.size==0:
                zpos=np.append(zpos,newzpos)
            if not zvel.size==0:
                zvel=np.vstack((zvel,newzvel))
            if zvel.size==0:
                zvel=np.append(zvel,newzvel)
        
        #print(data)
        print("Final time:",max(wtime))        
        #print(momentumarray)
        #print("xpos:\n",xpos)
        #print("xvel:\n",xvel)
        #print("ypos:\n",ypos)
        #print("yvel:\n",yvel)
        #print("zpos:\n",zpos)
        #print("zvel:\n",zvel)

        fig=plt.figure()  
        plt.subplot()
        ax=fig.gca(projection='3d')    
        for i in range(0,self.masses.size,1):
            labelfori=self.names[i]
            #print("i:",i)
            #print("name:",self.names[i])
            #print("xpos:\n",xpos[:,i])
            #print("xpos[:,i] length:",xpos[:,i].size)
            ax.plot(xpos[:,i],ypos[:,i],zpos[:,i],',-',label=str(labelfori))
        ax.set_title("Paths of Bodies")
        #ax.xlim(xpos.min(),xpos.max())
        #ax.ylim(ypos.min(),ypos.max())
        #ax.zlim(zpos.min(),zpos.max())
        ax.set_xlabel("x-direction")
        ax.set_ylabel("y-direction")
        ax.set_zlabel("z-direction")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()
        #make aspect ratio equal, as far as i can tell this is not built into matplotlib so i found this fix w
        # which builds a box around the data  from: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        max_range = np.array([xpos.max()-xpos.min(),ypos.max()-ypos.min(),zpos.max()-zpos.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xpos.max()+xpos.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ypos.max()+ypos.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zpos.max()+zpos.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        plt.show()

        plt.subplot()
        plt.plot(wtime,momentumarray[:,0], ',-r',label='Total Momentum in x-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,1], ',-g',label='Total Momentum in y-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,2], ',-b',label='Total Momentum in z-Direction of System',linewidth=1)
        plt.title("Change in Momentum in over time")        
        plt.subplot
        plt.plot(wtime,momentummagarray,',-k',label="Magnitude of momentum")
        plt.xlabel("time (s)")
        plt.ylabel("Total Momentum (kgm/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,0],',-r',label='Total Angular Momentum in x-Direction of of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,1],',-g',label='Total Angular Momentum in y-Direction of of System',linewidth=1)        
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,2],',-b',label='Total Angular Momentum in z-Direction of of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentmagarray,',-k',label='Total Magnitude Angular Momentum of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        print("This method simulation took", mytime.time()-self.start_time, "seconds from start to exit")
        plt.show() 
        return


    def verletIterate(self, time, timestep):
        """
        Using the Verlet method; iterates all bodies for a given amount of time and given timestep (Note:uses the Euler method for the new acceleration)
        time: Total amount of time the simulation is run for.
        timestep: Size of the timestep used in the algorithms. Caution: A large time to timestep ratio may take a long time!
        """
        totalT=0.0        
        wtime=np.array([])        
        momentumarray=np.array([])
        momentummagarray=np.array([])
        angmomentumarray=np.array([])
        angmomentmagarray=np.array([])
        xpos=np.array([])
        xvel=np.array([])
        ypos=np.array([])
        yvel=np.array([])
        zpos=np.array([])
        zvel=np.array([])
        newxpos=np.empty([self.masses.size])
        newxvel=np.empty([self.masses.size])
        newypos=np.empty([self.masses.size])
        newyvel=np.empty([self.masses.size])
        newzpos=np.empty([self.masses.size])
        newzvel=np.empty([self.masses.size])
        data=[]
        while(totalT<time):
            for i in range(0,self.masses.size,1):
                for j in range(0,3,1):
                    self.positions[i][j]=self.verletPosition(Gravfield,self.positions[i][j],self.velocities[i][j],self.accelerations[i][j],timestep)
                    newpositions=copy.deepcopy(self.positions)
                    newpositions[i][j]=self.eulerPosition(Gravfield,self.positions[i][j],self.velocities[i][j],timestep)
                    newacceleration=self.newAccelerations(Gravfield,self.masses,newpositions)
                    self.velocities[i][j]=self.verletVelocity(Gravfield,self.velocities[i][j],self.accelerations[i][j],newacceleration[i][j],timestep)
                    if j==0:
                        newxpos[i]=self.positions[i][0]
                        newxvel[i]=self.velocities[i][0]
                    if j==1:
                        newypos[i]=self.positions[i][1]
                        newyvel[i]=self.velocities[i][1]
                    if j==2:    
                        newzpos[i]=self.positions[i][2]
                        newzvel[i]=self.positions[i][2]
            self.accelerations=self.newAccelerations(Gravfield,self.masses,self.positions)
            totalT=totalT+timestep            
            print("Time:",totalT,)
            #print("Names:\n",self.names)
            #print("masses:\n",self.masses)
            #print("positions:\n",self.positions)
            #print("velocities:\n",self.velocities)
            #print("accelerations:\n",self.accelerations)
            #print("Momentum:",self.calculateTotalMomentum(Gravfield))
            #print("Angular momentum:",self.calculateTotalAngularMomentum(Gravfield))
            #print("Time: %6.3f,xposition: %s,yposition: %s,zposition: %s,xvelocity: %s,yvelocity: %s,zvelocity: %s"%(totalT,newxpos,newypos,newzpos,newxvel,newyvel,newzvel))
            item=[totalT]
            data.append(item)            
            wtime=np.append(wtime,totalT)  
            
            if not momentumarray.size==0:
                momentumarray=np.vstack((momentumarray,self.calculateTotalMomentum(Gravfield)))  
            if momentumarray.size==0:
                momentumarray=np.append(momentumarray,self.calculateTotalMomentum(Gravfield))
            momentummagarray=np.append(momentummagarray,np.linalg.norm(self.calculateTotalMomentum(Gravfield)))            
            if not angmomentumarray.size==0:
                angmomentumarray=np.vstack((angmomentumarray,self.calculateTotalAngularMomentum(Gravfield)))
            if angmomentumarray.size==0:
                angmomentumarray=np.append(angmomentumarray,self.calculateTotalAngularMomentum(Gravfield))    
            angmomentmagarray=np.append(angmomentmagarray,np.linalg.norm(self.calculateTotalAngularMomentum(Gravfield)))
            if not xpos.size==0:
                xpos=np.vstack((xpos,newxpos))
            if xpos.size==0:
                xpos=np.append(xpos,newxpos)            
            if not xvel.size==0:
                xvel=np.vstack((xvel,newxvel))
            if xvel.size==0:
                xvel=np.append(xvel,newxvel)            
            if not ypos.size==0:
                ypos=np.vstack((ypos,newypos))
            if ypos.size==0:
                ypos=np.append(ypos,newypos)            
            if not yvel.size==0:
                yvel=np.vstack((yvel,newyvel))
            if yvel.size==0:
                yvel=np.append(yvel,newyvel)
            if not zpos.size==0:
                zpos=np.vstack((zpos,newzpos))
            if zpos.size==0:
                zpos=np.append(zpos,newzpos)
            if not zvel.size==0:
                zvel=np.vstack((zvel,newzvel))
            if zvel.size==0:
                zvel=np.append(zvel,newzvel)
        
        #print(data)
        print("Final time:",max(wtime))        
        #print(momentumarray)
        #print("xpos:\n",xpos)
        #print("xvel:\n",xvel)
        #print("ypos:\n",ypos)
        #print("yvel:\n",yvel)
        #print("zpos:\n",zpos)
        #print("zvel:\n",zvel)

        fig=plt.figure()  
        plt.subplot()
        ax=fig.gca(projection='3d')    
        for i in range(0,self.masses.size,1):
            labelfori=self.names[i]
            #print("i:",i)
            #print("name:",self.names[i])
            #print("xpos:\n",xpos[:,i])
            #print("xpos[:,i] length:",xpos[:,i].size)
            ax.plot(xpos[:,i],ypos[:,i],zpos[:,i],',-',label=str(labelfori))
        ax.set_title("Paths of Bodies")
        #ax.xlim(xpos.min(),xpos.max())
        #ax.ylim(ypos.min(),ypos.max())
        #ax.zlim(zpos.min(),zpos.max())
        ax.set_xlabel("x-direction")
        ax.set_ylabel("y-direction")
        ax.set_zlabel("z-direction")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()
        #make aspect ratio equal, as far as i can tell this is not built into matplotlib so i found this fix w
        # which builds a box around the data  from: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        max_range = np.array([xpos.max()-xpos.min(),ypos.max()-ypos.min(),zpos.max()-zpos.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xpos.max()+xpos.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ypos.max()+ypos.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zpos.max()+zpos.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        plt.show()

        plt.subplot()
        plt.plot(wtime,momentumarray[:,0], ',-r',label='Total Momentum in x-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,1], ',-g',label='Total Momentum in y-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,2], ',-b',label='Total Momentum in z-Direction of System',linewidth=1)
        plt.title("Change in Momentum in over time")        
        plt.subplot
        plt.plot(wtime,momentummagarray,',-k',label="Magnitude of momentum")
        plt.xlabel("time (s)")
        plt.ylabel("Total Momentum (kgm/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,0],',-r',label='Total Angular Momentum in x-Direction of of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,1],',-g',label='Total Angular Momentum in y-Direction of of System',linewidth=1)        
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,2],',-b',label='Total Angular Momentum in z-Direction of of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentmagarray,',-k',label='Total Magnitude Angular Momentum of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        print("This method simulation took", mytime.time()-self.start_time, "seconds from start to exit")
        plt.show() 
        return


    def rungekuttaIterate(self, time, timestep):
        """
        Using the runge-kutta (RK4) method; iterates all bodies for a given amount of time and given timestep
        time: Total amount of time the simulation is run for.
        timestep: Size of the timestep used in the algorithms. Caution: A large time to timestep ratio may take a long time!
        """
        totalT=0.0        
        wtime=np.array([])        
        momentumarray=np.array([])
        momentummagarray=np.array([])
        angmomentumarray=np.array([])
        angmomentmagarray=np.array([])
        xpos=np.array([])
        xvel=np.array([])
        ypos=np.array([])
        yvel=np.array([])
        zpos=np.array([])
        zvel=np.array([])
        newxpos=np.empty([self.masses.size])
        newxvel=np.empty([self.masses.size])
        newypos=np.empty([self.masses.size])
        newyvel=np.empty([self.masses.size])
        newzpos=np.empty([self.masses.size])
        newzvel=np.empty([self.masses.size])
        data=[]
        while(totalT<time):
            for i in range(0,self.masses.size,1):
                for j in range(0,3,1):
                   
                    acceleration2=copy.deepcopy(self.accelerations)
                    positionarray2=copy.deepcopy(self.positions)
                    positionarray2[i][j]=self.positions[i][j]+self.accelerations[i][j]*timestep/2
                    newacceleration2=self.newAccelerations(Gravfield,self.masses,positionarray2)
                    acceleration2[i][j]=newacceleration2[i][j]

                    acceleration3=copy.deepcopy(self.accelerations)
                    positionarray3=copy.deepcopy(self.positions)
                    positionarray3[i][j]=self.positions[i][j]+acceleration2[i][j]*timestep/2
                    newacceleration3=self.newAccelerations(Gravfield,self.masses,positionarray3)
                    acceleration3[i][j]=newacceleration3[i][j]

                    acceleration4=copy.deepcopy(self.accelerations)
                    positionarray4=copy.deepcopy(self.positions)
                    positionarray4[i][j]=self.positions[i][j]+acceleration3[i][j]*timestep
                    newacceleration4=self.newAccelerations(Gravfield,self.masses,positionarray4)
                    acceleration4[i][j]=newacceleration4[i][j]

                    self.velocities[i][j]=self.rungekuttaVelocity(Gravfield,self.velocities[i][j],self.accelerations[i][j],acceleration2[i][j],acceleration3[i][j],acceleration4[i][j],timestep)
                    self.positions[i][j]=self.rungekuttaPosition(Gravfield,self.positions[i][j],self.velocities[i][j],timestep)
                    if j==0:
                        newxpos[i]=self.positions[i][0]
                        newxvel[i]=self.velocities[i][0]
                    if j==1:
                        newypos[i]=self.positions[i][1]
                        newyvel[i]=self.velocities[i][1]
                    if j==2:    
                        newzpos[i]=self.positions[i][2]
                        newzvel[i]=self.positions[i][2]
            self.accelerations=self.newAccelerations(Gravfield,self.masses,self.positions)
            totalT=totalT+timestep            
            print("Time:",totalT,)
            item=[totalT]
            data.append(item)            
            wtime=np.append(wtime,totalT)  
            
            if not momentumarray.size==0:
                momentumarray=np.vstack((momentumarray,self.calculateTotalMomentum(Gravfield)))  
            if momentumarray.size==0:
                momentumarray=np.append(momentumarray,self.calculateTotalMomentum(Gravfield))
            momentummagarray=np.append(momentummagarray,np.linalg.norm(self.calculateTotalMomentum(Gravfield)))            
            if not angmomentumarray.size==0:
                angmomentumarray=np.vstack((angmomentumarray,self.calculateTotalAngularMomentum(Gravfield)))
            if angmomentumarray.size==0:
                angmomentumarray=np.append(angmomentumarray,self.calculateTotalAngularMomentum(Gravfield))    
            angmomentmagarray=np.append(angmomentmagarray,np.linalg.norm(self.calculateTotalAngularMomentum(Gravfield)))
            if not xpos.size==0:
                xpos=np.vstack((xpos,newxpos))
            if xpos.size==0:
                xpos=np.append(xpos,newxpos)            
            if not xvel.size==0:
                xvel=np.vstack((xvel,newxvel))
            if xvel.size==0:
                xvel=np.append(xvel,newxvel)            
            if not ypos.size==0:
                ypos=np.vstack((ypos,newypos))
            if ypos.size==0:
                ypos=np.append(ypos,newypos)            
            if not yvel.size==0:
                yvel=np.vstack((yvel,newyvel))
            if yvel.size==0:
                yvel=np.append(yvel,newyvel)
            if not zpos.size==0:
                zpos=np.vstack((zpos,newzpos))
            if zpos.size==0:
                zpos=np.append(zpos,newzpos)
            if not zvel.size==0:
                zvel=np.vstack((zvel,newzvel))
            if zvel.size==0:
                zvel=np.append(zvel,newzvel)
        print("Final time:",max(wtime))       
        fig=plt.figure()  
        plt.subplot()
        ax=fig.gca(projection='3d')    
        for i in range(0,self.masses.size,1):
            labelfori=self.names[i]
            ax.plot(xpos[:,i],ypos[:,i],zpos[:,i],',-',label=str(labelfori))
        ax.set_title("Paths of Bodies")
        ax.set_xlabel("x-direction")
        ax.set_ylabel("y-direction")
        ax.set_zlabel("z-direction")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()
        #make aspect ratio equal, as far as i can tell this is not built into matplotlib so i found this fix w
        # which builds a box around the data  from: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        max_range = np.array([xpos.max()-xpos.min(),ypos.max()-ypos.min(),zpos.max()-zpos.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xpos.max()+xpos.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ypos.max()+ypos.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zpos.max()+zpos.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        plt.show()

        plt.subplot()
        plt.plot(wtime,momentumarray[:,0], ',-r',label='Total Momentum in x-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,1], ',-g',label='Total Momentum in y-Direction of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,momentumarray[:,2], ',-b',label='Total Momentum in z-Direction of System',linewidth=1)
        plt.title("Change in Momentum in over time")        
        plt.subplot
        plt.plot(wtime,momentummagarray,',-k',label="Magnitude of momentum")
        plt.xlabel("time (s)")
        plt.ylabel("Total Momentum (kgm/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,0],',-r',label='Total Angular Momentum in x-Direction of of System',linewidth=1)
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,1],',-g',label='Total Angular Momentum in y-Direction of of System',linewidth=1)        
        plt.subplot()
        plt.plot(wtime,angmomentumarray[:,2],',-b',label='Total Angular Momentum in z-Direction of of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        plt.show()

        plt.subplot()
        plt.plot(wtime,angmomentmagarray,',-k',label='Total Magnitude Angular Momentum of System',linewidth=1)
        plt.xlabel("time (s)")
        plt.ylabel("Total Angular Momentum in System (kgm^2/s)")
        plt.legend()
        print("This method simulation took", mytime.time()-self.start_time, "seconds from start to exit")
        plt.show() 
        return

    def newAccelerations(self, masses, positions):
        """
        Finds acceleration for all bodies and returns the accelerations as a new array to be used.
        Does not replace old array by default.
        masses: masses of the bodies in kilograms
        positions: positions of the bodies in kilometres in array of size N x 3
        """
        G=6.67*(10**-11)
        newaccelerations=copy.deepcopy(self.accelerations)
        for i in range(0,masses.size,1):
            for j in range(0,3,1):
                for k in range(0,masses.size,1):
                    if(i!=k):
                        rdistance=np.linalg.norm((positions[i]-positions[k]))
                        newaccel=(-G*masses[k]*(positions[i]-positions[k])/(rdistance**3))
                        newaccelerations[i]=newaccel
        return newaccelerations

    def eulerVelocity(self, velocity, acceleration, timestep):
        """
        Euler method of finding velocity.
        velocity: the component of velocity to find the new value for
        acceleration: component of acceleration for the velocity to be found
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        newvelocity=velocity+acceleration*timestep
        return newvelocity

    def eulerPosition(self, position, velocity, timestep):
        """
        Euler method of finding position.
        position: component of the position to find the new value for
        velocity: component of velocity used to find the new position component
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        newposition=position+velocity*timestep
        return newposition

    def eulercromerVelocity(self, velocity, acceleration, timestep):
        """
        Euler-Cromer method of finding velocity.
        velocity: the component of velocity to find the new value for
        acceleration: component of acceleration for the velocity to be found
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        newvelocity=velocity+acceleration*timestep
        return newvelocity

    def eulercromerPosition(self, position, newvelocity, timestep):
        """
        Euler-Cromer method of finding position
        position: component of position to find the new value for
        newvelocity: component of velocity (at end of step(v_n+1)) used to find new position component
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        newposition=position+newvelocity*timestep
        return newposition

    def eulerrichardsonVelocity(self, velocity, midacceleration, timestep):
        """
        Euler-Richardson method of finding velocity.
        position: complete position array needed to find the new value of velocity
        velocity: the component of velocity to find the new value for
        acceleration: component of acceleration for the velocity to be found
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        #midposition=self.eulerPosition(position, velocity, 0.5*timestep)
        #midacceleration=self.newAccelerations(masses,midposition,acceleration)
        newvelocity=velocity+midacceleration*timestep
        return newvelocity

    def eulerrichardsonPosition(self, position, velocity, acceleration, timestep):
        """
        Euler-Richardson method of finding position.
        position: component of position to find the new value for
        velocity: component of velocity used to find the new position component
        acceleration: component of initial acceleration used to find the new position component
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        #midvelocity=self.eulerVelocity(velocity, acceleration, 0.5*timestep)
        midvelocity=velocity+0.5*acceleration*timestep
        newposition=position+midvelocity*timestep
        return newposition

    def verletVelocity(self, velocity, acceleration, newaccel, timestep):
        """
        Verlet method of finding velocity.
        velocity: the component of velocity to find the new value for
        acceleration: component of initial acceleration for the velocity to be found
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        newvelocity=velocity+0.5*timestep*(acceleration+newaccel)
        return newvelocity

    def verletPosition(self, position, velocity, acceleration, timestep):
        """
        Verlet method of finding position.
        position: component of position to find the new value for
        velocity: component of velocity used to find the new position component
        acceleration: component of initial acceleration used to find the new position component
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        newposition=position+velocity*timestep+0.5*acceleration*timestep*timestep
        return newposition

    def rungekuttaVelocity(self, velocity, acceleration, acceleration2, acceleration3, acceleration4, timestep):
        """
        Runge-Kutta (RK4) method of finding velocity.
        position: complete position array to find new accelerations used to find new velocities
        velocity: the component of velocity to find the new value for
        acceleration: component of acceleration for the velocity to be found
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        kv1=acceleration       
        kv2=acceleration2
        kv3=acceleration3
        kv4=acceleration4
        newvelocity=velocity+timestep*(kv1+2*kv2+2*kv3+kv4)/6
        return newvelocity

    def rungekuttaPosition(self, position, velocity, timestep):
        """
        Runge-Kutta (RK4) method of finding position.
        position: component of position to find the new value for
        velocity: component of velocity used to find the new position component
        timestep: Timestep we are using to find the next set of position vectors and velocity vectors
        """
        kx1=velocity
        kx2=velocity*kx1*timestep*0.5
        kx3=velocity*kx2*timestep*0.5
        kx4=velocity*kx3*timestep
        newposition=position+timestep*(kx1+2*kx2+2*kx3+kx4)/6
        return newposition
