class VehicleClass():
    def __init__(self,*argv): #constructor
        #*Argv is a variable number of inputs
        #nargin does not exist (as far as i know)
        #can make by
        #nargin = len(argv)

        import numpy as np
        self.Velocity = np.array([0,0],dtype=float)

        try:
            self.NumPassengers = argv[1]
        except:
            self.NumPassengers = 1

        self.StorageVolume = 0.0
        self.Color = argv[0]
        # the argv usage

    def __add__(V1, V2): #overloads the '+' symbol
        a = VehicleClass(V1.Color, V1.NumPassengers + V2.NumPassengers)
        a.Velocity = V1.Velocity + V2.Velocity
        a.StorageVolume = V1.StorageVolume + V2.StorageVolume
        return a

    def __repr__(self):  #Like Matlab Display Function
        OutputString="""Vehicle with:
        Velocity {0}, 
        Storage Volume {1}, 
        Carries {2} Passangers 
        and is the Color {3}""".format(self.Velocity,
                                       self.StorageVolume,
                                       self.NumPassengers,
                                       self.Color)
        return OutputString

    def accelerate(self, deltaV):
        self.Velocity += deltaV
