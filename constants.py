#This folder contains *.py files which save features and targets lists. The file name are like "userID-k-interval.py"
PATH_SALIDA_LISTA_FEATURES = "OCEAN-outputs/FeaturesAndTargetList/"

inputFileWithUsersIDs = open("preprocessed files/IDsUsersOCEAN.txt", "r") # files with users' ID.

"""
    Experiments:        
        - 32
            k = 1   ** DONE
            k = 2   ** DONE
        - 1024
            k = 1   ** DONE
            k = 2   **
        - Mod2-1-1 
            k = 1   ** DONE
            k = 2   ** DONE
        - Mod2-2-1 
            k = 1   ** DONE 
            k = 2   ** DONE
        - Mod2-3-1
            k = 1   ** DONE
            k = 2   ** DONE
        - Mod2-4-1
            k = 1   ** DONE  
            k = 2   ** DONE
       . . . . . . . . 
        - Mod2-1-2
            k = 1   ** DONE
            k = 2   ** DONE
        - Mod2-2-2
            k = 1   ** DONE
            k = 2   ** DONE
        - Mod2-3-2
            k = 1   ** DONE
            k = 2   ** DONE
        - Mod2-4-2
            k = 1   ** DONE
            k = 2   ** DONE
    
    
    -----------------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------------
    Second group choosing higher trait with higher deviation to Big-2 and OCEAN classes from 1 to 4
    ...............................................................................................
        - DesvioSinNyV
            k = 1  ** DONE
            k = 2  ** DONE
        
        - DesvioMod2-1-2
            k = 1  ** DONE
            k = 2  ** DONE
        - DesvioMod2-2-2
            k = 1  ** DONE
            k = 2  ** DONE
        - DesvioMod2-3-2
            k = 1  ** DONE
            k = 2  ** DONE
        - DesvioMod2-4-2
            k = 1  ** DONE
            k = 2  ** 
"""

OCEAN_TYPE = "DesvioMod2-4-2"
kThatRemenber = [2]

OCEAN_TYPE_STRING = "(tipoNoDefinido)"
if OCEAN_TYPE == "32":
    OCEAN_TYPE_STRING = "OCEAN-1-32"
elif OCEAN_TYPE == "1024":
    OCEAN_TYPE_STRING = "OCEAN-1-1024"
elif OCEAN_TYPE == "Mod2-1-1":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-1-1"
elif OCEAN_TYPE == "Mod2-2-1":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-2-1"
elif OCEAN_TYPE == "Mod2-3-1":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-3-1"
elif OCEAN_TYPE == "Mod2-4-1":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-4-1"
elif OCEAN_TYPE == "Mod2-1-2":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-1-2"
elif OCEAN_TYPE == "Mod2-2-2":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-2-2"
elif OCEAN_TYPE == "Mod2-3-2":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-3-2"
elif OCEAN_TYPE == "Mod2-4-2":
    OCEAN_TYPE_STRING = "OCEAN-1-16-Mod2-4-2"
elif OCEAN_TYPE == "DesvioSinNyV":
    OCEAN_TYPE_STRING = "OCEAN-1-4-DesvioSinNyV"
elif OCEAN_TYPE == "DesvioMod2-1-2":
    OCEAN_TYPE_STRING = "OCEAN-1-4-Mod2-1-2"
elif OCEAN_TYPE == "DesvioMod2-2-2":
    OCEAN_TYPE_STRING = "OCEAN-1-4-Mod2-2-2"
elif OCEAN_TYPE == "DesvioMod2-3-2":
    OCEAN_TYPE_STRING = "OCEAN-1-4-Mod2-3-2"
elif OCEAN_TYPE == "DesvioMod2-4-2":
    OCEAN_TYPE_STRING = "OCEAN-1-4-Mod2-4-2"