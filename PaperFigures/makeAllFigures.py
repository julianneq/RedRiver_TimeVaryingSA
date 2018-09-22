from makeFigure3 import makeFigure3
from makeFigure4_S2 import makeFigure4_S2
from makeFigure5 import makeFigure5
from makeFigure6_S4_S5 import makeFigure6_S4_S5
from makeFigure7_S7 import makeFigure7_S7
from makeFigure8_S8 import makeFigure8_S8
from makeFigureS1 import makeFigureS1
from makeFigureS3_S6 import makeFigureS3_S6

def makeAllFigures():
    # Figure 1 same as Figure 2 of Quinn et al., 2017 - WRR (Rival Framings)
    # Figure 2 made manually in Illustrator
    makeFigure3() # example of negative interactions
    makeFigure4_S2() # Pareto set from optimization (4) and re-evaluation (S2)
    makeFigure5() # time-varying and state-space PDFs
    makeFigure6_S4_S5() # compares analytical uHB sensitivities of different policies (6); numerical uHB (S4); numerical rHB (S5)
    makeFigure7_S7() # compares analytical uHB and numerical rHB sensitivity of compromise to guidelines rHB sensitivity w/ different deltas (7 and S7)
    makeFigure8_S8() # compares analytical u and numerical r sensitivity of compromise across reservoirs to guidelines w/ different deltas (8 and S8)
    makeFigureS1() # guidelines rule curves
    makeFigureS3_S6() # state trajectories associated with Figure 6 sensitivities (S3) and difference between u and r (S6)
    
    return None