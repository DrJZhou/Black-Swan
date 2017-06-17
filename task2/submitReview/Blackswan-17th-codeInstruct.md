The code framework
=================================

open run.sh run:to produce the file dataImputation.rda
------------------------------------
R CMD BATCH installPackages.R
R CMD BATCH basicPreprocess.R
R CMD BATCH advancedPreprocess.R
R CMD BATCH dataImputation.R

copy file to DLModeling/ and DLModeling/OneByOne for model training
--------------------------------------------------------------------------
cp dataImputation.rda DLModeling
cp dataImputation.rda DLModeling/OneByOne

model training
--------------------------------
R CMD BATCH DLModeling/extendBaseline.R
R CMD BATCH DLModeling/OneByOne/baselineB1.R
R CMD BATCH DLModeling/OneByOne/baselineC1.R
R CMD BATCH DLModeling/OneByOne/baselineC3.R

model result ensemble: copy all 4 files to directory modelStacking, run resultEnsemble to produce final result
-------------------------------------
R CMD BATCH modelStacking/resultEnsemble.R



