set term postscript enhanced color
set output "A-B-C.ps"

set title ""
set xrange [0:1000]
set yrange [0:1600]
set zrange [10:20]
set ztics 3
set pm3d at s
set hidden3d front
set ticslevel 0
set isosamples 10
splot 'data' using 1:($2+0):3 with pm3d  

